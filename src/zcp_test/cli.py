from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from zcp_test.artifacts import (
    JsonlWriter,
    RunContext,
    read_jsonl,
    read_score_records,
    score_component,
)
from zcp_test.artifacts.run import file_sha256
from zcp_test.acceptance import freeze_training_candidates, measure_architecture_resources
from zcp_test.benchmarks import BENCHMARKS, load_builtin_benchmarks
from zcp_test.config import TRAIN_PROFILE_KEYS, load_config, reject_unknown_config_keys
from zcp_test.doctor import diagnostics
from zcp_test.data import (
    DataAsset,
    DataRegistry,
    bootstrap_benchmarks,
    convert_vitbench101,
    convert_imagenet16_120,
    data_checklist,
    export_data_manifest,
    verify_data_manifest,
    vitbench101_release_parser,
)
from zcp_test.data.setup import runtime_catalog_contract
from zcp_test.gpu import (
    GPULockError,
    NoGPUError,
    configure_cuda,
    enumerate_gpus,
    gpu_lock,
    gpu_lock_status,
    select_gpu,
)
from zcp_test.inputs import CandidateInputResolver, make_dataset_batch_stream
from zcp_test.legacy import import_pickle
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.evaluator import evaluate_proxy
from zcp_test.research import create_sample_manifest, load_sample_indices
from zcp_test.reporting import correlation_summary, curve_plot, jsonl_to_csv, static_html
from zcp_test.reporting.analysis import (
    build_report_bundle,
    correlation_table,
    plot_search,
    plot_sensitivity,
    plot_sensitivity_rank,
    plot_training,
    proxy_cost_pareto,
    rank_aggregation,
    read_scores,
    sample_size_convergence,
    sensitivity_rank_table,
    top_k_comparison,
    transfer_correlation_table,
    validate_analysis_scores,
)
from zcp_test.reporting.benchmark_budget import nasbench101_budget_study
from zcp_test.reporting.benchmark_darts import nasbench301_darts_study
from zcp_test.reporting.benchmark_report import write_benchmark_study
from zcp_test.reporting.benchmark_studies import (
    nats_size_study,
    topology_study,
    transnas_transfer_study,
    vit_architecture_study,
)
from zcp_test.reporting.monitor import refresh_once
from zcp_test.search import (
    EvolutionSearch,
    PlainNetSourceAlignedSearch,
    cache_key,
    load_plainnet_search_state,
    load_search_state,
    resolve_target_profile,
    validate_search_state_identity,
)
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.training import TrainingConfig, train_model
from zcp_test.types import MetricSpec, ModelFidelity


def _json(value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    try:
        print(payload, flush=True)
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _evaluation_status_summary(statuses: list[str]) -> dict[str, int]:
    counts = {
        status: statuses.count(status) for status in ("ok", "failed", "unsupported", "skipped")
    }
    return {
        "succeeded": counts["ok"],
        "failed": counts["failed"],
        "unsupported": counts["unsupported"],
        "skipped": counts["skipped"],
        "non_ok": len(statuses) - counts["ok"],
    }


def _args_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "function" and not key.startswith("_")
    }


def _argument_was_explicit(args: argparse.Namespace, name: str) -> bool:
    return name in getattr(args, "_explicit_options", ())


def _candidate_input_resolver(args: argparse.Namespace, device: Any) -> CandidateInputResolver:
    return CandidateInputResolver(
        source=args.input_source,
        dataset=args.dataset,
        batch_size=args.batch_size,
        requested_input_size=args.input_size,
        classes=args.classes,
        seed=args.seed,
        device=device,
        data_root=_resolve_data_root(args, args.dataset),
        explicit_input_size=_argument_was_explicit(args, "input_size"),
    )


def _space_provenance(space: Any) -> dict[str, Any]:
    return {
        "model_fidelity": getattr(space, "model_fidelity", ModelFidelity.PROXY_APPROXIMATION.value),
        "implementation_source": getattr(space, "implementation_source", None),
        "implementation_commit": getattr(space, "implementation_commit", None),
        "model_profile": getattr(space, "model_profile", None),
    }


def _research_model_provenance(space: Any, adapter: Any = None) -> dict[str, Any]:
    provenance = _space_provenance(space)
    if (
        adapter is not None
        and adapter.benchmark_id == "vitbench101"
        and space.search_space_id == "autoformer"
    ):
        provenance.update(
            implementation_source="https://github.com/lliai/Auto-Prox-AAAI24",
            implementation_commit="90ed458eff6948a6f0d23e440a8d21bbec50d091",
            model_profile="vitbench-autoprox-90ed458",
        )
    return provenance


def _seed_training(
    seed: int,
    rank: int,
    deterministic: bool,
    *,
    cudnn_benchmark: bool = False,
    allow_tf32: bool = False,
) -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic and cudnn_benchmark:
        raise ValueError("cudnn_benchmark is incompatible with deterministic training")
    torch.backends.cudnn.benchmark = cudnn_benchmark if not deterministic else False
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    rank_seed = (int(seed) + int(rank)) % (2**32)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    return {
        "base_seed": int(seed),
        "rank_seed": rank_seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _search_model_seed(seed: int, architecture_id: str) -> int:
    payload = f"zcp-test-search-model-v1:{int(seed)}:{architecture_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _seed_search_model(seed: int, architecture_id: str) -> int:
    import random

    import numpy as np
    import torch

    model_seed = _search_model_seed(seed, architecture_id)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    return model_seed


def _require_research_model(
    space: Any, allow_approximation: bool, adapter: Any = None
) -> dict[str, Any]:
    provenance = _research_model_provenance(space, adapter)
    if (
        provenance["model_fidelity"]
        in {ModelFidelity.PROXY_APPROXIMATION.value, ModelFidelity.METRIC_ONLY.value}
        and not allow_approximation
    ):
        raise RuntimeError(
            f"Search space {space.search_space_id!r} has model fidelity "
            f"{provenance['model_fidelity']!r}; pass --allow-approximation only for an "
            "explicitly labelled methodological ablation"
        )
    return provenance


def _prepare_model_weights(args: argparse.Namespace, space: Any) -> tuple[Any, dict[str, Any]]:
    weight_mode = getattr(args, "weight_mode", "independent_scratch")
    if weight_mode == "independent_scratch":
        return None, {"weight_mode": weight_mode}
    if weight_mode != "ofa_inherited" or space.search_space_id != "ofa_proxyless_mbv2":
        raise ValueError("ofa_inherited weight mode is supported only for ofa_proxyless_mbv2")
    checkpoint = getattr(args, "model_checkpoint", None)
    if checkpoint is None:
        try:
            checkpoint = DataRegistry(args.catalog).get("ofa_proxyless_supernet").path
        except KeyError as error:
            raise FileNotFoundError(
                "OFA supernet is not registered; run `zcp-test data bootstrap --root "
                "/path/to/data --benchmarks ofa_proxyless_supernet --yes` or pass "
                "--model-checkpoint"
            ) from error
    from zcp_test.models.mobile import OFAProxylessCheckpoint

    checkpoint_loader = OFAProxylessCheckpoint(
        checkpoint,
        trusted=getattr(args, "trusted", False),
        expected_sha256="10ce40eec63dd020b4fa0096b1ff3c1e81e5b740446ddef6a59651bb36e6b907",
    )
    return checkpoint_loader, {
        "weight_mode": "inherited_supernet",
        **checkpoint_loader.source,
        "bn_recalibration_required": True,
        "bn_recalibrated_batches": 0,
    }


def _prepare_bn_recalibration(
    args: argparse.Namespace,
    device: Any,
    weight_loader: Any,
    *,
    input_size: int | None = None,
) -> Any:
    batches = int(getattr(args, "bn_recalibration_batches", 0))
    if batches == 0:
        return None
    if weight_loader is None:
        raise ValueError("BN recalibration is available only with --weight-mode ofa_inherited")
    data_root = _resolve_data_root(args, args.dataset)
    if not data_root:
        raise ValueError("BN recalibration requires a registered dataset or explicit --data-root")
    batch_size = int(getattr(args, "bn_recalibration_batch_size", None) or args.batch_size)
    return make_dataset_batch_stream(
        args.dataset,
        data_root,
        batch_size,
        args.input_size if input_size is None else input_size,
        args.seed + 10_000,
        batches,
        device,
        role="ofa_bn_recalibration",
    )


def _transnas_unsupported_reason(benchmark_id: str | None, task: str, proxy_id: str) -> str | None:
    if benchmark_id != "transnasbench101":
        return None
    load_builtin_proxies()
    capability = PROXIES.create(proxy_id).capability
    if capability.requires_labels and task not in {
        "class_scene",
        "class_object",
        "jigsaw",
    }:
        return (
            f"{proxy_id} requires labels/loss, but the {task} Taskonomy label protocol "
            "is not implemented"
        )
    return None


def _infer_target_direction(metric_name: str | None) -> str:
    normalized = str(metric_name or "").casefold().replace("-", "_")
    if any(token in normalized for token in ("neg_loss", "negative_loss")):
        return "maximize"
    if any(token in normalized for token in ("loss", "error", "time", "latency")):
        return "minimize"
    return "maximize"


def _resolve_data_root(args: argparse.Namespace, dataset: str) -> str | None:
    if getattr(args, "data_root", None):
        return str(args.data_root)
    catalog = getattr(args, "catalog", None)
    if not catalog:
        return None
    registry = DataRegistry(catalog)
    normalized_dataset = {
        "cifar10-valid": "cifar10",
        "ImageNet16-120": "imagenet16_120",
    }.get(dataset, dataset)
    asset_ids = [f"dataset_{normalized_dataset}"]
    from zcp_test.data.transnas_inputs import is_transnas_task

    if is_transnas_task(dataset):
        asset_ids.append("dataset_transnas_taskonomy")
    for asset_id in asset_ids:
        try:
            asset = registry.get(asset_id)
        except KeyError:
            continue
        verification = registry.verify(asset_id)
        if not verification["valid"]:
            raise FileNotFoundError(f"Registered dataset asset is invalid: {asset_id}")
        path = Path(asset.path).expanduser()
        return str(path.parent if dataset == "ImageNet16-120" and path.is_file() else path)
    return None


def _resolve_benchmark_path(args: argparse.Namespace) -> str:
    if args.benchmark_path:
        path = Path(args.benchmark_path).expanduser()
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Explicit benchmark path does not exist: {path}")
    catalog_ids = {
        "nasbench101": "nasbench101",
        "nasbench201": "nasbench201",
        "nats_tss": "nats_tss",
        "nats_sss": "nats_sss",
        "nasbench301_surrogate": "nasbench301_surrogate_0",
        "transnasbench101": "transnasbench101_0"
        if args.transnas_space == "micro"
        else "transnasbench101_1",
        "vitbench101": {
            "autoformer_main": "vitbench101_0",
            "autoformer_ext": "vitbench101_1",
            "pit": "vitbench101_2",
        }.get(args.slice_id, "vitbench101_0"),
    }
    asset_id = catalog_ids[args.benchmark]
    suffix = asset_id.rpartition("_")[2]
    asset_index = int(suffix) if suffix.isdigit() else 0
    try:
        expected_version, expected_protocol = runtime_catalog_contract(args.benchmark, asset_index)
        asset = DataRegistry(args.catalog).get_verified(
            asset_id,
            expected_version=expected_version,
            expected_protocol=expected_protocol,
        )
        return asset.path
    except KeyError:
        pass
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"Registered benchmark asset {asset_id!r} failed catalog verification: {error}"
        ) from error
    root = args.data_root or os.environ.get("ZCP_DATA_ROOT", "/path/to/data")
    raise FileNotFoundError(
        f"{args.benchmark} standard-answer data is unavailable. Run: "
        f"zcp-test data checklist --root {root} && "
        f"zcp-test data bootstrap --root {root} --benchmarks {args.benchmark}"
    )


def _resolve_nb301_runtime_path(args: argparse.Namespace) -> str | None:
    if args.runtime_benchmark_path:
        path = Path(args.runtime_benchmark_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"NAS-Bench-301 runtime ensemble does not exist: {path}")
        return str(path)
    try:
        asset = DataRegistry(args.catalog).get("nasbench301_surrogate_1")
    except (KeyError, FileNotFoundError, ValueError):
        return None
    path = Path(asset.path).expanduser()
    return str(path) if path.exists() else None


def _device(name: str) -> Any:
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


@contextmanager
def _selected_device(args: argparse.Namespace):
    requested_device = getattr(args, "device", None)
    if requested_device:
        yield _device(requested_device), None
        return
    selection = _prepare_gpu(args)
    candidates = [selection]
    auto_selection = str(getattr(args, "gpu", "auto")).casefold() == "auto"
    if auto_selection:
        remaining = [gpu for gpu in enumerate_gpus() if str(gpu["uuid"]) != str(selection["uuid"])]
        while remaining:
            try:
                alternative = select_gpu(
                    "auto",
                    model=getattr(args, "gpu_model", None),
                    min_free_mb=getattr(args, "min_free_memory", 0),
                    gpus=remaining,
                )
            except NoGPUError:
                break
            candidates.append(alternative)
            remaining = [gpu for gpu in remaining if str(gpu["uuid"]) != str(alternative["uuid"])]
    lock_timeout = getattr(args, "gpu_lock_timeout", 0.0)
    if lock_timeout is not None and lock_timeout < 0:
        raise ValueError("gpu_lock_timeout must be non-negative or None")

    def try_lock(candidate: dict[str, Any], timeout: float | None):
        stack = ExitStack()
        try:
            stack.enter_context(gpu_lock(candidate, timeout=timeout))
        except GPULockError as error:
            stack.close()
            return None, error
        return stack, None

    acquired_stack = None
    acquired_candidate = None
    last_error = None
    if auto_selection:
        for candidate in candidates:
            acquired_stack, last_error = try_lock(candidate, 0.0)
            if acquired_stack is not None:
                acquired_candidate = candidate
                break
        if acquired_stack is None and (lock_timeout is None or lock_timeout > 0):
            deadline = None if lock_timeout is None else time.monotonic() + lock_timeout
            round_start = 1 % len(candidates)
            while acquired_stack is None:
                remaining_timeout = None if deadline is None else deadline - time.monotonic()
                if remaining_timeout is not None and remaining_timeout <= 0:
                    break
                time.sleep(0.1 if remaining_timeout is None else min(0.1, remaining_timeout))
                for offset in range(len(candidates)):
                    candidate = candidates[(round_start + offset) % len(candidates)]
                    acquired_stack, last_error = try_lock(candidate, 0.0)
                    if acquired_stack is not None:
                        acquired_candidate = candidate
                        break
                round_start = (round_start + 1) % len(candidates)
    else:
        acquired_stack, last_error = try_lock(selection, lock_timeout)
        if acquired_stack is not None:
            acquired_candidate = selection

    if acquired_stack is None or acquired_candidate is None:
        message = (
            "All matching GPUs are locked by other zcp-test processes"
            if auto_selection
            else "The selected GPU is locked by another zcp-test process"
        )
        raise GPULockError(message) from last_error

    with acquired_stack:
        configured = configure_cuda(acquired_candidate)
        configured["selection_strategy"] = "auto" if auto_selection else "explicit"
        configured["nvidia_smi_index"] = configured["index"]
        configured["torch_logical_index"] = 0
        args._gpu_selection = configured
        device = _device("cuda:0")
        yield device, configured


def _prepare_gpu(args: argparse.Namespace) -> dict[str, Any] | None:
    if getattr(args, "device", None):
        return None
    existing = getattr(args, "_gpu_selection", None)
    if existing is not None:
        return existing
    selection = select_gpu(
        getattr(args, "gpu", "auto"),
        model=getattr(args, "gpu_model", None),
        min_free_mb=getattr(args, "min_free_memory", 0),
    )
    args._gpu_selection = selection
    return selection


@contextmanager
def _training_device(
    args: argparse.Namespace,
    world_size: int,
    rank: int,
    local_rank: int,
):
    if world_size == 1:
        with _selected_device(args) as selected:
            yield selected
        return
    if getattr(args, "device", None):
        raise ValueError("Distributed training derives cuda:LOCAL_RANK; do not pass --device")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Distributed training requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if len(visible_tokens) < world_size:
        raise RuntimeError(
            "Distributed training requires CUDA_VISIBLE_DEVICES to list at least WORLD_SIZE "
            "launcher-managed GPUs, preferably by UUID"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError("LOCAL_RANK is outside the visible CUDA device range")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    selection = {
        "selection_strategy": "torchrun_launcher_managed",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "torch_logical_index": local_rank,
        "visible_devices": visible_tokens,
        "selected_visible_device": visible_tokens[local_rank],
    }
    try:
        yield torch.device("cuda", local_rank), selection
    finally:
        torch.distributed.destroy_process_group()


@contextmanager
def _training_signal_handlers():
    previous = {}

    def interrupt(signum: int, frame: Any) -> None:
        del frame
        raise InterruptedError(f"Training interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _checkpoint_lineage(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    source_manifest = path.parent.parent / "manifest.json"
    source_run_id = None
    if source_manifest.is_file():
        try:
            source_run_id = json.loads(source_manifest.read_text(encoding="utf-8")).get("run_id")
        except (json.JSONDecodeError, OSError):
            source_run_id = None
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "source_run_id": source_run_id,
    }


@contextmanager
def _training_run_context(
    root: str | Path,
    command: list[str],
    config: dict[str, Any],
    runtime: dict[str, Any],
    world_size: int,
    rank: int,
):
    if world_size == 1:
        with _training_signal_handlers():
            with RunContext(root, command, config, runtime=runtime) as run:
                yield run
        return
    import torch

    primary_context = None
    payload: list[Any] = [None]
    if rank == 0:
        primary_context = RunContext(root, command, config, runtime=runtime)
        primary_context.__enter__()
        payload[0] = {
            "directory": str(primary_context.directory),
            "run_id": primary_context.run_id,
        }
    torch.distributed.broadcast_object_list(payload, src=0)
    shared = payload[0]
    if not isinstance(shared, dict):
        raise RuntimeError("Rank zero did not broadcast a shared training run directory")
    run = primary_context or SimpleNamespace(
        directory=Path(shared["directory"]),
        run_id=shared["run_id"],
    )
    caught = None
    try:
        with _training_signal_handlers():
            yield run
    except BaseException as error:
        caught = error
        raise
    finally:
        if primary_context is not None:
            primary_context.__exit__(
                None if caught is None else type(caught),
                caught,
                None,
            )


def _add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu",
        default="auto",
        help="auto, nvidia-smi index, GPU UUID, PCI bus ID, or model substring",
    )
    parser.add_argument("--gpu-model")
    parser.add_argument("--min-free-memory", type=int, default=0, metavar="MIB")
    parser.add_argument("--gpu-lock-timeout", type=float, default=0.0)
    parser.add_argument(
        "--device",
        help="advanced compatibility override such as cpu or cuda:0; bypasses GPU selection",
    )


def _synthetic_loader(batch_size: int, input_size: int, classes: int, batches: int = 2) -> Any:
    import torch

    dataset = torch.utils.data.TensorDataset(
        torch.randn(batch_size * batches, 3, input_size, input_size),
        torch.randint(classes, (batch_size * batches,)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def _imagenet_transforms(input_size: int, config: dict[str, Any]) -> tuple[Any, Any]:
    from torchvision import transforms

    class AlexNetLighting:
        def __init__(self) -> None:
            import torch

            self.eigenvalues = torch.tensor([0.2175, 0.0188, 0.0045])
            self.eigenvectors = torch.tensor(
                [
                    [-0.5675, 0.7192, 0.4009],
                    [-0.5808, -0.0045, -0.8140],
                    [-0.5836, -0.6948, 0.4203],
                ]
            )

        def __call__(self, image: Any) -> Any:
            alpha = image.new_empty(3).normal_(0, 0.1)
            noise = (
                self.eigenvectors.to(image)
                * alpha.view(1, 3)
                * self.eigenvalues.to(image).view(1, 3)
            ).sum(1)
            return image + noise.view(3, 1, 1)

    if config.get("auto_augment") or config.get("random_erase_probability"):
        from timm.data import create_transform

        train_transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=float(config.get("color_jitter", 0.4)),
            auto_augment=config.get("auto_augment"),
            interpolation=str(config.get("train_interpolation", "bicubic")),
            re_prob=float(config.get("random_erase_probability", 0.0)),
            re_mode=str(config.get("random_erase_mode", "pixel")),
            re_count=int(config.get("random_erase_count", 1)),
        )
    else:
        distortion = str(config.get("color_distortion", "torch")).casefold()
        if distortion in {"tf", "tensorflow"}:
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0,
                saturation=0.5,
            )
        elif distortion == "aznas_imagenet":
            color_transform = transforms.ColorJitter(0.4, 0.4, 0.4)
        elif distortion in {"torch", "strong"}:
            color_transform = transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
        elif distortion in {"none", "null", "false"}:
            color_transform = None
        else:
            raise ValueError(f"Unsupported ImageNet color_distortion {distortion!r}")
        crop_arguments: dict[str, Any] = {
            "scale": (float(config.get("resize_scale", 0.08)), 1.0)
        }
        if distortion == "aznas_imagenet":
            crop_arguments["interpolation"] = transforms.InterpolationMode.BICUBIC
        train_steps: list[Any] = [
            transforms.RandomResizedCrop(input_size, **crop_arguments),
            transforms.RandomHorizontalFlip(),
        ]
        if color_transform is not None:
            train_steps.append(color_transform)
        train_steps.extend(
            [
                transforms.ToTensor(),
            ]
        )
        if distortion == "aznas_imagenet":
            train_steps.append(AlexNetLighting())
        train_steps.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        train_transform = transforms.Compose(train_steps)
    valid_transform = transforms.Compose(
        [
            transforms.Resize(math.ceil(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, valid_transform


def _real_loaders(
    dataset: str,
    root: str,
    batch_size: int,
    input_size: int,
    workers: int,
    config: dict[str, Any],
    fraction: float = 1.0,
    seed: int = 42,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> tuple[Any, Any]:
    import torch
    from torchvision import datasets, transforms

    if dataset in {"cifar10", "cifar100"}:
        statistics = {
            "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        }
        mean, standard_deviation = statistics[dataset]
        train_steps: list[Any] = [
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
        if int(config.get("cutout_length", 0)):
            train_steps.append(Cutout(int(config["cutout_length"])))
        train_transform = transforms.Compose(train_steps)
        valid_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, standard_deviation)]
        )
        dataset_type = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        train_set = dataset_type(root, train=True, transform=train_transform, download=False)
        valid_set = dataset_type(root, train=False, transform=valid_transform, download=False)
    else:
        train_transform, valid_transform = _imagenet_transforms(input_size, config)
        root_path = Path(root)
        train_set = datasets.ImageFolder(root_path / "train", train_transform)
        validation_directory = (
            root_path / "val" if (root_path / "val").exists() else root_path / "test"
        )
        valid_set = datasets.ImageFolder(validation_directory, valid_transform)
    train_set = _stratified_subset(train_set, fraction, seed)
    valid_set = _stratified_subset(valid_set, fraction, seed + 1)
    if distributed_world_size <= 0:
        raise ValueError("distributed_world_size must be positive")
    if not 0 <= distributed_rank < distributed_world_size:
        raise ValueError("distributed_rank must be within distributed_world_size")
    generator = torch.Generator().manual_seed(seed)
    if workers < 0:
        raise ValueError("workers must be non-negative")
    valid_workers = int(config.get("valid_workers", workers))
    if valid_workers < 0:
        raise ValueError("valid_workers must be non-negative")
    train_common = {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)),
        "persistent_workers": workers > 0 and bool(config.get("persistent_workers", True)),
    }
    if workers > 0:
        prefetch_factor = int(config.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        train_common["prefetch_factor"] = prefetch_factor
    valid_common = {
        "num_workers": valid_workers,
        "pin_memory": bool(config.get("pin_memory", True)),
        "persistent_workers": valid_workers > 0
        and bool(config.get("valid_persistent_workers", config.get("persistent_workers", True))),
    }
    if valid_workers > 0:
        valid_prefetch_factor = int(
            config.get("valid_prefetch_factor", config.get("prefetch_factor", 2))
        )
        if valid_prefetch_factor <= 0:
            raise ValueError("valid_prefetch_factor must be positive")
        valid_common["prefetch_factor"] = valid_prefetch_factor
    train_sampler = None
    if bool(config.get("repeated_augmentation", False)):
        from timm.data.distributed_sampler import RepeatAugSampler

        train_sampler = RepeatAugSampler(
            train_set,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=True,
            num_repeats=int(config.get("repeated_augmentation_repeats", 3)),
            selected_round=int(config.get("repeated_augmentation_selected_round", 256)),
            selected_ratio=int(config.get("repeated_augmentation_selected_ratio", 0)),
        )
    elif distributed_world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_set,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=True,
            seed=seed,
        )
    valid_sampler = None
    if distributed_world_size > 1:
        valid_sampler = torch.utils.data.DistributedSampler(
            valid_set,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=False,
            seed=seed + 1,
        )
    return (
        torch.utils.data.DataLoader(
            train_set,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=generator,
            batch_size=batch_size,
            **train_common,
        ),
        torch.utils.data.DataLoader(
            valid_set,
            shuffle=False,
            sampler=valid_sampler,
            batch_size=int(config.get("test_batch_size", batch_size)),
            **valid_common,
        ),
    )


def _stratified_subset(dataset: Any, fraction: float, seed: int) -> Any:
    import math
    import random
    import torch

    if not 0 < fraction <= 1:
        raise ValueError("data fraction must be in (0, 1]")
    if fraction == 1:
        return dataset
    targets = getattr(dataset, "targets", None)
    if targets is None and hasattr(dataset, "samples"):
        targets = [sample[1] for sample in dataset.samples]
    if targets is None:
        raise ValueError("stratified subset requires dataset class targets")
    groups: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        groups.setdefault(int(target), []).append(index)
    rng = random.Random(seed)
    if len(targets) != len(dataset):
        raise ValueError("dataset class targets must match dataset length")
    target_size = round(len(targets) * fraction)
    if target_size == 0:
        raise ValueError("data fraction rounds to an empty subset")
    class_order = list(groups)
    rng.shuffle(class_order)
    quotas: dict[int, int] = {}
    remainders: dict[int, float] = {}
    for class_id, indices in groups.items():
        ideal = len(indices) * target_size / len(targets)
        quotas[class_id] = math.floor(ideal)
        remainders[class_id] = ideal - quotas[class_id]
    remaining = target_size - sum(quotas.values())
    tie_order = {class_id: index for index, class_id in enumerate(class_order)}
    allocation_order = sorted(
        groups,
        key=lambda class_id: (-remainders[class_id], tie_order[class_id]),
    )
    for class_id in allocation_order:
        if remaining == 0:
            break
        if quotas[class_id] < len(groups[class_id]):
            quotas[class_id] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("failed to allocate the requested stratified subset size")
    selected: list[int] = []
    for class_id, indices in groups.items():
        rng.shuffle(indices)
        selected.extend(indices[: quotas[class_id]])
    if len(selected) != target_size:
        raise RuntimeError("stratified subset size does not match its global target")
    return torch.utils.data.Subset(dataset, sorted(selected))


class Cutout:
    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, image: Any) -> Any:
        import torch

        height, width = image.shape[1:]
        center_y = int(torch.randint(height, (1,)).item())
        center_x = int(torch.randint(width, (1,)).item())
        half = self.length // 2
        image[
            :,
            max(0, center_y - half) : min(height, center_y + half),
            max(0, center_x - half) : min(width, center_x + half),
        ] = 0
        return image


def command_doctor(args: argparse.Namespace) -> None:
    report = diagnostics(args.catalog)
    if args.data_root:
        report["benchmark_data"] = data_checklist(args.data_root, args.catalog)
    _json(report)


def command_gpu(args: argparse.Namespace) -> None:
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    rows = enumerate_gpus()
    for pci_order, row in enumerate(rows):
        row["pci_order"] = pci_order
        row["visible_logical_index"] = (
            visible.index(row["uuid"]) if row["uuid"] in visible else None
        )
        row["zcp_test_lock"] = gpu_lock_status(row)
    _json(rows)


def command_proxy(args: argparse.Namespace) -> None:
    if args.action in {"list", "inspect", "matrix", "validate"}:
        load_builtin_proxies()
    if args.action == "list":
        _json(PROXIES.names())
        return
    if args.action == "inspect":
        _json(PROXIES.create(args.name).capability.__dict__)
        return
    if args.action == "matrix":
        _json([PROXIES.create(name).capability.__dict__ for name in PROXIES.names()])
        return
    if args.action == "scaffold":
        _scaffold_proxy(args.name)
        return
    _validate_proxy(args.name)


def _scaffold_proxy(name: str) -> None:
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError("Proxy name must be a public Python identifier")
    root = Path(__file__).resolve().parents[1]
    module = root / "zcp_test" / "proxies" / "custom" / f"{name}.py"
    test = root.parent / "tests" / f"test_proxy_{name}.py"
    if module.exists() or test.exists():
        raise FileExistsError(f"Custom proxy {name!r} already exists")
    module.write_text(
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from zcp_test.proxies import PROXIES\n"
        "from zcp_test.proxies.builtin import FunctionProxy\n"
        "from zcp_test.types import ProxyCapability\n\n\n"
        f"def compute_{name}(model: Any, inputs: Any, labels: Any, loss_fn: Any) -> float:\n"
        '    raise NotImplementedError("implement the proxy formula")\n\n\n'
        f'CAPABILITY = ProxyCapability("{name}", version="custom-v1")\n'
        f'PROXIES.register("{name}", lambda: FunctionProxy(CAPABILITY, compute_{name}))\n',
        encoding="utf-8",
    )
    test.write_text(
        "import torch\n\n"
        "from zcp_test.proxies.evaluator import evaluate_proxy\n\n\n"
        f"def test_{name}_contract():\n"
        "    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))\n"
        f'    result = evaluate_proxy("{name}", model, torch.randn(2, 3, 2, 2))\n'
        '    assert result.status.value == "ok"\n',
        encoding="utf-8",
    )
    _json({"module": str(module), "test": str(test), "next": f"zcp-test proxy validate {name}"})


def _validate_proxy(name: str) -> None:
    import random

    import numpy as np
    import torch

    proxy = PROXIES.create(name)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    )
    inputs = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    state = {key: value.clone() for key, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    gradient_flags = [parameter.requires_grad for parameter in model.parameters()]
    hooks = sum(
        len(module._forward_hooks) + len(module._backward_hooks) for module in model.modules()
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    result = evaluate_proxy(
        name,
        model,
        inputs,
        labels,
        torch.nn.CrossEntropyLoss(),
    )
    unchanged = all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    modes_unchanged = modes == [module.training for module in model.modules()]
    gradient_flags_unchanged = gradient_flags == [
        parameter.requires_grad for parameter in model.parameters()
    ]
    hooks_after = sum(
        len(module._forward_hooks) + len(module._backward_hooks) for module in model.modules()
    )
    python_rng_unchanged = python_rng == random.getstate()
    numpy_after = np.random.get_state()
    numpy_rng_unchanged = (
        numpy_rng[0] == numpy_after[0]
        and np.array_equal(numpy_rng[1], numpy_after[1])
        and numpy_rng[2:] == numpy_after[2:]
    )
    torch_rng_unchanged = torch.equal(torch_rng, torch.get_rng_state())
    primary_component_matches = result.primary_component == proxy.capability.primary_component
    report = {
        "proxy": name,
        "status": result.status.value,
        "score": result.score,
        "primary_component": result.primary_component,
        "components": result.components,
        "model_state_unchanged": unchanged,
        "model_modes_unchanged": modes_unchanged,
        "gradient_flags_unchanged": gradient_flags_unchanged,
        "hooks_clean": hooks == hooks_after,
        "python_rng_unchanged": python_rng_unchanged,
        "numpy_rng_unchanged": numpy_rng_unchanged,
        "torch_rng_unchanged": torch_rng_unchanged,
        "primary_component_matches_capability": primary_component_matches,
        "capability": proxy.capability.__dict__,
        "error": result.error_message,
    }
    _json(report)
    if not all(
        (
            result.status.value == "ok",
            unchanged,
            modes_unchanged,
            gradient_flags_unchanged,
            hooks == hooks_after,
            python_rng_unchanged,
            numpy_rng_unchanged,
            torch_rng_unchanged,
            primary_component_matches,
        )
    ):
        raise RuntimeError(f"Proxy validation failed: {report}")


def command_registry(args: argparse.Namespace, registry: str) -> None:
    if registry == "benchmark":
        load_builtin_benchmarks()
        target = BENCHMARKS
    elif registry == "space":
        load_builtin_spaces()
        target = SPACES
    else:
        load_builtin_proxies()
        target = PROXIES
    if args.action == "list":
        _json(target.names())
        return
    kwargs: dict[str, Any] = {}
    if registry == "benchmark":
        if (
            args.name in {"nasbench201", "nats_tss", "nats_sss", "nasbench301_surrogate"}
            and not args.trusted
        ):
            raise PermissionError(
                f"{args.name} uses a native serialized format; pass --trusted only for a verified source"
            )
        if args.path:
            resolved_path = args.path
        else:
            resolver_args = argparse.Namespace(
                benchmark=args.name,
                benchmark_path=None,
                catalog=args.catalog,
                data_root=args.data_root,
                transnas_space=args.transnas_space,
                slice_id=args.slice_id,
            )
            resolved_path = _resolve_benchmark_path(resolver_args)
        kwargs["path"] = resolved_path
        versions = {
            "nasbench101": "full",
            "nasbench201": "1.1",
            "nats_tss": "1.0",
            "nats_sss": "1.0",
            "nasbench301_surrogate": "1.0",
            "transnasbench101": "v10141024",
        }
        version = args.version or versions.get(args.name)
        if version:
            kwargs["version"] = version
        if args.name == "vitbench101":
            kwargs["slice_id"] = args.slice_id
        if args.name == "transnasbench101":
            kwargs["space"] = args.transnas_space
        if args.name == "nasbench301_surrogate":
            kwargs["architecture_path"] = args.architecture_path
            runtime_args = argparse.Namespace(
                runtime_benchmark_path=args.runtime_path,
                catalog=args.catalog,
            )
            runtime_path = _resolve_nb301_runtime_path(runtime_args)
            if runtime_path:
                kwargs["runtime_path"] = runtime_path
    instance = target.create(args.name, **kwargs)
    if registry == "benchmark":
        result = {"metadata": instance.metadata(), "capabilities": instance.capabilities()}
        if args.metric_name:
            architecture = next(instance.iter_architectures(args.start, args.start + 1))
            metric = MetricSpec(
                args.dataset,
                args.split,
                args.metric_name,
                args.epoch_budget,
                args.metric_seed,
                args.metric_seed_reduction,
                version,
                args.surrogate_noise,
            )
            result["query"] = {
                "architecture_id": architecture.architecture_id,
                "benchmark_index": architecture.benchmark_index,
                "metric_spec": asdict(metric),
                "value": instance.query_metrics(architecture, metric),
            }
        _json(result)
    elif registry == "proxy":
        _json(instance.capability.__dict__)
    else:
        _json(
            {
                "search_space_id": instance.search_space_id,
                "model_family": instance.model_family,
                "model_fidelity": getattr(instance, "model_fidelity", "unspecified"),
                "sample": instance.sample(args.seed).to_dict(),
            }
        )


def command_benchmark_sample(args: argparse.Namespace) -> None:
    load_builtin_spaces()
    load_builtin_benchmarks()
    if (
        args.name in {"nasbench201", "nats_tss", "nats_sss", "nasbench301_surrogate"}
        and not args.trusted
    ):
        raise PermissionError(
            f"{args.name} uses a native serialized format; pass --trusted only for a verified source"
        )
    resolver_args = argparse.Namespace(
        benchmark=args.name,
        benchmark_path=args.path,
        catalog=args.catalog,
        data_root=args.data_root,
        transnas_space=args.transnas_space,
        slice_id=args.slice_id,
    )
    kwargs: dict[str, Any] = {"path": _resolve_benchmark_path(resolver_args)}
    version = args.version or {
        "nasbench101": "full",
        "nasbench201": "1.1",
        "nats_tss": "1.0",
        "nats_sss": "1.0",
        "nasbench301_surrogate": "1.0",
        "transnasbench101": "v10141024",
    }.get(args.name)
    if version:
        kwargs["version"] = version
    if args.name == "vitbench101":
        kwargs["slice_id"] = args.slice_id
    elif args.name == "transnasbench101":
        kwargs["space"] = args.transnas_space
    elif args.name == "nasbench301_surrogate":
        kwargs["architecture_path"] = args.architecture_path
        runtime_args = argparse.Namespace(
            runtime_benchmark_path=args.runtime_path,
            catalog=args.catalog,
        )
        runtime_path = _resolve_nb301_runtime_path(runtime_args)
        if runtime_path:
            kwargs["runtime_path"] = runtime_path
    adapter = BENCHMARKS.create(args.name, **kwargs)
    architectures = adapter.iter_architectures()
    if args.name == "nasbench301_surrogate" and args.architecture_path is None:
        population_count = args.population_count
        if population_count is None:
            if args.fraction is not None:
                raise ValueError(
                    "Generated NAS-Bench-301 fraction sampling requires --population-count"
                )
            population_count = args.count
        if population_count is None or population_count <= 0:
            raise ValueError("Generated NAS-Bench-301 requires a positive population count")
        architectures = adapter.iter_architectures(0, population_count)
    manifest = create_sample_manifest(
        adapter.benchmark_id,
        str(adapter.metadata().get("version")) if adapter.metadata().get("version") else None,
        architectures,
        count=args.count,
        fraction=args.fraction,
        seed=args.seed,
        shards=args.shards,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    summary = {key: value for key, value in manifest.items() if key != "selected"}
    _json({**summary, "output": str(output)})


def command_data(args: argparse.Namespace) -> None:
    if args.action == "prepare-transnas-input":
        from zcp_test.data.transnas_inputs import generate_transnas_input_manifest

        output = args.output or str(Path(args.data_root) / "transnas-inputs.json")
        manifest = generate_transnas_input_manifest(
            args.split_json,
            args.data_root,
            output,
            split=args.split,
            verify_files=args.verify_files,
        )
        _json(
            {
                "output": str(Path(output).expanduser().resolve()),
                "records": len(manifest["records"]),
                "split": manifest["split"],
                "upstream_commit": manifest["upstream"]["commit"],
            }
        )
        return
    if args.action == "convert-imagenet16":
        from zcp_test.data.assets import sha256_file

        manifest = convert_imagenet16_120(
            args.source,
            args.output,
            trusted=args.trusted,
            replace=args.replace,
        )
        result: dict[str, Any] = {"manifest": str(manifest)}
        if args.register:
            registry = DataRegistry(args.catalog)
            registry.register(
                DataAsset(
                    "dataset_imagenet16_120",
                    str(manifest),
                    "npy-shards-v1",
                    sha256=sha256_file(manifest),
                    protocol="imagenet16-120-official-md5-safe-conversion-v1",
                    trusted=False,
                ),
                replace=args.replace,
            )
            result["asset_id"] = "dataset_imagenet16_120"
        _json(result)
        return
    if args.action == "checklist":
        records = data_checklist(args.root, args.catalog)
        if args.json:
            _json(records)
        else:
            from tabulate import tabulate

            print(
                tabulate(
                    [
                        [
                            record["benchmark_id"],
                            record["version"],
                            record["state"],
                            record["catalog_state"],
                            f"{record['estimated_bytes'] / 1024**3:.2f} GiB",
                            record["remediation"] or "-",
                        ]
                        for record in records
                    ],
                    headers=(
                        "benchmark",
                        "version",
                        "state",
                        "catalog",
                        "estimated",
                        "remediation",
                    ),
                    tablefmt="github",
                )
            )
        return
    if args.action == "bootstrap":
        benchmarks = list(data_checklist(args.root, args.catalog)) if args.all else None
        selected = (
            [record["benchmark_id"] for record in benchmarks]
            if benchmarks is not None
            else [value.strip() for value in (args.benchmarks or "").split(",") if value.strip()]
        )
        if not selected:
            raise ValueError("Specify --benchmarks or --all")
        total = sum(
            record["estimated_bytes"]
            for record in data_checklist(args.root, args.catalog)
            if record["benchmark_id"] in selected and record["state"] != "ready"
        )
        if not args.yes:
            if not sys.stdin.isatty():
                raise RuntimeError("Non-interactive bootstrap requires --yes")
            answer = input(
                f"Download/prepare {', '.join(selected)} under {args.root} "
                f"(up to {total / 1024**3:.2f} GiB)? [y/N] "
            )
            if answer.strip().lower() not in {"y", "yes"}:
                raise RuntimeError("Data bootstrap cancelled")
        _json(bootstrap_benchmarks(args.root, selected, catalog=args.catalog))
        return
    if args.action == "export-manifest":
        selected = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
        _json({"manifest": str(export_data_manifest(args.root, args.output, selected))})
        return
    if args.action == "import-manifest":
        result = verify_data_manifest(args.root, args.manifest)
        _json(result)
        if not result["valid"]:
            raise RuntimeError("Transferred benchmark data failed manifest verification")
        return
    registry = DataRegistry(args.catalog)
    if args.action == "list":
        _json([asset.__dict__ for asset in registry.list()])
    elif args.action == "register":
        registry.register(
            DataAsset(
                args.asset_id,
                str(Path(args.path).expanduser().resolve()),
                args.version,
                args.sha256,
                args.source_url,
                args.protocol,
                args.trusted,
            ),
            replace=args.replace,
        )
        _json(registry.get(args.asset_id).__dict__)
    elif args.action == "verify":
        if args.all:
            records = data_checklist(args.root, args.catalog)
            _json(records)
            if any(record["state"] != "ready" for record in records):
                raise RuntimeError("One or more benchmark data groups are not ready")
        else:
            _json(registry.verify(args.asset_id))
    elif args.action == "fetch":
        _json({"path": str(registry.fetch(args.asset_id, args.destination))})
    elif args.action == "convert-vit":
        path = convert_vitbench101(
            args.source,
            args.output,
            slice_id=args.slice_id,
            parser=vitbench101_release_parser,
            trusted=args.trusted,
        )
        _json({"path": str(path), "slice_id": args.slice_id})


def command_evaluate(args: argparse.Namespace) -> None:
    if bool(args.space) == bool(args.benchmark):
        raise ValueError("evaluate requires exactly one of space or benchmark")
    _prepare_gpu(args)
    load_builtin_spaces()
    load_builtin_benchmarks()
    adapter = None
    sample_manifest = None
    resolved_benchmark_version = args.benchmark_version or {
        "nasbench101": "full",
        "nasbench201": "1.1",
        "nats_tss": "1.0",
        "nats_sss": "1.0",
        "nasbench301_surrogate": "1.0",
        "transnasbench101": "v10141024",
        "vitbench101": "auto-prox-90ed458",
    }.get(args.benchmark)
    if args.benchmark:
        if (
            args.benchmark in {"nasbench201", "nats_tss", "nats_sss", "nasbench301_surrogate"}
            and not args.trusted
        ):
            raise PermissionError(
                f"{args.benchmark} uses a native serialized format; pass --trusted only for a verified source"
            )
        kwargs: dict[str, Any] = {"path": _resolve_benchmark_path(args)}
        if args.benchmark in {"nasbench201", "nats_tss", "nats_sss"}:
            kwargs["version"] = resolved_benchmark_version
        elif args.benchmark == "vitbench101":
            kwargs["slice_id"] = args.slice_id
        elif args.benchmark == "transnasbench101":
            kwargs["space"] = args.transnas_space
            kwargs["version"] = resolved_benchmark_version
        elif args.benchmark == "nasbench101":
            kwargs["version"] = resolved_benchmark_version
        elif args.benchmark == "nasbench301_surrogate":
            kwargs["architecture_path"] = args.architecture_path
            runtime_path = _resolve_nb301_runtime_path(args)
            if runtime_path:
                kwargs["runtime_path"] = runtime_path
        adapter = BENCHMARKS.create(args.benchmark, **kwargs)
        space = SPACES.create(adapter.search_space_id)
        if args.sample_manifest:
            if args.start != 0:
                raise ValueError("--start cannot be combined with --sample-manifest")
            sample_indices, sample_manifest = load_sample_indices(
                args.sample_manifest,
                benchmark_id=adapter.benchmark_id,
                benchmark_version=resolved_benchmark_version,
                search_space_id=adapter.search_space_id,
                shard_index=args.sample_shard,
            )
            expected_ids = {
                int(record["benchmark_index"]): record["architecture_id"]
                for record in sample_manifest["selected"]
            }
            architectures = []
            for benchmark_index in sample_indices:
                architecture = next(
                    adapter.iter_architectures(benchmark_index, benchmark_index + 1)
                )
                if architecture.architecture_id != expected_ids[benchmark_index]:
                    raise ValueError(
                        "Sample manifest architecture_id does not match the benchmark adapter"
                    )
                architectures.append(architecture)
        else:
            architectures = list(adapter.iter_architectures(args.start, args.start + args.count))
    else:
        space = SPACES.create(args.space)
        architectures = [space.sample(args.seed + index) for index in range(args.count)]
    if not architectures:
        raise ValueError("evaluate requires at least one architecture")
    model_provenance = _require_research_model(space, args.allow_approximation, adapter)
    weight_loader, weight_provenance = _prepare_model_weights(args, space)
    target_epoch_budget = args.epoch_budget
    target_seed_reduction = args.metric_seed_reduction
    target_direction = args.target_direction
    benchmark_variant = None
    benchmark_protocol = None
    if adapter:
        metadata = adapter.metadata()
        benchmark_protocol = metadata.get("protocol") or metadata.get("format")
        if adapter.benchmark_id == "nasbench301_surrogate":
            noise_mode = "noisy" if args.surrogate_noise else "deterministic"
            benchmark_protocol = f"{benchmark_protocol}:{noise_mode}"
        if args.benchmark == "vitbench101":
            benchmark_variant = args.slice_id
        elif args.benchmark == "transnasbench101":
            benchmark_variant = args.transnas_space
        if args.target_metric and target_epoch_budget is None:
            default_budget = getattr(adapter, "default_epoch_budget", None)
            budgets = tuple(adapter.capabilities().get("epoch_budgets", ()))
            if default_budget is not None:
                target_epoch_budget = int(default_budget)
            elif len(budgets) == 1:
                target_epoch_budget = int(budgets[0])
            elif len(budgets) > 1:
                raise ValueError(
                    f"{args.benchmark} target metric has multiple epoch budgets {budgets}; "
                    "specify --epoch-budget"
                )
    if target_direction == "auto":
        target_direction = _infer_target_direction(args.target_metric)
    with _selected_device(args) as (device, selection):
        import torch

        input_resolver = _candidate_input_resolver(args, device)
        resolved_batches = {
            architecture.architecture_id: input_resolver.resolve(architecture)
            for architecture in architectures
        }
        if space.search_space_id == "ofa_proxyless_mbv2":
            input_protocol = {
                **input_resolver.protocol_summary(space.search_space_id),
                "resolved_input_sizes": sorted(
                    {batch.protocol["input_size"] for batch in resolved_batches.values()}
                ),
                "input_fingerprints_by_size": {
                    str(batch.protocol["input_size"]): batch.fingerprint
                    for batch in resolved_batches.values()
                },
            }
        else:
            input_protocol = next(iter(resolved_batches.values())).protocol
        bn_recalibration_streams: dict[int, Any] = {}
        if args.bn_recalibration_batches:
            weight_provenance = {
                **weight_provenance,
                "bn_recalibration_required": False,
                "bn_recalibrated_batches": args.bn_recalibration_batches,
                "bn_recalibration_fingerprint": "per_candidate_input_size",
                "bn_recalibration_protocol_fidelity": "project_deterministic",
            }
        run_config = {
            **_args_config(args),
            "input_protocol": input_protocol,
            "sample_protocol": (
                {
                    "strategy": sample_manifest["strategy"],
                    "seed": sample_manifest["seed"],
                    "population_size": sample_manifest["population_size"],
                    "sample_count": len(architectures),
                    "manifest_sha256": file_sha256(args.sample_manifest),
                    "shard_index": args.sample_shard,
                }
                if sample_manifest is not None
                else None
            ),
            **model_provenance,
            **weight_provenance,
            "bn_recalibration_protocol": (
                "per_candidate_input_size" if args.bn_recalibration_batches else None
            ),
            "bn_recalibration_fingerprint": (
                "per_candidate_input_size" if args.bn_recalibration_batches else None
            ),
            "model_initialization_protocol": "architecture-hash-v1",
        }
        runtime = {
            "gpu_selection": selection,
            "input_fingerprint": (
                next(iter(resolved_batches.values())).fingerprint
                if space.search_space_id != "ofa_proxyless_mbv2"
                else "per_candidate_input_size"
            ),
        }
        with RunContext(args.output, sys.argv, run_config, runtime=runtime) as run:
            writer = JsonlWriter(run.directory / "scores.jsonl", fsync_every=1)
            loss_fn = torch.nn.CrossEntropyLoss()
            calls = 0
            statuses: list[str] = []
            primary_components: dict[str, str] = {}
            for architecture in architectures:
                batch = resolved_batches[architecture.architecture_id]
                model_initialization_seed = _seed_search_model(
                    args.seed, architecture.architecture_id
                )
                input_metadata = {
                    **input_resolver.metadata(architecture, batch),
                    "model_initialization_seed": model_initialization_seed,
                    "model_initialization_protocol": "architecture-hash-v1",
                }
                actual_input_size = int(input_metadata["actual_input_size"])
                model = (
                    adapter.build_model(architecture, args.dataset)
                    if adapter
                    else space.build_model(architecture, args.classes)
                )
                architecture_weight_provenance = (
                    weight_loader.export(model) if weight_loader is not None else weight_provenance
                )
                input_resolver.validate_model(architecture, model, batch)
                model = model.to(device)
                if args.bn_recalibration_batches:
                    from zcp_test.models.mobile import recalibrate_batch_norm

                    if actual_input_size not in bn_recalibration_streams:
                        bn_recalibration_streams[actual_input_size] = _prepare_bn_recalibration(
                            args,
                            device,
                            weight_loader,
                            input_size=actual_input_size,
                        )
                    bn_recalibration = bn_recalibration_streams[actual_input_size]
                    calibrated = recalibrate_batch_norm(model, bn_recalibration, device=device)
                    architecture_weight_provenance = {
                        **architecture_weight_provenance,
                        "bn_recalibration_required": False,
                        "bn_recalibrated_batches": calibrated,
                        "bn_recalibration_fingerprint": bn_recalibration.fingerprint,
                        "bn_recalibration_protocol_fidelity": "project_deterministic",
                    }
                target = None
                if adapter and args.target_metric:
                    target = adapter.query_metrics(
                        architecture,
                        MetricSpec(
                            args.dataset,
                            args.target_split,
                            args.target_metric,
                            target_epoch_budget,
                            args.metric_seed,
                            target_seed_reduction,
                            benchmark_version=resolved_benchmark_version,
                            surrogate_noise=args.surrogate_noise,
                        ),
                    ).get(args.target_metric)
                for proxy_name in args.proxies.split(","):
                    proxy_id = proxy_name.strip().lower()
                    result = evaluate_proxy(
                        proxy_id,
                        model,
                        batch.inputs,
                        batch.labels,
                        loss_fn,
                        space.model_family,
                        unsupported_reason=_transnas_unsupported_reason(
                            adapter.benchmark_id if adapter else None,
                            args.dataset,
                            proxy_id,
                        ),
                    )
                    calls += 1
                    statuses.append(result.status.value)
                    primary_components[proxy_id] = result.primary_component
                    writer.append(
                        {
                            "schema_version": "2.1",
                            "run_id": run.run_id,
                            "benchmark_id": adapter.benchmark_id if adapter else None,
                            "benchmark_version": resolved_benchmark_version if adapter else None,
                            "search_space_id": space.search_space_id,
                            "architecture_id": architecture.architecture_id,
                            "architecture": architecture.spec,
                            "benchmark_index": architecture.benchmark_index,
                            "dataset": args.dataset,
                            "proxy_id": proxy_id,
                            "proxy_version": result.proxy_version,
                            "proxy_implementation_fidelity": result.implementation_fidelity,
                            "proxy_source": result.source,
                            "proxy_alias_of": result.alias_of,
                            "direction": result.direction.value,
                            "resource_direction": (
                                result.resource_direction.value
                                if result.resource_direction is not None
                                else None
                            ),
                            "primary_component": result.primary_component,
                            "score": result.score,
                            "components": result.components,
                            "target_metric": args.target_metric,
                            "target_split": args.target_split,
                            "target_value": target,
                            "target_direction": target_direction,
                            "target_epoch_budget": target_epoch_budget,
                            "target_seed": args.metric_seed,
                            "target_seed_reduction": target_seed_reduction,
                            "benchmark_variant": benchmark_variant,
                            "benchmark_protocol": benchmark_protocol,
                            "surrogate_noise": args.surrogate_noise if adapter else None,
                            **model_provenance,
                            **weight_provenance,
                            **architecture_weight_provenance,
                            "status": result.status.value,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                            "seed": args.seed,
                            "duration_seconds": result.duration_seconds,
                            "peak_memory_mb": result.peak_memory_mb,
                            "input_source": args.input_source,
                            **input_metadata,
                        }
                    )
                del model
            _json(
                {
                    "run": str(run.directory),
                    "architectures": len(architectures),
                    "proxy_calls": calls,
                    "score_rows": calls,
                    **_evaluation_status_summary(statuses),
                    "primary_components": primary_components,
                }
            )


def command_correlate(args: argparse.Namespace) -> None:
    scores: dict[tuple[Any, Any], tuple[Any, str]] = {}
    for row in read_score_records(args.scores):
        if row.get("status", "ok") != "ok":
            continue
        key = (row[args.id_field], row.get("proxy_id"))
        if key in scores:
            raise ValueError(f"Duplicate score key in correlate input: {key}")
        scores[key] = (
            score_component(row, args.component) if args.component else row.get(args.score_field),
            str(row.get("direction", "maximize")),
        )
    targets: dict[Any, Any] = {}
    for row in read_jsonl(args.targets):
        if row.get(args.target_field) is None:
            continue
        key = row[args.id_field]
        if key in targets:
            raise ValueError(f"Duplicate target architecture ID in correlate input: {key}")
        targets[key] = row[args.target_field]
    groups: dict[str, tuple[list[float], list[float], set[str]]] = {}
    available_by_proxy: dict[str, int] = {}
    for (architecture_id, proxy_id), (score, direction) in scores.items():
        proxy_name = str(proxy_id)
        available_by_proxy[proxy_name] = available_by_proxy.get(proxy_name, 0) + int(
            score is not None
        )
        if architecture_id in targets and score is not None:
            if direction not in ("maximize", "minimize"):
                raise ValueError(f"Unknown score direction {direction!r} for proxy {proxy_id!r}")
            target_values, score_values, directions = groups.setdefault(proxy_name, ([], [], set()))
            target = float(targets[architecture_id])
            target_values.append(-target if args.target_direction == "minimize" else target)
            numeric_score = float(score)
            score_values.append(-numeric_score if direction == "minimize" else numeric_score)
            directions.add(direction)
    writer = JsonlWriter(args.output, fsync_every=1)
    records = []
    for proxy_id, (target_values, score_values, directions) in groups.items():
        if len(directions) != 1:
            raise ValueError(f"Mixed score directions for proxy {proxy_id!r}: {sorted(directions)}")
        score_direction = next(iter(directions))
        paired_count = len(score_values)
        record = {
            "proxy_id": proxy_id,
            "component": args.component or "primary",
            "target_field": args.target_field,
            "score_direction": score_direction,
            "target_direction": args.target_direction,
            "direction_normalized_to_maximize": True,
            "available_score_count": available_by_proxy[proxy_id],
            "available_target_count": len(targets),
            "paired_count": paired_count,
            "score_coverage": paired_count / max(1, available_by_proxy[proxy_id]),
            "target_coverage": paired_count / max(1, len(targets)),
            **correlation_summary(target_values, score_values, args.ndcg_k),
        }
        writer.append(record)
        records.append(record)
    _json({"correlations": records, "rows": len(records), "output": args.output})


def command_search(args: argparse.Namespace) -> None:
    _prepare_gpu(args)
    if not args.space:
        raise ValueError("search requires --space or a config-provided space")
    load_builtin_spaces()
    space = SPACES.create(args.space)
    source_aligned_plainnet = args.controller == "plainnet_source_aligned"
    resource_limits = {
        name: int(value)
        for name, value in {
            "max_parameters": args.max_parameters,
            "max_macs": args.max_macs,
        }.items()
        if value is not None
    }
    invalid_limits = {
        name: value for name, value in resource_limits.items() if value <= 0
    }
    if invalid_limits:
        raise ValueError(f"Search resource limits must be positive: {invalid_limits}")
    if args.constraint_max_attempts <= 0:
        raise ValueError("constraint_max_attempts must be positive")
    source_target = None
    if source_aligned_plainnet:
        if resource_limits:
            raise ValueError(
                "PlainNet source-aligned search uses --flops-target and does not accept "
                "generic --max-parameters/--max-macs limits"
            )
        source_target = resolve_target_profile(args.flops_target)
        required = {
            "space": (args.space, "zennas_plainnet_mbv2"),
            "proxy": (args.proxy, "az_nas_plainnet"),
            "aggregator": (args.aggregator, "az_nas_log_rank"),
            "population": (args.population, 1024),
            "generations": (args.generations, 0),
            "batch_size": (args.batch_size, 64),
            "input_size": (args.input_size, 224),
            "classes": (args.classes, 1000),
            "dataset": (args.dataset, "imagenet1k"),
            "input_source": (args.input_source, "random"),
            "weight_mode": (args.weight_mode, "independent_scratch"),
            "bn_recalibration_batches": (args.bn_recalibration_batches, 0),
        }
        mismatched = {
            name: {"actual": actual, "required": expected}
            for name, (actual, expected) in required.items()
            if actual != expected
        }
        if mismatched:
            raise ValueError(
                "PlainNet source-aligned controller protocol mismatch: "
                f"{mismatched}"
            )
        if args.valid_candidates is None:
            args.valid_candidates = 100_000
        if args.valid_candidates != 100_000:
            raise ValueError(
                "PlainNet source-aligned production search requires exactly "
                "100000 valid candidates"
            )
        if args.model_checkpoint or args.allow_approximation:
            raise ValueError(
                "PlainNet source-aligned search uses independent scratch models "
                "and does not allow approximation"
            )
    elif args.flops_target is not None or args.valid_candidates is not None:
        raise ValueError(
            "--flops-target and --valid-candidates require "
            "--controller plainnet_source_aligned"
        )
    model_provenance = _require_research_model(space, args.allow_approximation)
    weight_loader, weight_provenance = _prepare_model_weights(args, space)
    if space.search_space_id == "ofa_proxyless_mbv2" and _argument_was_explicit(args, "input_size"):
        raise ValueError(
            "OFA search input_size is controlled by each candidate resolution; "
            "remove the explicit input_size setting"
        )
    with _selected_device(args) as (device, selection):
        import torch

        input_resolver = _candidate_input_resolver(args, device)
        fixed_batch = (
            input_resolver.resolve_input_size(args.input_size)
            if space.search_space_id != "ofa_proxyless_mbv2"
            else None
        )
        bn_recalibration_streams: dict[int, Any] = {}
        if args.bn_recalibration_batches:
            weight_provenance = {
                **weight_provenance,
                "bn_recalibration_required": False,
                "bn_recalibrated_batches": args.bn_recalibration_batches,
                "bn_recalibration_fingerprint": "per_candidate_input_size",
                "bn_recalibration_protocol_fidelity": "project_deterministic",
            }
        load_builtin_proxies()
        proxy_capability = PROXIES.create(args.proxy).capability
        if (
            "approximation" in proxy_capability.implementation_fidelity
            and not args.allow_approximation
        ):
            raise ValueError(
                f"Proxy {args.proxy!r} is {proxy_capability.implementation_fidelity!r}; "
                "pass --allow-approximation only for an explicitly exploratory search"
            )
        component_aggregator = None
        az_nas_components = {
            "az_nas_autoformer": ("expressivity", "trainability", "complexity"),
            "az_nas_plainnet": (
                "expressivity",
                "progressivity",
                "trainability",
                "complexity",
            ),
        }
        az_nas_spaces = {
            "az_nas_autoformer": "autoformer",
            "az_nas_plainnet": "zennas_plainnet_mbv2",
        }
        if args.aggregator == "az_nas_log_rank":
            if args.proxy not in az_nas_components:
                raise ValueError(
                    "az_nas_log_rank requires --proxy az_nas_autoformer or az_nas_plainnet"
                )
            from zcp_test.proxies.az_nas import log_rank_aggregate

            def component_aggregator(rows: Any) -> list[float]:
                return log_rank_aggregate(rows, az_nas_components[args.proxy])
        elif args.proxy in az_nas_components:
            raise ValueError(
                f"{args.proxy} requires --aggregator az_nas_log_rank; "
                "expressivity alone is not the AZ-NAS search score"
            )
        expected_az_nas_space = az_nas_spaces.get(args.proxy)
        if expected_az_nas_space and space.search_space_id != expected_az_nas_space:
            raise ValueError(
                f"{args.proxy} requires --space {expected_az_nas_space}; "
                f"it is not defined for {space.search_space_id}"
            )
        search_input_fingerprint = (
            "per_candidate_input_size" if fixed_batch is None else fixed_batch.fingerprint
        )
        search_input_size: int | str = (
            "candidate_resolution" if fixed_batch is None else args.input_size
        )
        search_identity = {
            "search_space_id": space.search_space_id,
            "model_fidelity": model_provenance["model_fidelity"],
            "model_profile": model_provenance.get("model_profile"),
            "implementation_commit": model_provenance.get("implementation_commit"),
            "proxy_id": args.proxy,
            "proxy_version": proxy_capability.version,
            "proxy_direction": proxy_capability.direction.value,
            "aggregator": args.aggregator,
            "model_initialization_protocol": "architecture-hash-v1",
            "dataset": args.dataset,
            "input_source": args.input_source,
            "input_fingerprint": search_input_fingerprint,
            "seed": args.seed,
            "population_size": args.population,
            "elite_ratio": args.elite_ratio,
            "batch_size": args.batch_size,
            "input_size": search_input_size,
            "classes": args.classes,
            "weight_mode": weight_provenance["weight_mode"],
            "model_checkpoint_sha256": weight_provenance.get("model_checkpoint_sha256"),
            "bn_recalibration_fingerprint": weight_provenance.get("bn_recalibration_fingerprint"),
        }
        if resource_limits:
            search_identity.update(
                {
                    "resource_constraints": resource_limits,
                    "constraint_max_attempts": args.constraint_max_attempts,
                }
            )
        if source_aligned_plainnet:
            assert source_target is not None
            search_identity.update(
                {
                    "controller_id": "plainnet_source_aligned",
                    "controller_fidelity": "source_aligned_control_flow_port",
                    "valid_candidates": args.valid_candidates,
                    "flops_target": source_target.target_id,
                    "flops_budget": source_target.flops_budget,
                    "max_layers": source_target.max_layers,
                    "crossover": False,
                }
            )
        resume_state = (
            load_plainnet_search_state(args.resume)
            if args.resume and source_aligned_plainnet
            else load_search_state(args.resume)
            if args.resume
            else None
        )
        if resume_state is not None and not source_aligned_plainnet:
            validate_search_state_identity(resume_state, search_identity)
        record_metadata = {
            **model_provenance,
            **weight_provenance,
            **search_identity,
            "proxy_implementation_fidelity": proxy_capability.implementation_fidelity,
            "proxy_source": proxy_capability.source,
            "resource_direction": (
                proxy_capability.resource_direction.value
                if proxy_capability.resource_direction is not None
                else None
            ),
        }
        loss_fn = torch.nn.CrossEntropyLoss()
        runtime = {
            "gpu_selection": selection,
            "input_fingerprint": search_input_fingerprint,
        }
        config = {
            **_args_config(args),
            "input_protocol": (
                input_resolver.protocol_summary(space.search_space_id)
                if fixed_batch is None
                else fixed_batch.protocol
            ),
            **model_provenance,
            **weight_provenance,
            "search_identity": search_identity,
            "bn_recalibration_protocol": (
                "per_candidate_input_size" if args.bn_recalibration_batches else None
            ),
            "bn_recalibration_fingerprint": (
                "per_candidate_input_size" if args.bn_recalibration_batches else None
            ),
        }
        with RunContext(args.output, sys.argv, config, runtime=runtime) as run:
            resource_cache: dict[str, dict[str, Any]] = {}

            def candidate_constraint(architecture: Any) -> dict[str, Any] | None:
                if not resource_limits:
                    return {}
                architecture_id = str(architecture.architecture_id)
                if architecture_id not in resource_cache:
                    measurement = measure_architecture_resources(
                        space,
                        architecture,
                        {
                            "input_size": int(
                                architecture.spec.get("resolution", args.input_size)
                            )
                        },
                        args.classes,
                    )
                    resource_cache[architecture_id] = {
                        "parameters": int(measurement.parameters),
                        "compute_value": int(measurement.compute_value),
                        "compute_metric": measurement.compute_metric,
                        "generic_flops": bool(measurement.generic_flops),
                        "resource_input_size": int(measurement.input_size),
                    }
                metadata = resource_cache[architecture_id]
                if (
                    "max_macs" in resource_limits
                    and metadata["compute_metric"] != "thop_macs"
                ):
                    raise ValueError(
                        "--max-macs requires the thop_macs resource protocol; "
                        f"{space.search_space_id} reported {metadata['compute_metric']}"
                    )
                if (
                    "max_parameters" in resource_limits
                    and metadata["parameters"] > resource_limits["max_parameters"]
                ):
                    return None
                if (
                    "max_macs" in resource_limits
                    and metadata["compute_value"] > resource_limits["max_macs"]
                ):
                    return None
                return {**metadata, "resource_constraints": resource_limits}

            def candidate_input(architecture: Any) -> tuple[Any, dict[str, Any]]:
                batch = input_resolver.resolve(architecture)
                return batch, {
                    **input_resolver.metadata(architecture, batch),
                    "model_initialization_seed": _search_model_seed(
                        args.seed, architecture.architecture_id
                    ),
                    "model_initialization_protocol": "architecture-hash-v1",
                }

            def evaluation_identity(
                architecture: Any,
            ) -> tuple[str, dict[str, Any]]:
                batch, metadata = candidate_input(architecture)
                identity = cache_key(
                    architecture,
                    args.proxy,
                    args.dataset,
                    args.seed,
                    batch.fingerprint,
                    proxy_capability.version,
                )
                return identity, {**metadata, "evaluation_cache_key": identity}

            def evaluator(architecture: Any) -> float | dict[str, float]:
                batch, input_metadata = candidate_input(architecture)
                actual_input_size = int(input_metadata["actual_input_size"])
                _seed_search_model(args.seed, architecture.architecture_id)
                model = space.build_model(architecture, args.classes)
                if weight_loader is not None:
                    weight_loader.export(model)
                input_resolver.validate_model(architecture, model, batch)
                model = model.to(device)
                if args.bn_recalibration_batches:
                    from zcp_test.models.mobile import recalibrate_batch_norm

                    if actual_input_size not in bn_recalibration_streams:
                        bn_recalibration_streams[actual_input_size] = _prepare_bn_recalibration(
                            args,
                            device,
                            weight_loader,
                            input_size=actual_input_size,
                        )
                    bn_recalibration = bn_recalibration_streams[actual_input_size]
                    recalibrate_batch_norm(model, bn_recalibration, device=device)
                result = evaluate_proxy(
                    args.proxy,
                    model,
                    batch.inputs,
                    batch.labels,
                    loss_fn,
                    space.model_family,
                )
                if result.status.value != "ok" or result.score is None:
                    raise RuntimeError(
                        result.error_message or "proxy did not return a primary score"
                    )
                if args.aggregator == "az_nas_log_rank":
                    return result.components
                return result.score if result.direction.value == "maximize" else -result.score

            if source_aligned_plainnet:
                assert source_target is not None
                search = PlainNetSourceAlignedSearch(
                    space=space,
                    evaluator=evaluator,
                    writer=JsonlWriter(run.directory / "search.jsonl", 1),
                    state_path=run.directory / "search-state.json",
                    seed=args.seed,
                    target=source_target,
                    valid_candidates=args.valid_candidates,
                    parent_pool=args.population,
                    classes=args.classes,
                    record_metadata=record_metadata,
                    state_identity=search_identity,
                    resume_state=resume_state,
                    resume_journal_path=(
                        Path(args.resume).expanduser().resolve().with_name("search.jsonl")
                        if args.resume
                        else None
                    ),
                )
                best = search.run()
                if best is None:
                    raise RuntimeError("PlainNet source-aligned search stopped before completion")
            else:
                search = EvolutionSearch(
                    space,
                    evaluator,
                    JsonlWriter(run.directory / "search.jsonl", 1),
                    args.population,
                    args.elite_ratio,
                    args.seed,
                    record_metadata=record_metadata,
                    evaluation_identity=evaluation_identity,
                    component_aggregator=component_aggregator,
                    state_path=run.directory / "search-state.json",
                    resume_state=resume_state,
                    state_identity=search_identity,
                    candidate_constraint=(
                        candidate_constraint if resource_limits else None
                    ),
                    max_constraint_attempts=args.constraint_max_attempts,
                )
                best = search.run(args.generations)
            (run.directory / "best_architecture.json").write_text(
                json.dumps(best.architecture.to_dict(), indent=2), encoding="utf-8"
            )
            _json(
                {
                    "run": str(run.directory),
                    "best_score": best.score,
                    "architecture": best.architecture.to_dict(),
                    "search_state": str(run.directory / "search-state.json"),
                }
            )


def command_train(args: argparse.Namespace) -> None:
    from zcp_test.training.protocols import (
        resolve_gradient_accumulation,
        resolve_acceptance_protocol,
        resolve_per_device_batch_size,
        scale_learning_rate,
        validate_acceptance_training_protocol,
        validate_formal_training_protocol,
    )

    config = load_config(args.config)
    acceptance_smoke = bool(getattr(args, "acceptance_smoke", False))
    real_data_preflight = bool(getattr(args, "real_data_preflight", False))
    full_batch_smoke = bool(getattr(args, "full_batch_smoke", False))
    if full_batch_smoke and not args.smoke:
        raise ValueError("--full-batch-smoke requires --smoke")
    formal_training = not args.smoke and not acceptance_smoke and not real_data_preflight
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_rank = int(os.environ.get("RANK", "0"))
    distributed_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed_world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= distributed_rank < distributed_world_size:
        raise ValueError("RANK must be within WORLD_SIZE")
    if distributed_world_size == 1:
        _prepare_gpu(args)
    load_builtin_spaces()
    space = SPACES.create(config["space"])
    model_provenance = _space_provenance(space)
    model_fidelity = model_provenance["model_fidelity"]
    if not args.smoke and model_fidelity != ModelFidelity.REFERENCE_MODEL.value:
        raise NotImplementedError(
            f"Formal training for {space.search_space_id} is unavailable because its current "
            f"model fidelity is {model_fidelity!r}; formal training requires reference_model"
        )
    formal_training_ready = bool(
        config.get("formal_training_ready", space.search_space_id == "darts")
    )
    if formal_training and not formal_training_ready:
        blockers = config.get("formal_training_blockers", [])
        suffix = f" Blockers: {', '.join(map(str, blockers))}." if blockers else ""
        raise NotImplementedError(
            f"Formal training protocol for {space.search_space_id!r} is not accepted yet; "
            "use --smoke only until the declared recipe is implemented and validated."
            f"{suffix}"
        )
    if formal_training:
        validate_formal_training_protocol(config)
        if args.batch_size is not None and args.batch_size != int(config["batch_size"]):
            raise ValueError("Formal training cannot override the accepted batch_size")
        if args.input_size is not None and args.input_size != int(config["input_size"]):
            raise ValueError("Formal training cannot override the accepted input_size")
    elif acceptance_smoke or real_data_preflight:
        validate_acceptance_training_protocol(config)
        if args.batch_size is not None and args.batch_size != int(config["batch_size"]):
            raise ValueError("Real-data validation cannot override the candidate batch_size")
        if args.input_size is not None and args.input_size != int(config["input_size"]):
            raise ValueError("Real-data validation cannot override the candidate input_size")
    architecture = (
        space.sample(args.seed)
        if args.architecture is None
        else space.canonicalize(_load_architecture_spec(args.architecture))
    )
    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    if full_batch_smoke and epochs != 1:
        raise ValueError("--full-batch-smoke requires exactly one epoch")
    acceptance_protocol = None
    if acceptance_smoke:
        acceptance_protocol = resolve_acceptance_protocol(
            config,
            epochs,
            args.data_fraction,
        )
    if real_data_preflight and (epochs != 1 or args.data_fraction != 1.0):
        raise ValueError("Real-data preflight requires exactly one epoch over the complete dataset")
    dataset = str(config["dataset"])
    classes = args.classes or {"cifar10": 10, "cifar100": 100, "imagenet1k": 1000}.get(dataset, 10)
    configured_batch_size = int(config.get("batch_size", 8))
    batch_size_semantics = str(config.get("batch_size_semantics", "per_device"))
    if not args.smoke and batch_size_semantics == "global":
        batch_size = resolve_per_device_batch_size(
            configured_batch_size,
            distributed_world_size,
            batch_size_semantics,
        )
    else:
        batch_size = args.batch_size or configured_batch_size
    input_size = args.input_size or int(config.get("input_size", 32))
    base_learning_rate = float(config["learning_rate"])
    requested_accumulation = config.get("gradient_accumulation_steps", 1)
    target_global_batch_size = config.get("target_global_batch_size")
    gradient_accumulation_steps = resolve_gradient_accumulation(
        requested_accumulation,
        batch_size,
        distributed_world_size,
        None if target_global_batch_size is None else int(target_global_batch_size),
        smoke=args.smoke,
    )
    learning_rate_reference_batch_size = config.get("learning_rate_reference_batch_size")
    if learning_rate_reference_batch_size is None or args.smoke:
        learning_rate = base_learning_rate
        effective_global_batch_size = (
            batch_size * distributed_world_size * gradient_accumulation_steps
        )
    else:
        learning_rate, effective_global_batch_size = scale_learning_rate(
            base_learning_rate,
            batch_size,
            distributed_world_size,
            int(learning_rate_reference_batch_size),
            gradient_accumulation_steps,
        )
    training = TrainingConfig(
        epochs=epochs,
        optimizer=str(config["optimizer"]),
        learning_rate=learning_rate,
        weight_decay=float(config["weight_decay"]),
        scheduler=str(config.get("scheduler", "cosine")),
        scheduler_step_size=int(config.get("scheduler_step_size", 1)),
        scheduler_gamma=float(config.get("scheduler_gamma", 0.97)),
        warmup_epochs=int(config.get("warmup_epochs", 0)),
        warmup_learning_rate=(
            None
            if config.get("warmup_learning_rate") is None
            else float(config["warmup_learning_rate"])
        ),
        minimum_learning_rate=float(config.get("minimum_learning_rate", 0.0)),
        label_smoothing=float(config.get("label_smoothing", 0)),
        validation_label_smoothing=float(config.get("validation_label_smoothing", 0.0)),
        amp=bool(config.get("amp", True)),
        momentum=float(config.get("momentum", 0.9)),
        nesterov=bool(config.get("nesterov", True)),
        auxiliary_weight=float(config.get("auxiliary_weight", 0.0)),
        drop_path_prob=float(config.get("drop_path_prob", 0.0)),
        grad_clip=None if config.get("grad_clip") is None else float(config["grad_clip"]),
        amp_initial_scale=float(config.get("amp_initial_scale", 65536.0)),
        mixup=float(config.get("mixup", 0.0)),
        cutmix=float(config.get("cutmix", 0.0)),
        mixup_probability=float(config.get("mixup_probability", 1.0)),
        mixup_switch_probability=float(config.get("mixup_switch_probability", 0.5)),
        mixup_mode=str(config.get("mixup_mode", "batch")),
        exclude_bias_norm_from_weight_decay=bool(
            config.get("exclude_bias_norm_from_weight_decay", False)
        ),
        exclude_norm_from_weight_decay=bool(config.get("exclude_norm_from_weight_decay", False)),
        gradient_accumulation_steps=gradient_accumulation_steps,
        schedule_epochs=int(config["epochs"]),
        non_blocking_transfer=bool(config.get("non_blocking_transfer", True)),
        memory_format=str(config.get("memory_format", "contiguous")),
        cudnn_benchmark=bool(config.get("cudnn_benchmark", False)),
        allow_tf32=bool(config.get("allow_tf32", False)),
    )
    data_root = None if args.smoke else _resolve_data_root(args, dataset)
    if not args.smoke and not data_root:
        raise ValueError(
            "Real-data training requires --data-root; synthetic data is restricted to --smoke"
        )
    with _training_device(
        args,
        distributed_world_size,
        distributed_rank,
        distributed_local_rank,
    ) as (device, selection):
        deterministic = bool(config.get("deterministic", True))
        seed_state = _seed_training(
            args.seed,
            distributed_rank,
            deterministic,
            cudnn_benchmark=training.cudnn_benchmark,
            allow_tf32=training.allow_tf32,
        )
        model = _build_training_model(space, architecture, classes, config)
        if distributed_world_size > 1:
            import torch

            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[distributed_local_rank],
                output_device=distributed_local_rank,
            )
        resolved = {
            **config,
            "architecture": architecture.to_dict(),
            "epochs": epochs,
            "data_fraction": args.data_fraction,
            "smoke": args.smoke,
            "full_batch_smoke": full_batch_smoke,
            "acceptance_smoke": acceptance_smoke,
            "real_data_preflight": real_data_preflight,
            "acceptance_protocol": acceptance_protocol,
            "training_mode": (
                "synthetic_full_batch_memory_smoke"
                if full_batch_smoke
                else "synthetic_smoke"
                if args.smoke
                else "acceptance_smoke"
                if acceptance_smoke
                else "real_data_preflight"
                if real_data_preflight
                else "formal"
            ),
            "classes": classes,
            "batch_size": batch_size,
            "configured_batch_size": configured_batch_size,
            "batch_size_semantics": batch_size_semantics,
            "input_size": input_size,
            "distributed_world_size": distributed_world_size,
            "distributed_rank": distributed_rank,
            "distributed_local_rank": distributed_local_rank,
            "seed": args.seed,
            "rank_seed": seed_state["rank_seed"],
            "train_workers": args.workers,
            "valid_workers": args.valid_workers if args.valid_workers is not None else args.workers,
            "deterministic": deterministic,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_global_batch_size": effective_global_batch_size,
            "base_learning_rate": base_learning_rate,
            "effective_learning_rate": learning_rate,
            "schedule_epochs": training.schedule_epochs,
            "learning_rate_reference_batch_size": learning_rate_reference_batch_size,
            "learning_rate_scaling": config.get("learning_rate_scaling", "none"),
            "model_fidelity": model_fidelity,
            "implementation_source": config.get("implementation_source")
            or model_provenance["implementation_source"],
            "implementation_commit": config.get("implementation_commit")
            or model_provenance["implementation_commit"],
            "model_implementation_source": model_provenance["implementation_source"],
            "model_implementation_commit": model_provenance["implementation_commit"],
            "training_implementation_source": config.get("implementation_source"),
            "training_implementation_commit": config.get("implementation_commit"),
        }
        runtime = {
            "gpu_selection": selection,
            "seed_state": seed_state,
            "resume": _checkpoint_lineage(args.resume) if args.resume else None,
            "distributed": {
                "world_size": distributed_world_size,
                "backend": "nccl" if distributed_world_size > 1 else None,
                "launcher": "torchrun" if distributed_world_size > 1 else None,
            },
        }
        with _training_run_context(
            args.output,
            sys.argv,
            resolved,
            runtime,
            distributed_world_size,
            distributed_rank,
        ) as run:
            batch = (
                batch_size
                if full_batch_smoke
                else min(batch_size, 2 if dataset == "imagenet1k" else 4)
                if args.smoke
                else batch_size
            )
            size = (
                input_size
                if dataset == "imagenet1k"
                else min(input_size, 64)
                if args.smoke
                else input_size
            )
            if args.smoke:
                train_loader = _synthetic_loader(batch, size, classes, 2)
                valid_loader = _synthetic_loader(batch, size, classes, 1)
            else:
                loader_config = dict(config)
                if args.valid_workers is not None:
                    loader_config["valid_workers"] = args.valid_workers
                train_loader, valid_loader = _real_loaders(
                    dataset,
                    data_root,
                    batch,
                    size,
                    args.workers,
                    loader_config,
                    args.data_fraction,
                    args.seed,
                    distributed_world_size,
                    distributed_rank,
                )
            result = train_model(
                model,
                train_loader,
                valid_loader,
                training,
                run.directory,
                device,
                args.resume,
                resume_trusted=args.trusted,
                run_identity={
                    "search_space_id": space.search_space_id,
                    "architecture_id": architecture.architecture_id,
                    "dataset": dataset,
                    "protocol": config.get("protocol"),
                    "classes": classes,
                    "input_size": input_size,
                    "model_fidelity": model_fidelity,
                    "seed": args.seed,
                    "data_fraction": args.data_fraction,
                    "training_mode": (
                        "synthetic_full_batch_memory_smoke"
                        if full_batch_smoke
                        else "synthetic_smoke"
                        if args.smoke
                        else "acceptance_smoke"
                        if acceptance_smoke
                        else "real_data_preflight"
                        if real_data_preflight
                        else "formal"
                    ),
                    "acceptance_protocol": acceptance_protocol,
                },
                progress_callback=(
                    (lambda kind, fields: run.event(kind, **fields))
                    if distributed_rank == 0
                    else None
                ),
            )
            if distributed_rank == 0:
                _json({"run": str(run.directory), **result})


def _load_architecture_spec(value: str) -> dict[str, Any]:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "--architecture must be an existing JSON file or an inline JSON object"
            ) from error
    if not isinstance(payload, dict):
        raise ValueError("Architecture JSON must be an object")
    specification = payload.get("spec", payload)
    if not isinstance(specification, dict):
        raise ValueError("Architecture 'spec' must be an object")
    return specification


def _build_training_model(
    space: Any, architecture: Any, classes: int, config: dict[str, Any]
) -> Any:
    if space.search_space_id == "darts":
        return space.build_model(architecture, classes, profile=config.get("model_profile"))
    if hasattr(space, "build_training_model"):
        return space.build_training_model(architecture, classes, config)
    return space.build_model(architecture, classes)


def command_report(args: argparse.Namespace) -> None:
    if args.action == "bundle":
        _json(_report_bundle(args.runs, args.output, args.title, args.bootstrap_samples))
        return
    if not args.source or not args.output:
        raise ValueError("Legacy report mode requires --source and --output")
    if args.format == "csv":
        count = jsonl_to_csv(args.source, args.output)
    elif args.format == "html":
        count = static_html(args.source, args.output, args.title)
    else:
        count = curve_plot(args.source, args.output, args.kind)
    _json({"rows": count, "output": args.output})


def _report_bundle(
    runs: list[str], output: str, title: str, bootstrap_samples: int
) -> dict[str, Any]:
    import pandas as pd

    expanded_runs: list[str] = []
    for run in runs:
        path = Path(run)
        if path.is_dir() and not any(
            (path / name).exists() for name in ("scores.jsonl", "search.jsonl", "training.jsonl")
        ):
            expanded_runs.extend(
                str(candidate)
                for candidate in sorted(path.iterdir())
                if candidate.is_dir()
                and any(
                    (candidate / name).exists()
                    for name in ("scores.jsonl", "search.jsonl", "training.jsonl")
                )
            )
        else:
            expanded_runs.append(run)
    score_runs = [
        run
        for run in expanded_runs
        if (Path(run) / "scores.jsonl").exists()
        or (Path(run).is_file() and Path(run).name == "scores.jsonl")
    ]
    frames = [read_scores(run, include_failed=True) for run in score_runs]
    scores = (
        pd.concat(frames, ignore_index=True)
        if len(frames) > 1
        else frames[0]
        if frames
        else pd.DataFrame()
    )
    search_files = [
        Path(run) if Path(run).is_file() else Path(run) / "search.jsonl"
        for run in expanded_runs
        if (Path(run).is_file() and Path(run).name == "search.jsonl")
        or (Path(run) / "search.jsonl").exists()
    ]
    training_files = [
        Path(run) if Path(run).is_file() else Path(run) / "training.jsonl"
        for run in expanded_runs
        if (Path(run).is_file() and Path(run).name == "training.jsonl")
        or (Path(run) / "training.jsonl").exists()
    ]
    return build_report_bundle(
        _analysis_component_frame(scores, None),
        output,
        search=search_files or None,
        training=training_files or None,
        title=title,
        bootstrap_samples=bootstrap_samples,
    )


def _analysis_component_frame(frame: Any, component: str | None) -> Any:
    if component:
        return frame[frame["component"].astype(str).eq(component)].reset_index(drop=True)
    if "primary_component" not in frame or not frame["primary_component"].notna().any():
        return frame.reset_index(drop=True)
    known = frame["primary_component"].notna()
    primary = frame["component"].astype(str).eq(frame["primary_component"].astype(str))
    return frame[~known | primary].reset_index(drop=True)


def command_analyze(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    output = Path(args.output)
    if args.action in {"correlation", "compare", "sensitivity"}:
        frame = read_scores(args.scores, include_failed=True)
        frame = _analysis_component_frame(frame, args.component)
        validate_analysis_scores(
            frame,
            args.action,
            sensitivity_parameter=getattr(args, "parameter", "seed"),
        )
        output.mkdir(parents=True, exist_ok=True)
        if args.action == "correlation":
            table = correlation_table(frame, bootstrap_samples=args.bootstrap_samples)
            table.to_csv(output / "correlations.csv", index=False)
        elif args.action == "compare":
            table = top_k_comparison(frame, k=args.top_k)
            table.to_csv(output / "top_k.csv", index=False)
            rank_aggregation(frame).to_csv(output / "rank_aggregation.csv", index=False)
            proxy_cost_pareto(frame).to_csv(output / "proxy_cost_pareto.csv", index=False)
            transfer_correlation_table(frame).to_csv(output / "transfer.csv", index=False)
        else:
            sample_size_convergence(frame, sizes=args.sample_sizes).to_csv(
                output / "sample_size_convergence.csv", index=False
            )
            rank_stability = sensitivity_rank_table(frame, parameter=args.parameter)
            rank_stability.to_csv(output / "sensitivity_rank.csv", index=False)
            for suffix in ("png", "svg"):
                figure = plot_sensitivity(
                    frame, output / f"sensitivity.{suffix}", parameter=args.parameter
                )
                plt.close(figure)
                figure = plot_sensitivity_rank(
                    rank_stability,
                    output / f"sensitivity_rank.{suffix}",
                    parameter=args.parameter,
                )
                plt.close(figure)
        result = build_report_bundle(
            frame,
            output,
            title=args.title or f"zcp-test {args.action}",
            bootstrap_samples=args.bootstrap_samples,
            top_k=args.top_k,
            sensitivity_parameter=getattr(args, "parameter", "seed"),
            sample_sizes=args.sample_sizes if args.action == "sensitivity" else None,
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        plotter = plot_search if args.action == "search" else plot_training
        artifacts = []
        for suffix in ("png", "svg"):
            path = output / f"{args.action}.{suffix}"
            figure = plotter(args.source, path)
            plt.close(figure)
            artifacts.append(str(path))
        result = {"output_directory": str(output), "artifacts": artifacts}
    _json(result)


def _benchmark_study_frame(args: argparse.Namespace) -> Any:
    import pandas as pd

    frames = [read_scores(source, include_failed=True) for source in args.scores]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = _analysis_component_frame(frame, args.component)
    filters = {
        "dataset": args.dataset,
        "target_metric": args.target_metric,
        "target_split": args.target_split,
        "benchmark_variant": args.benchmark_variant,
    }
    for field, value in filters.items():
        if value is not None:
            if field not in frame:
                raise ValueError(f"Score data does not contain filter field {field}")
            frame = frame[frame[field].astype(str).eq(str(value))]
    if args.target_epoch_budget is not None:
        frame = frame[
            pd.to_numeric(frame["target_epoch_budget"], errors="coerce").eq(
                args.target_epoch_budget
            )
        ]
    if frame.empty:
        raise ValueError("Benchmark analysis filters selected no successful score rows")
    if "target_direction" in frame:
        minimize = frame["target_direction"].astype(str).str.casefold().eq("minimize")
        frame = frame.copy()
        frame.loc[minimize, "target_value"] = -pd.to_numeric(
            frame.loc[minimize, "target_value"], errors="coerce"
        )
        frame.loc[minimize, "target_direction"] = "maximize"
    return frame.reset_index(drop=True)


def command_analyze_benchmark(args: argparse.Namespace) -> None:
    frame = _benchmark_study_frame(args)
    available = sorted(
        value for value in frame["benchmark_id"].dropna().astype(str).unique() if value
    )
    if args.benchmark == "auto":
        if len(available) != 1:
            raise ValueError(
                "--benchmark auto requires exactly one benchmark_id after filtering; "
                f"found {available}"
            )
        benchmark_id = available[0]
    else:
        benchmark_id = args.benchmark
        frame = frame[frame["benchmark_id"].astype(str).eq(benchmark_id)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No score rows match benchmark_id={benchmark_id}")
    default_views = {
        "nasbench101": "budget",
        "nasbench201": "topology",
        "nats_tss": "topology",
        "nats_sss": "size",
        "nasbench301_surrogate": "darts",
        "transnasbench101": "transfer",
        "vitbench101": "architecture",
    }
    view = default_views.get(benchmark_id) if args.view == "auto" else args.view
    if view is None:
        raise ValueError(f"No automatic benchmark study is registered for {benchmark_id}")
    if view == "budget":
        if benchmark_id != "nasbench101":
            raise ValueError("The budget view currently supports only nasbench101")
        if not args.benchmark_path:
            raise ValueError("NAS-Bench-101 budget study requires --benchmark-path")
        load_builtin_benchmarks()
        adapter = BENCHMARKS.create(
            "nasbench101", path=args.benchmark_path, version=args.benchmark_version
        )
        tables = nasbench101_budget_study(
            frame,
            adapter,
            budgets=args.budgets,
            dataset=args.study_dataset,
            split=args.study_split,
            metric_name=args.study_metric,
            repeat_index=args.repeat_index,
            seed_reduction=args.seed_reduction,
            target_direction=args.study_target_direction,
            bootstrap_samples=args.bootstrap_samples,
            top_k=args.top_k,
        )
    elif view == "topology":
        if benchmark_id not in {"nasbench201", "nats_tss"}:
            raise ValueError("The topology view supports nasbench201 and nats_tss")
        tables = topology_study(frame)
    elif view == "size":
        if benchmark_id != "nats_sss":
            raise ValueError("The size view supports nats_sss")
        tables = nats_size_study(frame)
    elif view == "darts":
        if benchmark_id != "nasbench301_surrogate":
            raise ValueError("The DARTS interaction view supports nasbench301_surrogate")
        tables = nasbench301_darts_study(frame)
    elif view == "architecture":
        if benchmark_id != "vitbench101":
            raise ValueError("The architecture view currently supports vitbench101")
        tables = vit_architecture_study(frame)
    elif view == "transfer":
        if benchmark_id != "transnasbench101":
            raise ValueError("The transfer view currently supports transnasbench101")
        tables = transnas_transfer_study(frame)
    else:
        raise ValueError(f"Unsupported benchmark study view: {view}")
    _json(write_benchmark_study(tables, args.output, view=view, benchmark_id=benchmark_id))


def command_monitor(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("monitor interval must be positive")
    destination = args.output or str(Path(args.run) / "reports" / "monitor.html")
    offset = 0
    history: list[dict[str, Any]] = []
    while True:
        try:
            result = refresh_once(
                args.run,
                destination,
                offset=offset,
                title=args.title,
                history=history,
                browser_refresh_seconds=args.interval,
            )
        except ValueError as error:
            if "offset must be between" not in str(error):
                raise
            offset = 0
            history.clear()
            result = refresh_once(
                args.run,
                destination,
                title=args.title,
                history=history,
                browser_refresh_seconds=args.interval,
            )
        history.extend(result["rows"])
        offset = result["next_offset"]
        _json({key: value for key, value in result.items() if key != "rows"})
        if args.once:
            return
        time.sleep(args.interval)


def command_legacy(args: argparse.Namespace) -> None:
    _json({"records": import_pickle(args.source, args.output, args.trusted), "output": args.output})


def command_acceptance(args: argparse.Namespace) -> None:
    from zcp_test.acceptance import reconcile_search_cohort

    if args.action == "freeze-candidates":
        _json(
            freeze_training_candidates(
                search_run=args.search_run,
                training_config_path=args.training_config,
                output=args.output,
                seed=args.seed,
                pool_size=args.pool_size,
                classes=args.classes,
                supporting_search_runs=args.supporting_search_run,
            )
        )
        return
    if args.action == "reconcile-search-cohort":
        _json(
            reconcile_search_cohort(
                cohort_root=args.cohort_root,
                search_runs=args.search_run,
                expected_space=args.expected_space,
                expected_population=args.expected_population,
                expected_seeds=args.expected_seed,
                expected_components=tuple(
                    component.strip()
                    for component in args.expected_components.split(",")
                    if component.strip()
                ),
            )
        )
        return
    raise ValueError(f"Unsupported acceptance action: {args.action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zcp-test")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--catalog")
    doctor.add_argument("--data-root")
    doctor.set_defaults(function=command_doctor)
    gpu = subparsers.add_parser("gpu")
    gpu_actions = gpu.add_subparsers(dest="action", required=True)
    gpu_list = gpu_actions.add_parser("list")
    gpu_list.set_defaults(function=command_gpu)
    data = subparsers.add_parser("data")
    data_actions = data.add_subparsers(dest="action", required=True)
    data_default = str(Path.home() / ".config" / "zcp-test" / "data.json")
    data_list = data_actions.add_parser("list")
    data_list.add_argument("--catalog", default=data_default)
    data_list.set_defaults(function=command_data)
    data_register = data_actions.add_parser("register")
    data_register.add_argument("asset_id")
    data_register.add_argument("path")
    data_register.add_argument("--catalog", default=data_default)
    data_register.add_argument("--version", required=True)
    data_register.add_argument("--sha256")
    data_register.add_argument("--source-url")
    data_register.add_argument("--protocol")
    data_register.add_argument("--trusted", action="store_true")
    data_register.add_argument("--replace", action="store_true")
    data_register.set_defaults(function=command_data)
    data_verify = data_actions.add_parser("verify")
    data_verify.add_argument("asset_id", nargs="?")
    data_verify.add_argument("--all", action="store_true")
    data_verify.add_argument("--root", default=str(Path.home() / "zcp-test-data"))
    data_verify.add_argument("--catalog", default=data_default)
    data_verify.set_defaults(function=command_data)
    data_fetch = data_actions.add_parser("fetch")
    data_fetch.add_argument("asset_id")
    data_fetch.add_argument("--catalog", default=data_default)
    data_fetch.add_argument("--destination")
    data_fetch.set_defaults(function=command_data)
    data_check = data_actions.add_parser("checklist")
    data_check.add_argument("--root", required=True)
    data_check.add_argument("--json", action="store_true")
    data_check.add_argument("--catalog", default=data_default)
    data_check.set_defaults(function=command_data)
    data_bootstrap = data_actions.add_parser("bootstrap")
    data_bootstrap.add_argument("--root", required=True)
    data_bootstrap.add_argument("--benchmarks")
    data_bootstrap.add_argument("--all", action="store_true")
    data_bootstrap.add_argument("--yes", action="store_true")
    data_bootstrap.add_argument("--catalog", default=data_default)
    data_bootstrap.set_defaults(function=command_data)
    data_export = data_actions.add_parser("export-manifest")
    data_export.add_argument("--root", required=True)
    data_export.add_argument("--benchmarks", required=True)
    data_export.add_argument("--output", required=True)
    data_export.add_argument("--catalog", default=data_default)
    data_export.set_defaults(function=command_data)
    data_import = data_actions.add_parser("import-manifest")
    data_import.add_argument("--root", required=True)
    data_import.add_argument("--manifest", required=True)
    data_import.add_argument("--catalog", default=data_default)
    data_import.set_defaults(function=command_data)
    convert_vit = data_actions.add_parser("convert-vit")
    convert_vit.add_argument("--source", required=True)
    convert_vit.add_argument("--output", required=True)
    convert_vit.add_argument(
        "--slice-id", choices=("autoformer_main", "autoformer_ext", "pit"), required=True
    )
    convert_vit.add_argument("--trusted", action="store_true")
    convert_vit.add_argument("--catalog", default=data_default)
    convert_vit.set_defaults(function=command_data)
    convert_imagenet16 = data_actions.add_parser("convert-imagenet16")
    convert_imagenet16.add_argument("--source", required=True)
    convert_imagenet16.add_argument("--output", required=True)
    convert_imagenet16.add_argument("--trusted", action="store_true")
    convert_imagenet16.add_argument("--replace", action="store_true")
    convert_imagenet16.add_argument("--register", action="store_true")
    convert_imagenet16.add_argument("--catalog", default=data_default)
    convert_imagenet16.set_defaults(function=command_data)
    prepare_transnas_input = data_actions.add_parser("prepare-transnas-input")
    prepare_transnas_input.add_argument("--data-root", required=True)
    prepare_transnas_input.add_argument("--split-json", required=True)
    prepare_transnas_input.add_argument("--output")
    prepare_transnas_input.add_argument("--split")
    prepare_transnas_input.add_argument("--verify-files", action="store_true")
    prepare_transnas_input.set_defaults(function=command_data)
    for command, registry in (("benchmark", "benchmark"), ("space", "space")):
        group = subparsers.add_parser(command)
        actions = group.add_subparsers(dest="action", required=True)
        listing = actions.add_parser("list")
        listing.set_defaults(function=lambda args, name=registry: command_registry(args, name))
        inspect = actions.add_parser("inspect")
        inspect.add_argument("name")
        inspect.add_argument("--seed", type=int, default=42)
        if registry == "benchmark":
            inspect.add_argument("--path")
            inspect.add_argument("--trusted", action="store_true")
            inspect.add_argument("--version")
            inspect.add_argument("--slice-id", default="autoformer_main")
            inspect.add_argument("--transnas-space", default="micro")
            inspect.add_argument("--architecture-path")
            inspect.add_argument("--runtime-path")
            inspect.add_argument("--catalog", default=data_default)
            inspect.add_argument("--data-root")
            inspect.add_argument("--start", type=int, default=0)
            inspect.add_argument("--dataset", default="cifar10")
            inspect.add_argument("--split", default="valid")
            inspect.add_argument("--metric-name")
            inspect.add_argument("--epoch-budget", type=int)
            inspect.add_argument("--metric-seed", type=int)
            inspect.add_argument(
                "--metric-seed-reduction", choices=("mean", "min", "max"), default="mean"
            )
            inspect.add_argument("--surrogate-noise", action="store_true")
        inspect.set_defaults(function=lambda args, name=registry: command_registry(args, name))
        if registry == "benchmark":
            sample = actions.add_parser("sample")
            sample.add_argument("name")
            sample.add_argument("--path")
            sample.add_argument("--trusted", action="store_true")
            sample.add_argument("--version")
            sample.add_argument("--slice-id", default="autoformer_main")
            sample.add_argument("--transnas-space", default="micro")
            sample.add_argument("--architecture-path")
            sample.add_argument(
                "--population-count",
                type=int,
                help="finite deterministic candidate corpus size for generated benchmarks",
            )
            sample.add_argument("--runtime-path")
            sample.add_argument("--catalog", default=data_default)
            sample.add_argument("--data-root")
            size = sample.add_mutually_exclusive_group(required=True)
            size.add_argument("--count", type=int)
            size.add_argument("--fraction", type=float)
            sample.add_argument("--seed", type=int, default=0)
            sample.add_argument("--shards", type=int, default=1)
            sample.add_argument("--output", required=True)
            sample.set_defaults(function=command_benchmark_sample)
    proxy = subparsers.add_parser("proxy")
    proxy_actions = proxy.add_subparsers(dest="action", required=True)
    for action in ("list", "matrix"):
        proxy_parser = proxy_actions.add_parser(action)
        proxy_parser.set_defaults(function=command_proxy)
    for action in ("inspect", "validate", "scaffold"):
        proxy_parser = proxy_actions.add_parser(action)
        proxy_parser.add_argument("name")
        proxy_parser.set_defaults(function=command_proxy)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config")
    evaluate_identity = evaluate.add_mutually_exclusive_group(required=False)
    evaluate_identity.add_argument("--space")
    evaluate_identity.add_argument("--benchmark")
    evaluate.add_argument("--benchmark-path")
    evaluate.add_argument("--trusted", action="store_true")
    evaluate.add_argument("--benchmark-version")
    evaluate.add_argument("--architecture-path")
    evaluate.add_argument("--runtime-benchmark-path")
    evaluate.add_argument("--slice-id", default="autoformer_main")
    evaluate.add_argument("--transnas-space", default="micro")
    evaluate.add_argument("--start", type=int, default=0)
    evaluate.add_argument("--sample-manifest")
    evaluate.add_argument("--sample-shard", type=int)
    evaluate.add_argument("--dataset", default="cifar10")
    evaluate.add_argument("--target-metric")
    evaluate.add_argument("--target-split", default="valid")
    evaluate.add_argument("--epoch-budget", type=int)
    evaluate.add_argument("--metric-seed", type=int)
    evaluate.add_argument("--metric-seed-reduction", choices=("mean", "min", "max"), default="mean")
    evaluate.add_argument(
        "--target-direction", choices=("auto", "maximize", "minimize"), default="auto"
    )
    evaluate.add_argument("--surrogate-noise", action="store_true")
    evaluate.add_argument("--allow-approximation", action="store_true")
    evaluate.add_argument(
        "--weight-mode",
        choices=("independent_scratch", "ofa_inherited"),
        default="independent_scratch",
    )
    evaluate.add_argument("--model-checkpoint")
    evaluate.add_argument("--bn-recalibration-batches", type=int, default=0)
    evaluate.add_argument("--bn-recalibration-batch-size", type=int)
    evaluate.add_argument("--proxies", default="er,naswot,synflow,gradnorm,params,flops")
    evaluate.add_argument("--count", type=int, default=1)
    evaluate.add_argument("--seed", type=int, default=42)
    _add_gpu_arguments(evaluate)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--input-size", type=int, default=32)
    evaluate.add_argument("--classes", type=int, default=10)
    evaluate.add_argument(
        "--input-source", choices=("dataset", "random", "noise"), default="dataset"
    )
    evaluate.add_argument("--data-root")
    evaluate.add_argument("--catalog", default=data_default)
    evaluate.add_argument("--output", default="runs/evaluate")
    evaluate.set_defaults(function=command_evaluate)
    correlate = subparsers.add_parser("correlate")
    correlate.add_argument("--config")
    correlate.add_argument("--scores", required=True)
    correlate.add_argument("--targets", required=True)
    correlate.add_argument("--output", required=True)
    correlate.add_argument("--id-field", default="architecture_id")
    correlate.add_argument("--score-field", default="score")
    correlate.add_argument("--component")
    correlate.add_argument("--target-field", required=True)
    correlate.add_argument(
        "--target-direction", choices=("maximize", "minimize"), default="maximize"
    )
    correlate.add_argument("--ndcg-k", type=int, default=10)
    correlate.set_defaults(function=command_correlate)
    search = subparsers.add_parser("search")
    search.add_argument("--config")
    search.add_argument(
        "--controller",
        choices=("generic", "plainnet_source_aligned"),
        default="generic",
    )
    search.add_argument("--space")
    search.add_argument("--proxy", default="er")
    search.add_argument(
        "--aggregator",
        choices=("primary", "az_nas_log_rank"),
        default="primary",
    )
    search.add_argument("--population", type=int, default=10)
    search.add_argument("--generations", type=int, default=3)
    search.add_argument("--elite-ratio", type=float, default=0.2)
    search.add_argument("--valid-candidates", type=int)
    search.add_argument("--flops-target", choices=("450m", "600m", "1g"))
    search.add_argument(
        "--max-parameters",
        type=int,
        help="reject generic-search candidates above this parameter count before ZCP evaluation",
    )
    search.add_argument(
        "--max-macs",
        type=int,
        help="reject generic-search candidates above this declared MAC/complexity count before ZCP evaluation",
    )
    search.add_argument(
        "--constraint-max-attempts",
        type=int,
        default=10_000,
        help="fail closed after this many consecutive resource-constraint attempts",
    )
    search.add_argument("--seed", type=int, default=42)
    search.add_argument(
        "--resume",
        help="resume from search-state.json with an identical scientific protocol",
    )
    search.add_argument("--allow-approximation", action="store_true")
    search.add_argument("--trusted", action="store_true")
    search.add_argument(
        "--weight-mode",
        choices=("independent_scratch", "ofa_inherited"),
        default="independent_scratch",
    )
    search.add_argument("--model-checkpoint")
    search.add_argument("--bn-recalibration-batches", type=int, default=0)
    search.add_argument("--bn-recalibration-batch-size", type=int)
    _add_gpu_arguments(search)
    search.add_argument("--batch-size", type=int, default=4)
    search.add_argument("--input-size", type=int, default=32)
    search.add_argument("--classes", type=int, default=10)
    search.add_argument("--dataset", default="cifar10")
    search.add_argument("--input-source", choices=("dataset", "random", "noise"), default="dataset")
    search.add_argument("--data-root")
    search.add_argument("--catalog", default=data_default)
    search.add_argument("--output", default="runs/search")
    search.set_defaults(function=command_search)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--architecture")
    train.add_argument("--resume")
    train.add_argument(
        "--trusted",
        action="store_true",
        help="allow loading the explicitly supplied trusted checkpoint",
    )
    train.add_argument("--epochs", type=int)
    train.add_argument("--seed", type=int, default=42)
    _add_gpu_arguments(train)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--input-size", type=int)
    train.add_argument("--classes", type=int)
    train.add_argument("--data-root")
    train.add_argument("--catalog", default=data_default)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument(
        "--valid-workers",
        type=int,
        help="validation DataLoader workers; defaults to --workers for compatibility",
    )
    train.add_argument("--data-fraction", type=float, default=1.0)
    train.add_argument("--output", default="runs/training")
    training_mode = train.add_mutually_exclusive_group()
    training_mode.add_argument("--smoke", action="store_true")
    train.add_argument(
        "--full-batch-smoke",
        action="store_true",
        help="with --smoke, execute one synthetic epoch at the configured micro-batch for memory validation",
    )
    training_mode.add_argument(
        "--acceptance-smoke",
        action="store_true",
        help="run a code-approved real-data 1%% acceptance protocol without enabling formal training",
    )
    training_mode.add_argument(
        "--real-data-preflight",
        action="store_true",
        help="run exactly one real-data epoch for timing and pipeline validation only",
    )
    train.set_defaults(function=command_train)
    report = subparsers.add_parser("report")
    report.set_defaults(action="legacy")
    report.add_argument("--config")
    report.add_argument("--source")
    report.add_argument("--output")
    report.add_argument("--format", choices=("csv", "html", "plot"), default="csv")
    report.add_argument("--kind", choices=("training", "search"), default="training")
    report.add_argument("--title", default="zcp-test report")
    report.set_defaults(function=command_report)
    report_actions = report.add_subparsers(dest="action")
    report_bundle = report_actions.add_parser("bundle")
    report_bundle.add_argument("runs", nargs="+")
    report_bundle.add_argument("--output", required=True)
    report_bundle.add_argument("--title", default="zcp-test report bundle")
    report_bundle.add_argument("--bootstrap-samples", type=int, default=1000)
    report_bundle.set_defaults(function=command_report)
    analyze = subparsers.add_parser("analyze")
    analyze_actions = analyze.add_subparsers(dest="action", required=True)
    for action in ("correlation", "compare", "sensitivity"):
        analysis = analyze_actions.add_parser(action)
        analysis.add_argument("--scores", nargs="+", required=True)
        analysis.add_argument("--output", required=True)
        analysis.add_argument("--component")
        analysis.add_argument("--title")
        analysis.add_argument("--bootstrap-samples", type=int, default=1000)
        analysis.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
        if action == "sensitivity":
            analysis.add_argument("--parameter", default="seed")
            analysis.add_argument(
                "--sample-sizes",
                type=int,
                nargs="+",
                default=[10, 25, 50, 100],
                help="sample counts used for convergence analysis",
            )
        analysis.set_defaults(function=command_analyze)
    for action in ("search", "training"):
        analysis = analyze_actions.add_parser(action)
        analysis.add_argument("--source", required=True)
        analysis.add_argument("--output", required=True)
        analysis.set_defaults(function=command_analyze)
    benchmark_analysis = analyze_actions.add_parser("benchmark")
    benchmark_analysis.add_argument("--scores", nargs="+", required=True)
    benchmark_analysis.add_argument("--output", required=True)
    benchmark_analysis.add_argument(
        "--benchmark",
        default="auto",
        choices=(
            "auto",
            "nasbench101",
            "nasbench201",
            "nats_tss",
            "nats_sss",
            "nasbench301_surrogate",
            "transnasbench101",
            "vitbench101",
        ),
    )
    benchmark_analysis.add_argument(
        "--view",
        default="auto",
        choices=("auto", "budget", "topology", "size", "darts", "architecture", "transfer"),
    )
    benchmark_analysis.add_argument("--component")
    benchmark_analysis.add_argument("--dataset")
    benchmark_analysis.add_argument("--target-metric")
    benchmark_analysis.add_argument("--target-split")
    benchmark_analysis.add_argument("--target-epoch-budget", type=int)
    benchmark_analysis.add_argument("--benchmark-variant")
    benchmark_analysis.add_argument("--bootstrap-samples", type=int, default=1000)
    benchmark_analysis.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 50])
    benchmark_analysis.add_argument("--benchmark-path")
    benchmark_analysis.add_argument("--benchmark-version", default="full")
    benchmark_analysis.add_argument("--budgets", type=int, nargs="+", default=[4, 12, 36, 108])
    benchmark_analysis.add_argument("--study-dataset", default="cifar10")
    benchmark_analysis.add_argument("--study-split", default="valid")
    benchmark_analysis.add_argument("--study-metric", default="final_accuracy")
    benchmark_analysis.add_argument("--repeat-index", type=int)
    benchmark_analysis.add_argument(
        "--seed-reduction", choices=("mean", "min", "max"), default="mean"
    )
    benchmark_analysis.add_argument(
        "--study-target-direction", choices=("maximize", "minimize"), default="maximize"
    )
    benchmark_analysis.set_defaults(function=command_analyze_benchmark)
    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("run")
    monitor.add_argument("--output")
    monitor.add_argument("--interval", type=float, default=5.0)
    monitor.add_argument("--title", default="zcp-test monitor")
    monitor.add_argument("--once", action="store_true")
    monitor.set_defaults(function=command_monitor)
    acceptance = subparsers.add_parser("acceptance")
    acceptance_actions = acceptance.add_subparsers(dest="action", required=True)
    freeze_candidates = acceptance_actions.add_parser("freeze-candidates")
    freeze_candidates.add_argument("--search-run", required=True)
    freeze_candidates.add_argument(
        "--supporting-search-run",
        action="append",
        default=[],
        help="Completed same-protocol run used only as seed-robustness provenance; repeatable",
    )
    freeze_candidates.add_argument("--training-config", required=True)
    freeze_candidates.add_argument("--output", required=True)
    freeze_candidates.add_argument("--seed", type=int, default=20260731)
    freeze_candidates.add_argument("--pool-size", type=int, default=32)
    freeze_candidates.add_argument("--classes", type=int, default=1000)
    freeze_candidates.set_defaults(function=command_acceptance)
    reconcile_cohort = acceptance_actions.add_parser("reconcile-search-cohort")
    reconcile_cohort.add_argument("--cohort-root", required=True)
    reconcile_cohort.add_argument("--search-run", action="append", required=True)
    reconcile_cohort.add_argument("--expected-space", required=True)
    reconcile_cohort.add_argument("--expected-population", type=int, required=True)
    reconcile_cohort.add_argument("--expected-seed", action="append", type=int, required=True)
    reconcile_cohort.add_argument("--expected-components", required=True)
    reconcile_cohort.set_defaults(function=command_acceptance)
    legacy = subparsers.add_parser("legacy")
    legacy_actions = legacy.add_subparsers(dest="action", required=True)
    legacy_import = legacy_actions.add_parser("import")
    legacy_import.add_argument("--source", required=True)
    legacy_import.add_argument("--output", required=True)
    legacy_import.add_argument("--trusted", action="store_true")
    legacy_import.set_defaults(function=command_legacy)
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_arguments)
    explicitly_set = {
        argument[2:].split("=", 1)[0].replace("-", "_")
        for argument in raw_arguments
        if argument.startswith("--")
    }
    configured_options: set[str] = set()
    config_path = getattr(args, "config", None)
    if config_path:
        loaded = load_config(config_path)
        values = loaded.get(args.command, loaded)
        if not isinstance(values, dict):
            raise ValueError(f"Config section {args.command!r} must be a mapping")
        runtime_keys = {key for key in values if hasattr(args, key)}
        allowed_keys = runtime_keys | (
            set(TRAIN_PROFILE_KEYS) if args.command == "train" else set()
        )
        reject_unknown_config_keys(values, allowed_keys, args.command)
        for key, value in values.items():
            if key == "trusted" and value and key not in explicitly_set:
                raise PermissionError(
                    "trusted execution must be acknowledged explicitly on the CLI"
                )
            if key not in explicitly_set and hasattr(args, key):
                setattr(args, key, value)
                configured_options.add(key)
    args._explicit_options = frozenset(explicitly_set | configured_options)
    args.function(args)


if __name__ == "__main__":
    main()
