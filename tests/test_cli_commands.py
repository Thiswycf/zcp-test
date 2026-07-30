import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

import zcp_test.cli as cli
from zcp_test.types import Architecture


def _output(capsys):
    return json.loads(capsys.readouterr().out)


class _TinyReferenceSpace:
    search_space_id = "tiny_reference"
    model_family = "cnn"
    model_fidelity = "reference_model"
    implementation_source = "test-fixture"
    implementation_commit = "fixture"

    def __init__(self):
        self.counter = 0

    def _architecture(self):
        self.counter += 1
        return Architecture(self.search_space_id, f"a-{self.counter}", {"value": self.counter})

    def sample(self, seed=None):
        return self._architecture()

    def canonicalize(self, specification):
        return Architecture(self.search_space_id, "inline", dict(specification))

    def mutate(self, architecture, seed=None):
        return self._architecture()

    def crossover(self, left, right, seed=None):
        return self._architecture()

    def build_model(self, architecture, num_classes):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 8 * 8, num_classes),
        )


def test_inherited_weight_mode_rejects_wrong_space_and_missing_asset(monkeypatch):
    arguments = argparse.Namespace(weight_mode="ofa_inherited", model_checkpoint=None)
    with pytest.raises(ValueError, match="only for ofa_proxyless_mbv2"):
        cli._prepare_model_weights(
            arguments,
            argparse.Namespace(search_space_id="darts"),
        )

    class MissingRegistry:
        def __init__(self, catalog):
            self.catalog = catalog

        def get(self, asset_id):
            raise KeyError(asset_id)

    monkeypatch.setattr(cli, "DataRegistry", MissingRegistry)
    arguments.catalog = "missing-catalog.json"
    with pytest.raises(FileNotFoundError, match="data bootstrap"):
        cli._prepare_model_weights(
            arguments,
            argparse.Namespace(search_space_id="ofa_proxyless_mbv2"),
        )

class _TinyVisionDataset(torch.utils.data.Dataset):
    def __init__(self, *args, transform=None, **kwargs):
        self.transform = transform if transform is not None else (args[1] if len(args) > 1 else None)
        self.targets = [0, 0, 1, 1]
        self.samples = [(f"image-{index}", target) for index, target in enumerate(self.targets)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), self.targets[index]


def test_cli_doctor_and_gpu_list(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "data_checklist", lambda root, catalog: [{"state": "missing"}])
    cli.main(["doctor", "--data-root", str(tmp_path)])
    report = _output(capsys)
    assert report["python"]
    assert report["benchmark_data"] == [{"state": "missing"}]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-a")
    monkeypatch.setattr(
        cli,
        "enumerate_gpus",
        lambda: [
            {"uuid": "GPU-a", "pci_bus_id": "0000:02:00.0"},
            {"uuid": "GPU-b", "pci_bus_id": "0000:01:00.0"},
        ],
    )
    cli.main(["gpu", "list"])
    rows = _output(capsys)
    assert rows[0]["pci_order"] == 0
    assert rows[0]["visible_logical_index"] == 1
    assert rows[1]["visible_logical_index"] == 0


def test_cli_registry_lists_and_inspects(capsys):
    cli.main(["benchmark", "list"])
    assert "nasbench101" in _output(capsys)
    cli.main(["space", "list"])
    assert "darts" in _output(capsys)
    cli.main(["space", "inspect", "autoformer", "--seed", "3"])
    space = _output(capsys)
    assert space["search_space_id"] == "autoformer"
    assert space["model_fidelity"] == "reference_model"

    cli.main(["proxy", "list"])
    assert "er" in _output(capsys)
    cli.main(["proxy", "inspect", "er"])
    proxy = _output(capsys)
    assert proxy["proxy_id"] == "er"
    assert proxy["primary_component"] == "mean"
    cli.main(["proxy", "matrix"])
    matrix = _output(capsys)
    assert any(row["proxy_id"] == "er" for row in matrix)
    cli.main(["proxy", "validate", "params"])
    validation = _output(capsys)
    assert validation["status"] == "ok"


def test_cli_data_register_list_verify_and_replace(capsys, tmp_path):
    catalog = tmp_path / "catalog.json"
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    digest = hashlib.sha256(b"asset").hexdigest()
    common = ["--catalog", str(catalog)]
    cli.main([
        "data", "register", "fixture", str(asset), "--version", "1", "--sha256", digest,
        "--protocol", "fixture", *common,
    ])
    assert _output(capsys)["asset_id"] == "fixture"
    cli.main(["data", "list", *common])
    assert _output(capsys)[0]["path"] == str(asset.resolve())
    cli.main(["data", "verify", "fixture", *common])
    assert _output(capsys)["valid"] is True
    with pytest.raises(KeyError):
        cli.main([
            "data", "register", "fixture", str(asset), "--version", "2", *common,
        ])
    cli.main([
        "data", "register", "fixture", str(asset), "--version", "2", "--replace", *common,
    ])
    assert _output(capsys)["version"] == "2"


def test_cli_legacy_and_report_validation(capsys, tmp_path):
    import pickle

    source = tmp_path / "legacy.pkl"
    output = tmp_path / "legacy.jsonl"
    source.write_bytes(pickle.dumps([1, 2]))
    with pytest.raises(PermissionError):
        cli.main(["legacy", "import", "--source", str(source), "--output", str(output)])
    cli.main([
        "legacy", "import", "--source", str(source), "--output", str(output), "--trusted",
    ])
    assert _output(capsys)["records"] == 2
    with pytest.raises(ValueError, match="requires --source"):
        cli.main(["report"])


def test_cli_path_resolution_failures(tmp_path):
    args = argparse.Namespace(data_root=None, catalog=None)
    assert cli._resolve_data_root(args, "cifar10") is None

    args = argparse.Namespace(runtime_benchmark_path=str(tmp_path / "missing"), catalog=None)
    with pytest.raises(FileNotFoundError, match="runtime ensemble"):
        cli._resolve_nb301_runtime_path(args)

    args = argparse.Namespace(
        benchmark_path=None,
        benchmark="nasbench101",
        transnas_space="micro",
        slice_id="autoformer_main",
        catalog=str(tmp_path / "missing-catalog.json"),
        data_root=str(tmp_path / "data"),
    )
    with pytest.raises(FileNotFoundError, match="data bootstrap"):
        cli._resolve_benchmark_path(args)


def test_inline_architecture_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="must be an object"):
        cli._load_architecture_spec("[]")
    with pytest.raises(ValueError, match="'spec' must be an object"):
        cli._load_architecture_spec('{"spec": 1}')
    missing = tmp_path / "looks-like-file.json"
    with pytest.raises(ValueError, match="existing JSON file"):
        cli._load_architecture_spec(str(missing))


def test_cli_search_end_to_end_with_tiny_reference_space(monkeypatch, capsys, tmp_path):
    space = _TinyReferenceSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    output = tmp_path / "search"
    cli.main([
        "search",
        "--space", "tiny_reference",
        "--proxy", "params",
        "--population", "2",
        "--generations", "1",
        "--elite-ratio", "0.5",
        "--device", "cpu",
        "--input-source", "random",
        "--batch-size", "2",
        "--input-size", "8",
        "--classes", "3",
        "--output", str(output),
    ])
    result = _output(capsys)
    run = Path(result["run"])
    assert result["architecture"]["search_space_id"] == "tiny_reference"
    assert (run / "search.jsonl").exists()
    assert (run / "best_architecture.json").exists()


def test_cli_training_smoke_end_to_end_with_tiny_reference_space(
    monkeypatch, capsys, tmp_path
):
    space = _TinyReferenceSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    config = tmp_path / "training.yaml"
    config.write_text(
        "space: tiny_reference\n"
        "dataset: cifar10\n"
        "input_size: 8\n"
        "epochs: 1\n"
        "optimizer: sgd\n"
        "learning_rate: 0.01\n"
        "weight_decay: 0.0\n"
        "scheduler: none\n"
        "batch_size: 2\n"
        "formal_training_ready: true\n"
        "protocol: test-smoke\n",
        encoding="utf-8",
    )
    output = tmp_path / "training"
    cli.main([
        "train",
        "--config", str(config),
        "--smoke",
        "--device", "cpu",
        "--classes", "3",
        "--output", str(output),
    ])
    result = _output(capsys)
    run = Path(result["run"])
    assert result["last_epoch"] == 0
    assert (run / "training.jsonl").exists()
    assert (run / "checkpoints" / "last.pt").exists()


def test_cli_distributed_training_uses_shared_rank_zero_run(
    monkeypatch, capsys, tmp_path
):
    from contextlib import contextmanager

    space = _TinyReferenceSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    @contextmanager
    def training_device(arguments, world_size, rank, local_rank):
        yield torch.device("cpu"), {
            "selection_strategy": "test-ddp",
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        }

    monkeypatch.setattr(cli, "_training_device", training_device)
    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        lambda model, **kwargs: model,
    )
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda payload, src: None)
    config = tmp_path / "training-ddp.yaml"
    config.write_text(
        "space: tiny_reference\n"
        "dataset: cifar10\n"
        "input_size: 8\n"
        "epochs: 1\n"
        "optimizer: sgd\n"
        "learning_rate: 0.01\n"
        "weight_decay: 0.0\n"
        "scheduler: none\n"
        "batch_size: 2\n"
        "formal_training_ready: true\n"
        "protocol: test-smoke\n",
        encoding="utf-8",
    )
    output = tmp_path / "training-ddp"
    cli.main(
        [
            "train",
            "--config",
            str(config),
            "--smoke",
            "--classes",
            "3",
            "--output",
            str(output),
        ]
    )
    result = _output(capsys)
    run = Path(result["run"])
    assert len(list(output.iterdir())) == 1
    assert (run / "training.jsonl").exists()
    assert json.loads((run / "manifest.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_real_loaders_cover_cifar_and_imagenet_protocols(monkeypatch, tmp_path):
    from torchvision import datasets

    monkeypatch.setattr(datasets, "CIFAR10", _TinyVisionDataset)
    monkeypatch.setattr(datasets, "CIFAR100", _TinyVisionDataset)
    monkeypatch.setattr(datasets, "ImageFolder", _TinyVisionDataset)

    train, valid = cli._real_loaders(
        "cifar10",
        str(tmp_path),
        batch_size=2,
        input_size=8,
        workers=0,
        config={"cutout_length": 4},
        fraction=0.5,
        seed=3,
    )
    assert len(train.dataset) == 2
    assert len(valid.dataset) == 2

    (tmp_path / "val").mkdir()
    train, valid = cli._real_loaders(
        "imagenet1k",
        str(tmp_path),
        batch_size=2,
        input_size=8,
        workers=0,
        config={},
        seed=4,
    )
    assert len(train.dataset) == 4
    assert len(valid.dataset) == 4

    monkeypatch.setattr("timm.data.create_transform", lambda **kwargs: "train-transform")
    train, _ = cli._real_loaders(
        "imagenet1k",
        str(tmp_path),
        batch_size=2,
        input_size=8,
        workers=0,
        config={
            "auto_augment": "rand-m9",
            "random_erase_probability": 0.25,
            "random_erase_mode": "pixel",
            "random_erase_count": 1,
        },
        seed=5,
    )
    assert train.dataset.transform == "train-transform"

    train, _ = cli._real_loaders(
        "imagenet1k",
        str(tmp_path),
        batch_size=2,
        input_size=8,
        workers=0,
        config={
            "repeated_augmentation": True,
            "repeated_augmentation_repeats": 3,
            "repeated_augmentation_selected_round": 0,
        },
        seed=6,
    )
    assert train.sampler.__class__.__name__ == "RepeatAugSampler"
    assert train.batch_sampler.sampler is train.sampler


def test_distributed_training_device_uses_launcher_visible_rank(monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.append(("set", rank)))
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: calls.append(("destroy", None)),
    )
    arguments = argparse.Namespace(device=None)
    with cli._training_device(arguments, 2, 1, 1) as (device, selection):
        assert device == torch.device("cuda:1")
        assert selection["selected_visible_device"] == "GPU-second"
        assert selection["selection_strategy"] == "torchrun_launcher_managed"
    assert calls[0] == ("set", 1)
    assert calls[-1] == ("destroy", None)


@pytest.mark.parametrize(
    ("environment", "device", "message"),
    [
        ({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}, "cpu", "cuda:LOCAL_RANK"),
        ({"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}, None, "PCI_BUS_ID"),
        ({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-a"}, None, "WORLD_SIZE"),
    ],
)
def test_distributed_training_device_rejects_unsafe_launcher_state(
    monkeypatch, environment, device, message
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises((ValueError, RuntimeError), match=message):
        with cli._training_device(argparse.Namespace(device=device), 2, 0, 0):
            pass



def test_distributed_training_device_rejects_cuda_and_local_rank_mismatch(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        with cli._training_device(argparse.Namespace(device=None), 2, 0, 0):
            pass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        with cli._training_device(argparse.Namespace(device=None), 2, 0, 2):
            pass


def test_distributed_run_context_shares_rank_zero_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda payload, src: None)
    with cli._training_run_context(
        tmp_path,
        ["train"],
        {"protocol": "fixture"},
        {"distributed": {"world_size": 2}},
        2,
        0,
    ) as run:
        assert run.directory.parent == tmp_path
        manifest = run.manifest_path
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "completed"

    shared_directory = tmp_path / "shared"
    shared_directory.mkdir()

    def broadcast(payload, src):
        payload[0] = {"directory": str(shared_directory), "run_id": "shared-run"}

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    with cli._training_run_context(tmp_path, ["train"], {}, {}, 2, 1) as run:
        assert run.directory == shared_directory
        assert run.run_id == "shared-run"

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda payload, src: None)
    with pytest.raises(RuntimeError, match="did not broadcast"):
        with cli._training_run_context(tmp_path, ["train"], {}, {}, 2, 1):
            pass

    failed_root = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected"):
        with cli._training_run_context(failed_root, ["train"], {}, {}, 2, 0):
            raise ValueError("injected")
    failed_manifest = next(failed_root.glob("*/manifest.json"))
    assert json.loads(failed_manifest.read_text(encoding="utf-8"))["status"] == "failed"

    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(InterruptedError, match="injected"):
        with cli._training_run_context(interrupted_root, ["train"], {}, {}, 2, 0):
            raise InterruptedError("injected")
    interrupted_manifest = next(interrupted_root.glob("*/manifest.json"))
    assert json.loads(interrupted_manifest.read_text(encoding="utf-8"))["status"] == "interrupted"


def test_checkpoint_lineage_records_hash_and_source_run(tmp_path):
    run = tmp_path / "source-run"
    checkpoint = run / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run / "manifest.json").write_text('{"run_id": "source-id"}', encoding="utf-8")

    lineage = cli._checkpoint_lineage(checkpoint)

    assert lineage["checkpoint"] == str(checkpoint.resolve())
    assert lineage["source_run_id"] == "source-id"
    assert lineage["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()


def test_cli_data_lifecycle_control_paths(monkeypatch, capsys, tmp_path):
    records = [{
        "benchmark_id": "nasbench101",
        "version": "full",
        "state": "missing",
        "catalog_state": "missing",
        "estimated_bytes": 1024,
        "remediation": "bootstrap",
    }]
    monkeypatch.setattr(cli, "data_checklist", lambda root, catalog: records)
    cli.main(["data", "checklist", "--root", str(tmp_path), "--json"])
    assert _output(capsys) == records
    cli.main(["data", "checklist", "--root", str(tmp_path)])
    assert "nasbench101" in capsys.readouterr().out

    with pytest.raises(ValueError, match="Specify --benchmarks"):
        cli.main(["data", "bootstrap", "--root", str(tmp_path), "--yes"])
    monkeypatch.setattr(
        cli,
        "bootstrap_benchmarks",
        lambda root, selected, catalog=None: {"selected": selected, "root": root},
    )
    cli.main([
        "data", "bootstrap", "--root", str(tmp_path),
        "--benchmarks", "nasbench101", "--yes",
    ])
    assert _output(capsys)["selected"] == ["nasbench101"]
    cli.main(["data", "bootstrap", "--root", str(tmp_path), "--all", "--yes"])
    assert _output(capsys)["selected"] == ["nasbench101"]

    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(cli, "export_data_manifest", lambda root, output, selected: manifest)
    cli.main([
        "data", "export-manifest", "--root", str(tmp_path),
        "--benchmarks", "nasbench101", "--output", str(manifest),
    ])
    assert _output(capsys)["manifest"] == str(manifest)

    monkeypatch.setattr(cli, "verify_data_manifest", lambda root, path: {"valid": True})
    cli.main([
        "data", "import-manifest", "--root", str(tmp_path), "--manifest", str(manifest),
    ])
    assert _output(capsys)["valid"] is True
    monkeypatch.setattr(cli, "verify_data_manifest", lambda root, path: {"valid": False})
    with pytest.raises(RuntimeError, match="failed manifest"):
        cli.main([
            "data", "import-manifest", "--root", str(tmp_path),
            "--manifest", str(manifest),
        ])
    _output(capsys)


def test_cli_data_verify_all_and_vit_conversion(monkeypatch, capsys, tmp_path):
    ready = [{"state": "ready"}]
    monkeypatch.setattr(cli, "data_checklist", lambda root, catalog: ready)
    cli.main(["data", "verify", "--all", "--root", str(tmp_path)])
    assert _output(capsys) == ready
    monkeypatch.setattr(cli, "data_checklist", lambda root, catalog: [{"state": "corrupt"}])
    with pytest.raises(RuntimeError, match="not ready"):
        cli.main(["data", "verify", "--all", "--root", str(tmp_path)])
    _output(capsys)

    converted = tmp_path / "vit.jsonl"
    monkeypatch.setattr(cli, "convert_vitbench101", lambda *args, **kwargs: converted)
    cli.main([
        "data", "convert-vit",
        "--source", str(tmp_path / "trusted.pth"),
        "--output", str(converted),
        "--slice-id", "pit",
        "--trusted",
    ])
    assert _output(capsys) == {"path": str(converted), "slice_id": "pit"}
