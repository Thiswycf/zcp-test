import argparse
import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd
import torch
import yaml

import zcp_test.cli as cli
from zcp_test.artifacts import read_jsonl
from zcp_test.types import Architecture


def _output(capsys):
    return json.loads(capsys.readouterr().out)


def test_cli_documentation_audit_inventory_matches_parser():
    parser = cli.build_parser()

    def choices(value):
        action = next(
            (
                item
                for item in value._actions
                if isinstance(item, argparse._SubParsersAction)
            ),
            None,
        )
        return {} if action is None else action.choices

    top_level = choices(parser)
    nested = {
        name: sorted(choices(subparser))
        for name, subparser in sorted(top_level.items())
        if choices(subparser)
    }
    direct = sorted(set(top_level) - set(nested))
    evidence_path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/cli_documentation_audit_20260804.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["level_one_commands"] == sorted(top_level)
    assert evidence["direct_endpoints"] == direct
    assert evidence["level_two_commands"] == nested
    assert evidence["counts"] == {
        "level_one_commands": len(top_level),
        "level_two_commands": sum(map(len, nested.values())),
        "direct_endpoints": len(direct),
        "total_leaf_endpoints": len(direct) + sum(map(len, nested.values())),
        "p0_documentation_gaps": 0,
    }


def test_evaluation_summary_does_not_count_unsupported_as_failed():
    assert cli._evaluation_status_summary(
        ["ok", "ok", "failed", "unsupported", "unsupported", "skipped"]
    ) == {
        "succeeded": 2,
        "failed": 1,
        "unsupported": 2,
        "skipped": 1,
        "non_ok": 4,
    }


def test_search_model_seed_is_architecture_stable_and_seed_sensitive():
    first = cli._search_model_seed(17, "architecture-a")

    assert first == cli._search_model_seed(17, "architecture-a")
    assert first != cli._search_model_seed(18, "architecture-a")
    assert first != cli._search_model_seed(17, "architecture-b")
    assert 0 <= first < 2**32


def test_imagenet_tf_color_distortion_matches_proxyless_protocol():
    from torchvision import transforms

    train_transform, valid_transform = cli._imagenet_transforms(
        224,
        {"resize_scale": 0.08, "color_distortion": "tf"},
    )
    crop = next(
        transform
        for transform in train_transform.transforms
        if isinstance(transform, transforms.RandomResizedCrop)
    )
    jitter = next(
        transform
        for transform in train_transform.transforms
        if isinstance(transform, transforms.ColorJitter)
    )
    resize = next(
        transform
        for transform in valid_transform.transforms
        if isinstance(transform, transforms.Resize)
    )

    assert crop.scale == pytest.approx((0.08, 1.0))
    assert jitter.brightness == pytest.approx((1 - 32 / 255, 1 + 32 / 255))
    assert jitter.saturation == pytest.approx((0.5, 1.5))
    assert jitter.contrast is None
    assert jitter.hue is None
    assert resize.size == 256


def test_imagenet_aznas_augmentation_includes_bicubic_jitter_and_lighting():
    from torchvision import transforms

    train_transform, _ = cli._imagenet_transforms(
        224,
        {"resize_scale": 0.08, "color_distortion": "aznas_imagenet"},
    )
    crop = next(
        transform
        for transform in train_transform.transforms
        if isinstance(transform, transforms.RandomResizedCrop)
    )
    jitter = next(
        transform
        for transform in train_transform.transforms
        if isinstance(transform, transforms.ColorJitter)
    )

    assert crop.interpolation == transforms.InterpolationMode.BICUBIC
    assert jitter.brightness == pytest.approx((0.6, 1.4))
    assert jitter.contrast == pytest.approx((0.6, 1.4))
    assert jitter.saturation == pytest.approx((0.6, 1.4))
    assert jitter.hue is None
    assert type(train_transform.transforms[-2]).__name__ == "AlexNetLighting"


def test_deterministic_training_rejects_cudnn_benchmark():
    with pytest.raises(ValueError, match="incompatible"):
        cli._seed_training(7, 0, True, cudnn_benchmark=True)
    cli._seed_training(7, 0, False)


def test_json_output_does_not_fail_completed_work_on_broken_pipe(monkeypatch):
    class BrokenStream:
        def write(self, value):
            del value
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli.sys, "stdout", BrokenStream())

    cli._json({"status": "completed"})

    replacement = cli.sys.stdout
    assert replacement.writable()
    replacement.close()


class _TinyReferenceSpace:
    search_space_id = "tiny_reference"
    model_family = "cnn"
    model_fidelity = "reference_model"
    implementation_source = "test-fixture"
    implementation_commit = "fixture"

    def _architecture(self, seed):
        value = int(seed)
        return Architecture(self.search_space_id, f"a-{value}", {"value": value})

    def sample(self, seed=None):
        return self._architecture(seed)

    def canonicalize(self, specification):
        return self._architecture(specification["value"])

    def mutate(self, architecture, seed=None):
        return self._architecture(seed)

    def crossover(self, left, right, seed=None):
        return self._architecture(seed)

    def build_model(self, architecture, num_classes):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 8 * 8, num_classes),
        )


def test_analysis_defaults_to_declared_primary_components():
    frame = pd.DataFrame(
        [
            {"proxy_id": "er", "component": "mean", "primary_component": "mean"},
            {"proxy_id": "er", "component": "sum", "primary_component": "mean"},
            {"proxy_id": "synflow", "component": "score", "primary_component": "score"},
            {"proxy_id": "legacy", "component": "legacy-a", "primary_component": None},
            {"proxy_id": "legacy", "component": "legacy-b", "primary_component": None},
        ]
    )

    primary = cli._analysis_component_frame(frame, None)
    auxiliary = cli._analysis_component_frame(frame, "sum")

    assert list(primary[["proxy_id", "component"]].itertuples(index=False, name=None)) == [
        ("er", "mean"),
        ("synflow", "score"),
        ("legacy", "legacy-a"),
        ("legacy", "legacy-b"),
    ]
    assert auxiliary[["proxy_id", "component"]].values.tolist() == [["er", "sum"]]


def test_sensitivity_parser_accepts_explicit_sample_sizes():
    args = cli.build_parser().parse_args(
        [
            "analyze",
            "sensitivity",
            "--scores",
            "scores.jsonl",
            "--output",
            "report",
            "--sample-sizes",
            "10",
            "100",
            "4237",
            "--title",
            "NB101 core proxies",
        ]
    )

    assert args.sample_sizes == [10, 100, 4237]
    assert args.title == "NB101 core proxies"


def test_nb301_sample_parser_accepts_generated_population_count():
    args = cli.build_parser().parse_args(
        [
            "benchmark",
            "sample",
            "nasbench301_surrogate",
            "--trusted",
            "--population-count",
            "11221",
            "--count",
            "1000",
            "--output",
            "sample.json",
        ]
    )

    assert args.population_count == 11221
    assert args.count == 1000


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("valid_neg_loss", "maximize"),
        ("valid-negative-loss", "maximize"),
        ("valid_loss", "minimize"),
        ("final_training_time", "minimize"),
        ("valid_top1", "maximize"),
    ],
)
def test_target_direction_inference_preserves_negative_loss_semantics(
    metric_name, expected
):
    assert cli._infer_target_direction(metric_name) == expected


def test_prepare_transnas_input_parser_defaults_output_to_data_root():
    args = cli.build_parser().parse_args(
        [
            "data",
            "prepare-transnas-input",
            "--data-root",
            "/taskonomy",
            "--split-json",
            "/taskonomy/train.json",
            "--verify-files",
        ]
    )

    assert args.action == "prepare-transnas-input"
    assert args.output is None
    assert args.verify_files is True


def test_transnas_proxy_support_reflects_real_jigsaw_input_contract():
    assert cli._transnas_unsupported_reason("transnasbench101", "jigsaw", "gradnorm") is None
    reason = cli._transnas_unsupported_reason("transnasbench101", "normal", "gradnorm")
    assert reason is not None and "label protocol is not implemented" in reason
    assert cli._transnas_unsupported_reason("transnasbench101", "normal", "params") is None


def test_benchmark_study_frame_retains_failed_primary_calls(tmp_path):
    scores = tmp_path / "scores.jsonl"
    rows = [
        {
            "architecture_id": architecture_id,
            "proxy_id": "synflow",
            "primary_component": "score",
            "components": {"score": score} if score is not None else {},
            "score": score,
            "status": status,
        }
        for architecture_id, score, status in (
            ("a", 1.0, "ok"),
            ("b", None, "failed"),
        )
    ]
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    arguments = argparse.Namespace(
        scores=[str(scores)],
        component=None,
        dataset=None,
        target_metric=None,
        target_split=None,
        target_epoch_budget=None,
        benchmark_variant=None,
    )

    frame = cli._benchmark_study_frame(arguments)

    assert frame[["architecture_id", "status"]].values.tolist() == [
        ["a", "ok"],
        ["b", "failed"],
    ]


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
    assert proxy["primary_component"] == "score"
    assert proxy["requires_edge_activations"] is True
    assert proxy["requires_topology"] is False
    cli.main(["proxy", "inspect", "params"])
    params = _output(capsys)
    assert params["version"] == "count-v2"
    assert params["direction"] == "maximize"
    assert params["resource_direction"] == "minimize"
    cli.main(["proxy", "matrix"])
    matrix = _output(capsys)
    capability_fields = {
        "proxy_id",
        "version",
        "model_families",
        "requires_data",
        "requires_inputs",
        "requires_labels",
        "requires_loss_fn",
        "supports_cpu",
        "direction",
        "components",
        "primary_component",
        "dependencies",
        "implementation_fidelity",
        "source",
        "alias_of",
        "resource_direction",
        "source_commit",
        "license",
        "official_code_available",
        "protocol_domain",
        "default_batches",
        "default_repetitions",
        "requires_edge_activations",
        "requires_topology",
        "formal_use",
    }
    assert all(set(row) == capability_fields for row in matrix)
    assert [row["proxy_id"] for row in matrix] == sorted(
        row["proxy_id"] for row in matrix
    )
    assert any(row["proxy_id"] == "er" for row in matrix)
    assert any(
        row["proxy_id"] == "flops"
        and row["direction"] == "maximize"
        and row["resource_direction"] == "minimize"
        for row in matrix
    )
    cli.main(["proxy", "validate", "params"])
    validation = _output(capsys)
    assert validation["status"] == "ok"
    assert validation["model_state_unchanged"] is True
    assert validation["model_modes_unchanged"] is True
    assert validation["gradient_flags_unchanged"] is True
    assert validation["hooks_clean"] is True
    assert validation["python_rng_unchanged"] is True
    assert validation["numpy_rng_unchanged"] is True
    assert validation["torch_rng_unchanged"] is True
    assert validation["primary_component_matches_capability"] is True

    cli.main(["proxy", "validate", "meco_opt"])
    meco_opt_validation = _output(capsys)
    assert meco_opt_validation["status"] == "ok"
    assert meco_opt_validation["hooks_clean"] is True
    assert meco_opt_validation["python_rng_unchanged"] is True


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

    report_source = tmp_path / "scores.jsonl"
    report_source.write_text(
        '{"architecture_id":"a","score":1.5}\n', encoding="utf-8"
    )
    report_output = tmp_path / "reports" / "scores.csv"
    cli.main(
        [
            "report",
            "--source",
            str(report_source),
            "--output",
            str(report_output),
        ]
    )
    assert _output(capsys) == {"rows": 1, "output": str(report_output)}
    assert report_output.read_text(encoding="utf-8").splitlines() == [
        "architecture_id,score",
        "a,1.5",
    ]


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


def test_cli_benchmark_catalog_path_requires_integrity_and_protocol(tmp_path):
    runtime = tmp_path / "vit.jsonl"
    runtime.write_text("original\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    cli.DataRegistry(catalog).register(
        cli.DataAsset(
            "vitbench101_0",
            str(runtime),
            "auto-prox-90ed458",
            sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
            protocol="auto-prox-90ed458-autoformer-main",
        )
    )
    args = argparse.Namespace(
        benchmark_path=None,
        benchmark="vitbench101",
        transnas_space="micro",
        slice_id="autoformer_main",
        catalog=str(catalog),
        data_root=str(tmp_path),
    )
    assert cli._resolve_benchmark_path(args) == str(runtime)

    runtime.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed catalog verification"):
        cli._resolve_benchmark_path(args)


def test_cli_convert_imagenet16_registers_safe_manifest(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "safe/manifest.json"

    def fake_convert(source, destination, *, trusted, replace):
        assert source == "raw"
        assert destination == str(tmp_path / "safe")
        assert trusted is True
        assert replace is False
        manifest.parent.mkdir()
        manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
        return manifest

    monkeypatch.setattr(cli, "convert_imagenet16_120", fake_convert)
    monkeypatch.setattr(
        "zcp_test.data.imagenet16.verify_safe_imagenet16",
        lambda _path: {"valid": True},
    )
    catalog = tmp_path / "catalog.json"
    cli.main(
        [
            "data",
            "convert-imagenet16",
            "--source",
            "raw",
            "--output",
            str(tmp_path / "safe"),
            "--trusted",
            "--register",
            "--catalog",
            str(catalog),
        ]
    )

    assert _output(capsys)["asset_id"] == "dataset_imagenet16_120"
    cli.DataRegistry(catalog).get_verified(
        "dataset_imagenet16_120",
        expected_version="npy-shards-v1",
        expected_protocol="imagenet16-120-official-md5-safe-conversion-v1",
    )
    args = argparse.Namespace(data_root=None, catalog=str(catalog))
    assert cli._resolve_data_root(args, "ImageNet16-120") == str(manifest.parent)


def test_transnas_tasks_share_one_registered_taskonomy_root(tmp_path):
    root = tmp_path / "taskonomy"
    root.mkdir()
    catalog = tmp_path / "catalog.json"
    cli.DataRegistry(catalog).register(
        cli.DataAsset(
            "dataset_transnas_taskonomy",
            str(root),
            "transnas-final5k",
            protocol="licensed-external-taskonomy-manifest-v1",
            trusted=True,
        )
    )
    args = argparse.Namespace(data_root=None, catalog=str(catalog))

    assert cli._resolve_data_root(args, "class_object") == str(root)
    assert cli._resolve_data_root(args, "segmentsemantic") == str(root)


def test_inline_architecture_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="must be an object"):
        cli._load_architecture_spec("[]")
    with pytest.raises(ValueError, match="'spec' must be an object"):
        cli._load_architecture_spec('{"spec": 1}')
    missing = tmp_path / "looks-like-file.json"
    with pytest.raises(ValueError, match="existing JSON file"):
        cli._load_architecture_spec(str(missing))


def test_cli_search_resume_end_to_end_with_tiny_reference_space(monkeypatch, capsys, tmp_path):
    space = _TinyReferenceSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    output = tmp_path / "first-search"
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
    state = run / "search-state.json"

    resumed_output = tmp_path / "resumed-search"
    cli.main([
        "search",
        "--space", "tiny_reference",
        "--proxy", "params",
        "--population", "2",
        "--generations", "3",
        "--elite-ratio", "0.5",
        "--resume", str(state),
        "--device", "cpu",
        "--input-source", "random",
        "--batch-size", "2",
        "--input-size", "8",
        "--classes", "3",
        "--output", str(resumed_output),
    ])
    resumed_result = _output(capsys)
    resumed_run = Path(resumed_result["run"])
    resumed_rows = list(read_jsonl(resumed_run / "search.jsonl"))
    summaries = [
        row["generation"]
        for row in resumed_rows
        if row["record_kind"] == "generation_summary"
    ]
    assert summaries == [0, 1, 2, 3]
    original_rows = list(read_jsonl(run / "search.jsonl"))
    assert resumed_rows[: len(original_rows)] == original_rows


def test_cli_evaluate_model_initialization_is_architecture_seeded(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())

    def evaluate(output):
        cli.main([
            "evaluate",
            "--space", "tiny_reference",
            "--proxies", "naswot",
            "--count", "2",
            "--seed", "20260731",
            "--device", "cpu",
            "--input-source", "random",
            "--batch-size", "2",
            "--input-size", "8",
            "--classes", "3",
            "--dataset", "cifar10",
            "--output", str(output),
        ])
        run = Path(_output(capsys)["run"])
        return list(read_jsonl(run / "scores.jsonl"))

    first = evaluate(tmp_path / "evaluate-a")
    second = evaluate(tmp_path / "evaluate-b")

    assert len(first) == len(second) == 2
    for left, right in zip(first, second, strict=True):
        assert left["architecture_id"] == right["architecture_id"]
        assert left["model_initialization_protocol"] == "architecture-hash-v1"
        assert left["model_initialization_seed"] == right["model_initialization_seed"]
        assert left["score"] == pytest.approx(right["score"], rel=0, abs=0)
    assert first[0]["model_initialization_seed"] != first[1]["model_initialization_seed"]


def test_cli_search_resume_rejects_identity_mismatch(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())
    output = tmp_path / "first-search"
    arguments = [
        "search",
        "--space", "tiny_reference",
        "--proxy", "params",
        "--population", "2",
        "--generations", "0",
        "--elite-ratio", "0.5",
        "--device", "cpu",
        "--input-source", "random",
        "--batch-size", "2",
        "--input-size", "8",
        "--classes", "3",
        "--output", str(output),
    ]
    cli.main(arguments)
    state = Path(_output(capsys)["search_state"])
    mismatched_output = tmp_path / "mismatched-search"

    with pytest.raises(ValueError, match="identity does not match"):
        cli.main([
            *arguments[:-2],
            "--seed", "43",
            "--resume", str(state),
            "--output", str(mismatched_output),
        ])

    assert not mismatched_output.exists()


def test_cli_search_rejects_aznas_on_unsupported_space(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())

    with pytest.raises(ValueError, match="requires --score-selector aggregate:az_nas_log_rank"):
        cli.main([
            "search",
            "--space", "tiny_reference",
            "--proxy", "az_nas",
            "--population", "2",
            "--generations", "0",
            "--device", "cpu",
            "--input-source", "random",
            "--batch-size", "2",
            "--input-size", "8",
            "--classes", "3",
            "--output", str(tmp_path / "rejected"),
        ])

    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    ("proxy", "aggregator", "message"),
    [
        ("az_nas_autoformer", "primary", "requires --score-selector aggregate:az_nas_log_rank"),
        ("az_nas_plainnet", "primary", "requires --score-selector aggregate:az_nas_log_rank"),
        ("params", "az_nas_log_rank", "requires a component-valued proxy"),
    ],
)
def test_cli_search_rejects_invalid_proxy_aggregator_pairs_before_run_creation(
    monkeypatch, tmp_path, proxy, aggregator, message
):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())
    output = tmp_path / "rejected-aggregator"

    with pytest.raises(ValueError, match=message):
        cli.main([
            "search",
            "--space", "tiny_reference",
            "--proxy", proxy,
            "--aggregator", aggregator,
            "--population", "2",
            "--generations", "0",
            "--device", "cpu",
            "--input-source", "random",
            "--batch-size", "2",
            "--input-size", "8",
            "--classes", "3",
            "--output", str(output),
        ])

    assert not output.exists()


@pytest.mark.parametrize(
    ("space_name", "proxy", "expected_space"),
    [
        ("tiny_reference", "az_nas_autoformer", "autoformer"),
        ("tiny_reference", "az_nas_plainnet", "zennas_plainnet_mbv2"),
    ],
)
def test_cli_search_rejects_aznas_proxy_space_mismatch_before_run_creation(
    monkeypatch, tmp_path, space_name, proxy, expected_space
):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())
    output = tmp_path / "rejected-space-pair"

    with pytest.raises(ValueError, match=expected_space):
        cli.main([
            "search",
            "--space", space_name,
            "--proxy", proxy,
            "--aggregator", "az_nas_log_rank",
            "--population", "2",
            "--generations", "0",
            "--device", "cpu",
            "--input-source", "random",
            "--batch-size", "2",
            "--input-size", "8",
            "--classes", "3",
            "--output", str(output),
        ])

    assert not output.exists()


def test_config_does_not_override_equals_form_cli_option(monkeypatch, tmp_path):
    config = tmp_path / "evaluate.yaml"
    config.write_text("evaluate:\n  count: 17\n", encoding="utf-8")
    captured = {}

    def command(arguments):
        captured["count"] = arguments.count

    monkeypatch.setattr(cli, "command_evaluate", command)
    cli.main(["evaluate", "--config", str(config), "--count=3"])

    assert captured["count"] == 3


def test_non_training_config_rejects_unknown_keys(tmp_path):
    config = tmp_path / "evaluate.yaml"
    config.write_text("evaluate:\n  coutn: 17\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys: coutn"):
        cli.main(["evaluate", "--config", str(config)])


def test_training_config_rejects_unknown_profile_keys(tmp_path):
    config = tmp_path / "training.yaml"
    config.write_text(
        "space: darts\n"
        "dataset: cifar10\n"
        "learnng_rate: 0.025\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys: learnng_rate"):
        cli.main(["train", "--config", str(config), "--smoke"])


@pytest.mark.parametrize("config_path", sorted(Path("configs/training").glob("*.yaml")))
def test_shipped_training_profiles_pass_config_key_schema(monkeypatch, config_path):
    captured = {}

    def command(arguments):
        captured["config"] = arguments.config

    monkeypatch.setattr(cli, "command_train", command)
    cli.main(["train", "--config", str(config_path), "--smoke"])

    assert captured["config"] == str(config_path)


def test_training_config_allows_runtime_cli_keys(monkeypatch, tmp_path):
    config = tmp_path / "training.yaml"
    config.write_text(
        "space: darts\n"
        "dataset: cifar10\n"
        f"data_root: {tmp_path.as_posix()}\n"
        "workers: 3\n",
        encoding="utf-8",
    )
    captured = {}

    def command(arguments):
        captured["data_root"] = arguments.data_root
        captured["workers"] = arguments.workers

    monkeypatch.setattr(cli, "command_train", command)
    cli.main(["train", "--config", str(config), "--smoke"])

    assert captured == {"data_root": str(tmp_path), "workers": 3}


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


def test_cli_full_batch_memory_smoke_preserves_configured_micro_batch(
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
        "batch_size: 6\n"
        "formal_training_ready: true\n"
        "protocol: test-full-batch-memory-smoke\n",
        encoding="utf-8",
    )
    output = tmp_path / "training"

    cli.main(
        [
            "train",
            "--config",
            str(config),
            "--smoke",
            "--full-batch-smoke",
            "--device",
            "cpu",
            "--classes",
            "3",
            "--output",
            str(output),
        ]
    )

    run = Path(_output(capsys)["run"])
    resolved = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert resolved["training_mode"] == "synthetic_full_batch_memory_smoke"
    assert resolved["per_device_batch_size"] == 6
    assert resolved["full_batch_smoke"] is True


def test_full_batch_memory_smoke_requires_smoke_mode(tmp_path):
    config = tmp_path / "training.yaml"
    config.write_text("space: darts\ndataset: cifar10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --smoke"):
        cli.main(
            [
                "train",
                "--config",
                str(config),
                "--full-batch-smoke",
                "--device",
                "cpu",
            ]
        )


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

    train, valid = cli._real_loaders(
        "cifar10",
        str(tmp_path),
        batch_size=2,
        input_size=8,
        workers=2,
        config={"valid_workers": 1},
        fraction=0.5,
        seed=3,
    )
    assert train.num_workers == 2
    assert valid.num_workers == 1

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


def test_training_seed_covers_python_numpy_and_torch():
    import random

    import numpy as np

    previous = torch.are_deterministic_algorithms_enabled()
    previous_cudnn = torch.backends.cudnn.deterministic
    try:
        first_state = cli._seed_training(2026, 3, True)
        first = (random.random(), float(np.random.random()), float(torch.rand(1)))
        second_state = cli._seed_training(2026, 3, True)
        second = (random.random(), float(np.random.random()), float(torch.rand(1)))

        assert first == second
        assert first_state == second_state
        assert first_state["base_seed"] == 2026
        assert first_state["rank_seed"] == 2029
        assert first_state["deterministic_algorithms"] is True
        assert first_state["cudnn_deterministic"] is True
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.deterministic = previous_cudnn


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


@pytest.mark.parametrize("action", ["export-manifest", "import-manifest"])
def test_data_transfer_manifest_rejects_unused_catalog_argument(action, tmp_path):
    arguments = ["data", action, "--root", str(tmp_path), "--catalog", "unused.json"]
    if action == "export-manifest":
        arguments.extend(
            ["--benchmarks", "nasbench101", "--output", str(tmp_path / "manifest.json")]
        )
    else:
        arguments.extend(["--manifest", str(tmp_path / "manifest.json")])

    with pytest.raises(SystemExit):
        cli.main(arguments)


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


def test_cli_search_applies_resource_constraints_before_proxy(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())
    output = tmp_path / "resource-search"

    cli.main(
        [
            "search",
            "--space",
            "tiny_reference",
            "--proxy",
            "params",
            "--population",
            "2",
            "--generations",
            "0",
            "--device",
            "cpu",
            "--input-source",
            "random",
            "--batch-size",
            "2",
            "--input-size",
            "8",
            "--classes",
            "3",
            "--max-parameters",
            "1000",
            "--max-macs",
            "1000",
            "--output",
            str(output),
        ]
    )

    run = Path(_output(capsys)["run"])
    rows = [row for row in read_jsonl(run / "search.jsonl") if row["record_kind"] == "candidate"]
    assert len(rows) == 2
    assert all(row["parameters"] == 579 for row in rows)
    assert all(row["compute_value"] == 576 for row in rows)
    assert all(row["compute_metric"] == "thop_macs" for row in rows)
    assert all(row["resource_constraints"] == {"max_macs": 1000, "max_parameters": 1000} for row in rows)
    assert all(row["cumulative_constraint_rejections"] == 0 for row in rows)


def test_cli_search_resource_constraints_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyReferenceSpace())

    with pytest.raises(RuntimeError, match="maximum number of consecutive attempts"):
        cli.main(
            [
                "search",
                "--space",
                "tiny_reference",
                "--proxy",
                "params",
                "--population",
                "2",
                "--generations",
                "0",
                "--device",
                "cpu",
                "--input-source",
                "random",
                "--batch-size",
                "2",
                "--input-size",
                "8",
                "--classes",
                "3",
                "--max-parameters",
                "1",
                "--constraint-max-attempts",
                "3",
                "--output",
                str(tmp_path / "rejected-resource-search"),
            ]
        )
