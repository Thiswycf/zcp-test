import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import zcp_test.cli as cli
from zcp_test.artifacts import normalize_score_records
from zcp_test.cli import _load_architecture_spec, _stratified_subset, main
from zcp_test.inputs import make_input_batch
from zcp_test.training.protocols import (
    resolve_acceptance_protocol,
    validate_candidate_training_protocol,
)


def test_evaluate_one_row_per_proxy_and_lazy_directories(tmp_path):
    output = tmp_path / "runs"
    main(
        [
            "evaluate",
            "--space",
            "nb201_topology",
            "--proxies",
            "er,naswot,synflow",
            "--count",
            "2",
            "--device",
            "cpu",
            "--input-source",
            "random",
            "--batch-size",
            "2",
            "--input-size",
            "8",
            "--output",
            str(output),
        ]
    )
    run = next(output.iterdir())
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(row["schema_version"] == "2.1" for row in rows)
    assert all("target_epoch_budget" in row for row in rows)
    er_rows = [row for row in rows if row["proxy_id"] == "er"]
    assert all(row["primary_component"] == "mean" for row in er_rows)
    assert all(set(row["components"]) == {"mean", "sum"} for row in er_rows)
    assert not (run / "checkpoints").exists()
    assert not (run / "parts").exists()
    assert not (run / "reports").exists()


def test_evaluate_identity_can_come_from_yaml_config(tmp_path):
    output = tmp_path / "configured-runs"
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        "evaluate:\n"
        "  space: nb201_topology\n"
        "  proxies: params\n"
        "  count: 1\n"
        "  device: cpu\n"
        "  input_source: random\n"
        "  batch_size: 1\n"
        "  input_size: 8\n"
        f"  output: {output}\n",
        encoding="utf-8",
    )

    main(["evaluate", "--config", str(config)])

    run = next(output.iterdir())
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["proxy_id"] == "params"
    assert rows[0]["proxy_version"] == "count-v2"
    assert rows[0]["direction"] == "maximize"
    assert rows[0]["resource_direction"] == "minimize"


def test_analyze_correlation_cli_retains_failed_rows_in_coverage(tmp_path):
    scores = tmp_path / "scores.jsonl"
    rows = [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "synflow",
            "primary_component": "score",
            "components": {"score": float(index)},
            "score": float(index),
            "target_value": float(index),
            "seed": 1,
            "status": "ok",
        }
        for index in (1, 2, 3)
    ]
    rows.append(
        {
            "architecture_id": "a4",
            "proxy_id": "synflow",
            "primary_component": "score",
            "components": {},
            "score": None,
            "target_value": 4.0,
            "seed": 1,
            "status": "failed",
        }
    )
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "analysis"

    main(
        [
            "analyze",
            "correlation",
            "--scores",
            str(scores),
            "--bootstrap-samples",
            "0",
            "--output",
            str(output),
        ]
    )

    import pandas as pd

    record = pd.read_csv(output / "correlations.csv").iloc[0]
    assert record["total_count"] == 4
    assert record["failed_count"] == 1
    assert record["sample_count"] == 3
    assert record["coverage"] == pytest.approx(0.75)


def test_legacy_score_rows_fold_without_modifying_source():
    rows = [
        {"run_id": "r", "architecture_id": "a", "proxy_id": "er", "component": "mean", "score": 2.0},
        {"run_id": "r", "architecture_id": "a", "proxy_id": "er", "component": "sum", "score": 8.0},
    ]
    normalized = list(normalize_score_records(rows))
    assert len(normalized) == 1
    assert normalized[0]["score"] == 2.0
    assert normalized[0]["components"] == {"mean": 2.0, "sum": 8.0}


def test_dataset_input_does_not_fall_back_to_random(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_input_batch(
            "dataset", "cifar10", 2, 32, 10, 1, torch.device("cpu"), str(tmp_path / "missing")
        )


def test_explicit_missing_benchmark_path_does_not_fall_back_to_catalog(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explicit benchmark path"):
        main(
            [
                "evaluate",
                "--benchmark",
                "nasbench101",
                "--benchmark-path",
                str(tmp_path / "missing.jsonl"),
                "--device",
                "cpu",
                "--input-source",
                "random",
            ]
        )


@pytest.mark.parametrize(
    ("dataset_name", "split", "sample_count", "class_count", "expected_size", "counts"),
    [
        ("cifar10", "train", 50_000, 10, 500, {50}),
        ("cifar10", "valid", 10_000, 10, 100, {10}),
        ("cifar100", "train", 50_000, 100, 500, {5}),
        ("cifar100", "valid", 10_000, 100, 100, {1}),
        ("imagenet1k", "train", 1_281_167, 1000, 12_812, {12, 13}),
        ("imagenet1k", "valid", 50_000, 1000, 500, {1}),
    ],
)
def test_one_percent_subset_matches_canonical_split_sizes_and_distribution(
    dataset_name,
    split,
    sample_count,
    class_count,
    expected_size,
    counts,
):
    class Dataset(torch.utils.data.Dataset):
        targets = [index % class_count for index in range(sample_count)]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    dataset = Dataset()
    first = _stratified_subset(dataset, 0.01, 2026)
    second = _stratified_subset(dataset, 0.01, 2026)
    selected_counts = Counter(dataset.targets[index] for index in first.indices)

    assert len(first) == expected_size, (dataset_name, split)
    assert first.indices == second.indices
    assert set(selected_counts.values()) == counts
    if expected_size < class_count:
        assert len(selected_counts) == expected_size


def test_stratified_subset_fails_closed_when_global_target_is_empty():
    class Dataset(torch.utils.data.Dataset):
        targets = [0] * 49

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    with pytest.raises(ValueError, match="empty subset"):
        _stratified_subset(Dataset(), 0.01, 7)


def test_config_cannot_enable_trusted_execution(tmp_path):
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        "evaluate:\n  benchmark: nasbench201\n  trusted: true\n",
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="explicitly"):
        main(["evaluate", "--config", str(config)])


def test_config_cannot_self_approve_formal_training_protocol(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "space: autoformer\n"
        "dataset: imagenet1k\n"
        "epochs: 1\n"
        "optimizer: adamw\n"
        "learning_rate: 0.001\n"
        "weight_decay: 0.01\n"
        "formal_training_ready: true\n",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="not approved"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_approved_formal_protocol_rejects_modified_recipe(tmp_path):
    config = tmp_path / "train.yaml"
    source = Path("configs/training/darts_cifar10.yaml").read_text(encoding="utf-8")
    config.write_text(source.replace("learning_rate: 0.025", "learning_rate: 0.1"), encoding="utf-8")

    with pytest.raises(ValueError, match="learning_rate"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_approved_formal_protocol_requires_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    config = Path("configs/training/darts_cifar10.yaml")
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                str(config),
                "--device",
                "cpu",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_autoformer_candidate_training_protocol_is_locked():
    config = yaml.safe_load(
        Path("configs/training/autoformer_imagenet.yaml").read_text(encoding="utf-8")
    )
    assert validate_candidate_training_protocol(config) == "aznas-autoformer-scratch"
    assert config["epochs"] == 500
    assert config["warmup_epochs"] == 20
    assert config["warmup_learning_rate"] == pytest.approx(1e-6)
    assert config["minimum_learning_rate"] == pytest.approx(1e-5)
    assert config["validation_label_smoothing"] == 0.0
    assert config["exclude_bias_norm_from_weight_decay"] is True
    config["mixup"] = 0.0
    with pytest.raises(ValueError, match="mixup"):
        validate_candidate_training_protocol(config)


def test_proxyless_candidate_training_protocol_is_locked():
    config = yaml.safe_load(
        Path("configs/training/ofa_proxyless_mbv2_imagenet.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert validate_candidate_training_protocol(config) == (
        "proxylessnas-mbv2-scratch-b23018c9"
    )
    assert config["scheduler"] == "cosine_step"
    assert config["color_distortion"] == "tf"
    assert config["exclude_norm_from_weight_decay"] is True
    assert config["formal_training_ready"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "4", "--data-fraction", "1.0"],
        ["--epochs", "500", "--data-fraction", "0.02"],
    ],
)
def test_autoformer_acceptance_training_rejects_non_one_percent_protocol(arguments):
    with pytest.raises(ValueError, match="1%"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                *arguments,
            ]
        )


def test_autoformer_acceptance_training_rejects_less_than_one_percent_data():
    with pytest.raises(ValueError, match="exactly 1% data"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "500",
                "--data-fraction",
                "0.005",
            ]
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [("--batch-size", "128", "batch_size"), ("--input-size", "192", "input_size")],
)
def test_autoformer_acceptance_training_rejects_shape_overrides(option, value, message):
    with pytest.raises(ValueError, match=message):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "1",
                option,
                value,
            ]
        )


def test_autoformer_acceptance_training_requires_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "5",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_darts_formal_protocol_is_allowed_for_one_percent_acceptance(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--acceptance-smoke",
                "--device",
                "cpu",
                "--epochs",
                "6",
                "--data-fraction",
                "1.0",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("config_path", "minimum_epochs"),
    [
        ("configs/training/darts_cifar10.yaml", 6),
        ("configs/training/darts_cifar100.yaml", 6),
        ("configs/training/darts_imagenet.yaml", 3),
    ],
)
def test_darts_acceptance_minimum_full_data_epochs(config_path, minimum_epochs):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert (
        resolve_acceptance_protocol(config, minimum_epochs, 1.0)
        == "full_data_one_percent_epochs"
    )
    with pytest.raises(ValueError, match="at least 1% epochs"):
        resolve_acceptance_protocol(config, minimum_epochs - 1, 1.0)


@pytest.mark.parametrize("data_fraction", [0.0100001, 0.010001, 0.0099999])
def test_one_percent_acceptance_rejects_nearby_fractions(data_fraction):
    config = yaml.safe_load(
        Path("configs/training/darts_cifar10.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="exactly 1% data"):
        resolve_acceptance_protocol(config, 600, data_fraction)


@pytest.mark.parametrize(
    ("mode", "epochs", "data_fraction", "expected_protocol", "expected_training_mode"),
    [
        (
            "--acceptance-smoke",
            6,
            1.0,
            "full_data_one_percent_epochs",
            "acceptance_smoke",
        ),
        (
            "--acceptance-smoke",
            600,
            0.01,
            "one_percent_data_protocol",
            "acceptance_smoke",
        ),
        ("--real-data-preflight", 1, 1.0, None, "real_data_preflight"),
    ],
)
def test_darts_real_data_modes_resolve_config_identity_and_ddp_batch(
    monkeypatch,
    tmp_path,
    mode,
    epochs,
    data_fraction,
    expected_protocol,
    expected_training_mode,
):
    captured = {}

    @contextmanager
    def training_device(*args, **kwargs):
        yield torch.device("cpu"), {"selection_strategy": "test"}

    @contextmanager
    def training_run_context(output, argv, resolved, runtime, world_size, rank):
        captured["resolved"] = resolved
        yield SimpleNamespace(directory=tmp_path)

    def fake_train_model(*args, **kwargs):
        captured["training_config"] = args[3]
        captured["run_identity"] = kwargs["run_identity"]
        return {"best_accuracy": 0.0}

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(cli, "_prepare_gpu", lambda args: None)
    monkeypatch.setattr(cli, "_training_device", training_device)
    monkeypatch.setattr(cli, "_training_run_context", training_run_context)
    monkeypatch.setattr(cli, "_build_training_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        lambda model, **kwargs: model,
    )
    monkeypatch.setattr(cli, "_resolve_data_root", lambda args, dataset: tmp_path)
    monkeypatch.setattr(cli, "_real_loaders", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(cli, "train_model", fake_train_model)

    main(
        [
            "train",
            "--config",
            "configs/training/darts_cifar10.yaml",
            mode,
            "--epochs",
            str(epochs),
            "--data-fraction",
            str(data_fraction),
        ]
    )

    assert captured["resolved"]["acceptance_protocol"] == expected_protocol
    assert captured["resolved"]["training_mode"] == expected_training_mode
    assert captured["resolved"]["configured_batch_size"] == 96
    assert captured["resolved"]["per_device_batch_size"] == 48
    assert captured["resolved"]["effective_global_batch_size"] == 96
    assert captured["resolved"]["schedule_epochs"] == 600
    assert captured["training_config"].schedule_epochs == 600
    assert captured["resolved"]["seed"] == 42
    assert captured["resolved"]["rank_seed"] == 42
    assert captured["resolved"]["deterministic"] is True
    assert captured["run_identity"]["acceptance_protocol"] == expected_protocol
    assert captured["run_identity"]["training_mode"] == expected_training_mode
    assert captured["run_identity"]["seed"] == 42
    assert captured["run_identity"]["data_fraction"] == data_fraction


def test_training_smoke_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        main(
            [
                "train",
                "--config",
                "configs/training/autoformer_imagenet.yaml",
                "--smoke",
                "--acceptance-smoke",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "2", "--data-fraction", "1.0"],
        ["--epochs", "1", "--data-fraction", "0.01"],
    ],
)
def test_real_data_preflight_requires_one_complete_data_epoch(arguments):
    with pytest.raises(ValueError, match="exactly one epoch over the complete dataset"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--real-data-preflight",
                "--device",
                "cpu",
                *arguments,
            ]
        )


def test_real_data_preflight_requires_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr("zcp_test.cli._resolve_data_root", lambda args, dataset: None)
    with pytest.raises(ValueError, match="data-root"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--real-data-preflight",
                "--epochs",
                "1",
                "--data-fraction",
                "1.0",
                "--device",
                "cpu",
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--batch-size", "8", "batch_size"),
        ("--input-size", "64", "input_size"),
    ],
)
def test_approved_formal_protocol_rejects_shape_overrides(option, value, message):
    config = Path("configs/training/darts_cifar10.yaml")
    with pytest.raises(ValueError, match=message):
        main(["train", "--config", str(config), option, value, "--device", "cpu"])


def test_formal_training_rejects_non_reference_space_before_protocol(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "space: darts_toy_legacy\n"
        "dataset: cifar10\n"
        "epochs: 1\n"
        "optimizer: sgd\n"
        "learning_rate: 0.1\n"
        "weight_decay: 0.0\n"
        "formal_training_ready: true\n",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="requires reference_model"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_incomplete_formal_training_protocol_is_rejected(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "space: autoformer\n"
        "dataset: imagenet1k\n"
        "epochs: 1\n"
        "optimizer: adamw\n"
        "learning_rate: 0.001\n"
        "weight_decay: 0.01\n"
        "formal_training_ready: false\n"
        "formal_training_blockers: [repeated augmentation]\n",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="repeated augmentation"):
        main(["train", "--config", str(config), "--device", "cpu"])


def test_train_rejects_manual_device_override_under_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="cuda:LOCAL_RANK"):
        main(
            [
                "train",
                "--config",
                "configs/training/darts_cifar10.yaml",
                "--smoke",
                "--device",
                "cpu",
            ]
        )


def test_train_architecture_accepts_inline_json_and_json_file(tmp_path):
    specification = {"normal": [], "reduce": []}
    assert _load_architecture_spec(json.dumps({"spec": specification})) == specification
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(specification), encoding="utf-8")
    assert _load_architecture_spec(str(path)) == specification
    with pytest.raises(ValueError, match="existing JSON file"):
        _load_architecture_spec("not-json")


def test_evaluate_rejects_approximation_without_explicit_opt_in(tmp_path):
    with pytest.raises(RuntimeError, match="allow-approximation"):
        main(
            [
                "evaluate",
                "--space",
                "darts_toy_legacy",
                "--proxies",
                "params",
                "--count",
                "1",
                "--device",
                "cpu",
                "--input-source",
                "random",
                "--output",
                str(tmp_path),
            ]
        )


def test_correlate_uses_selected_score_field_and_inner_join(tmp_path):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    output = tmp_path / "correlations.jsonl"
    score_rows = [
        {
            "architecture_id": architecture_id,
            "proxy_id": "p",
            "score": wrong,
            "alternate": correct,
            "status": status,
        }
        for architecture_id, wrong, correct, status in (
            ("a", 3.0, 1.0, "ok"),
            ("b", 2.0, 2.0, "ok"),
            ("c", 1.0, 3.0, "ok"),
            ("failed", 4.0, 4.0, "failed"),
        )
    ]
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    targets.write_text(
        "".join(
            json.dumps({"architecture_id": key, "accuracy": value}) + "\n"
            for key, value in (("a", 1.0), ("b", 2.0), ("c", 3.0), ("target-only", 9.0))
        )
    )
    main(
        [
            "correlate",
            "--scores",
            str(scores),
            "--targets",
            str(targets),
            "--output",
            str(output),
            "--score-field",
            "alternate",
            "--target-field",
            "accuracy",
        ]
    )
    row = json.loads(output.read_text())
    assert row["sample_count"] == 3
    assert row["spearman"] == pytest.approx(1.0)
    assert row["score_direction"] == "maximize"
    assert row["target_direction"] == "maximize"
    assert row["score_coverage"] == pytest.approx(1.0)
    assert row["target_coverage"] == pytest.approx(0.75)


def test_correlate_normalizes_score_and_target_directions(tmp_path):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    output = tmp_path / "correlations.jsonl"
    scores.write_text(
        "".join(
            json.dumps(
                {
                    "architecture_id": architecture_id,
                    "proxy_id": "latency",
                    "score": score,
                    "direction": "minimize",
                }
            )
            + "\n"
            for architecture_id, score in (("a", 3.0), ("b", 2.0), ("c", 1.0))
        ),
        encoding="utf-8",
    )
    targets.write_text(
        "".join(
            json.dumps({"architecture_id": architecture_id, "error": target}) + "\n"
            for architecture_id, target in (("a", 3.0), ("b", 2.0), ("c", 1.0))
        ),
        encoding="utf-8",
    )

    main(
        [
            "correlate",
            "--scores",
            str(scores),
            "--targets",
            str(targets),
            "--output",
            str(output),
            "--target-field",
            "error",
            "--target-direction",
            "minimize",
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["spearman"] == pytest.approx(1.0)
    assert row["score_direction"] == "minimize"
    assert row["target_direction"] == "minimize"
    assert row["direction_normalized_to_maximize"] is True


@pytest.mark.parametrize("duplicate_source", ["scores", "targets"])
def test_correlate_rejects_duplicate_architecture_keys(tmp_path, duplicate_source):
    scores = tmp_path / "scores.jsonl"
    targets = tmp_path / "targets.jsonl"
    score_rows = [
        {
            "schema_version": "2.1",
            "architecture_id": "a",
            "proxy_id": "p",
            "score": 1.0,
            "components": {"score": 1.0},
        },
        {
            "schema_version": "2.1",
            "architecture_id": "b",
            "proxy_id": "p",
            "score": 2.0,
            "components": {"score": 2.0},
        },
    ]
    target_rows = [
        {"architecture_id": "a", "accuracy": 1.0},
        {"architecture_id": "b", "accuracy": 2.0},
    ]
    if duplicate_source == "scores":
        score_rows.append(dict(score_rows[0]))
    else:
        target_rows.append(dict(target_rows[0]))
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    targets.write_text("".join(json.dumps(row) + "\n" for row in target_rows))
    with pytest.raises(ValueError, match="Duplicate"):
        main(
            [
                "correlate",
                "--scores",
                str(scores),
                "--targets",
                str(targets),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--target-field",
                "accuracy",
            ]
        )
