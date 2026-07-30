import json
from pathlib import Path

import pytest
import torch

from zcp_test.artifacts import normalize_score_records
from zcp_test.cli import _load_architecture_spec, _stratified_subset, main
from zcp_test.inputs import make_input_batch


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


def test_one_percent_subset_is_deterministic_and_stratified():
    class Dataset(torch.utils.data.Dataset):
        targets = [class_index for class_index in range(4) for _ in range(100)]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    first = _stratified_subset(Dataset(), 0.01, 7)
    second = _stratified_subset(Dataset(), 0.01, 7)
    assert first.indices == second.indices
    assert len(first) == 4
    assert {Dataset.targets[index] for index in first.indices} == {0, 1, 2, 3}


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
