import json

import pytest
import torch

from zcp_test.artifacts import normalize_score_records
from zcp_test.cli import main
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
    er_rows = [row for row in rows if row["proxy_id"] == "er"]
    assert all(row["primary_component"] == "mean" for row in er_rows)
    assert all(set(row["components"]) == {"mean", "sum"} for row in er_rows)
    assert not (run / "checkpoints").exists()
    assert not (run / "parts").exists()
    assert not (run / "reports").exists()


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
