import json

import pytest

from zcp_test.benchmarks.vitbench101 import VIT_SOURCE_SHA256, VitBench101Adapter
from zcp_test.types import MetricSpec


def _record(slice_id, specification):
    spaces = {
        "autoformer_main": ("autoformer", "auto-prox-90ed458-autoformer-main"),
        "autoformer_ext": ("autoformer", "auto-prox-90ed458-autoformer-ext"),
        "pit": ("pit", "auto-prox-90ed458-pit"),
    }
    search_space_id, protocol = spaces[slice_id]
    return {
        "record_kind": "benchmark_architecture",
        "benchmark_id": "vitbench101",
        "search_space_id": search_space_id,
        "benchmark_version": "auto-prox-90ed458",
        "benchmark_index": 0,
        "protocol": protocol,
        "source_sha256": VIT_SOURCE_SHA256[slice_id],
        "specification": specification,
        "metrics": [
            {
                "dataset": "cifar100",
                "split": "test",
                "metric_name": "accuracy_kd",
                "value": 77.5,
            }
        ],
    }


def _write_record(tmp_path, slice_id, specification, **updates):
    record = _record(slice_id, specification)
    record.update(updates)
    path = tmp_path / f"{slice_id}.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("slice_id", "specification", "equivalent"),
    [
        (
            "autoformer_main",
            {"depth": 12, "hidden_dim": 192, "num_heads": [3] * 12, "mlp_ratio": [3.5] * 12},
            {"depth": "12", "hidden_dim": "192", "num_heads": ("3",) * 12, "mlp_ratio": ("3.5",) * 12},
        ),
        (
            "autoformer_ext",
            {"depth": 14, "hidden_dim": 240, "num_heads": [4] * 14, "mlp_ratio": [4.0] * 14},
            {"depth": 14.0, "hidden_dim": 240.0, "num_heads": (4.0,) * 14, "mlp_ratio": (4,) * 14},
        ),
        (
            "pit",
            {"depth": [2, 8, 4], "base_dim": 16, "num_heads": [2, 4, 4], "mlp_ratio": 6},
            {"depth": ("2", 8.0, 4), "base_dim": "16", "num_heads": (2.0, "4", 4), "mlp_ratio": 6.0},
        ),
    ],
)
def test_vit_slices_normalize_architecture_and_expose_provenance(
    tmp_path, slice_id, specification, equivalent
):
    path = _write_record(tmp_path, slice_id, specification)
    adapter = VitBench101Adapter(str(path), slice_id=slice_id)

    assert adapter.architecture_id(specification) == adapter.architecture_id(equivalent)
    metadata = adapter.metadata()
    assert metadata["slice_id"] == slice_id
    assert metadata["protocol"] == _record(slice_id, specification)["protocol"]
    assert metadata["source_sha256"] == VIT_SOURCE_SHA256[slice_id]
    architecture = next(adapter.iter_architectures())
    result = adapter.query_metrics(
        architecture, MetricSpec("cifar100", "test", "accuracy_kd")
    )
    assert result["accuracy_kd"] == 77.5
    assert adapter.query_provenance() == {
        "slice_id": slice_id,
        "protocol": metadata["protocol"],
        "source_sha256": VIT_SOURCE_SHA256[slice_id],
    }


@pytest.mark.parametrize(
    "specification",
    [
        {"depth": 12, "hidden_dim": 192, "num_heads": [3] * 11, "mlp_ratio": [3.5] * 12},
        {"depth": 15, "hidden_dim": 192, "num_heads": [3] * 15, "mlp_ratio": [3.5] * 15},
        {"depth": 12, "hidden_dim": 192, "num_heads": [3] * 12, "mlp_ratio": [3.5] * 12, "score": 1},
    ],
)
def test_vit_rejects_invalid_or_non_architecture_fields(tmp_path, specification):
    path = _write_record(tmp_path, "autoformer_main", specification)
    with pytest.raises(ValueError):
        VitBench101Adapter(str(path), slice_id="autoformer_main")


@pytest.mark.parametrize("source_sha256", [None, "not-a-sha", "A" * 64, "a" * 64])
def test_vit_rejects_invalid_source_sha(tmp_path, source_sha256):
    specification = {
        "depth": [2, 8, 4],
        "base_dim": 16,
        "num_heads": [2, 4, 4],
        "mlp_ratio": 6,
    }
    path = _write_record(tmp_path, "pit", specification, source_sha256=source_sha256)
    with pytest.raises(ValueError, match="source_sha256"):
        VitBench101Adapter(str(path), slice_id="pit")


def test_vit_rejects_mixed_source_sha(tmp_path):
    specification = {"depth": 12, "hidden_dim": 192, "num_heads": [3] * 12, "mlp_ratio": [4] * 12}
    records = [_record("autoformer_main", specification) for _ in range(2)]
    records[1]["benchmark_index"] = 1
    records[1]["specification"] = {**specification, "hidden_dim": 216}
    records[1]["source_sha256"] = VIT_SOURCE_SHA256["autoformer_ext"]
    path = tmp_path / "mixed.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="one source_sha256"):
        VitBench101Adapter(str(path), slice_id="autoformer_main")


@pytest.mark.parametrize(
    "specification",
    [
        {"depth": [8, 1, 2], "base_dim": 16, "num_heads": [2, 4, 8], "mlp_ratio": 6},
        {"depth": [2, 8, 4], "base_dim": 16, "num_heads": [8, 4, 2], "mlp_ratio": 6},
    ],
)
def test_pit_rejects_values_in_the_wrong_stage(tmp_path, specification):
    path = _write_record(tmp_path, "pit", specification)
    with pytest.raises(ValueError, match="PiT stage"):
        VitBench101Adapter(str(path), slice_id="pit")
