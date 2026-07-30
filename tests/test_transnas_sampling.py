from __future__ import annotations

import hashlib
import json

import pytest

from zcp_test.artifacts import JsonlWriter
from zcp_test.benchmarks.transnasbench101 import TransNasBench101Adapter
from zcp_test.cli import build_parser
from zcp_test.research.sampling import (
    architecture_stratum,
    create_sample_manifest,
    load_sample_indices,
)


def _record(index: int, architecture: str, *, protocol: str) -> dict[str, object]:
    return {
        "record_kind": "benchmark_architecture",
        "benchmark_id": "transnasbench101",
        "benchmark_version": "v10141024",
        "benchmark_index": index,
        "search_space_id": "transnas_micro",
        "protocol": protocol,
        "source_sha256": "a" * 64,
        "specification": {"architecture": architecture},
        "metrics": [],
    }


def _write_records(path, records) -> None:
    writer = JsonlWriter(path, fsync_every=1)
    for record in records:
        writer.append(record)


def test_transnas_micro_stratum_separates_macro_digits_from_six_cell_operations():
    first = json.loads(
        architecture_stratum(
            "transnasbench101", {"architecture": "64-1111-1_12_222"}
        )
    )
    second = json.loads(
        architecture_stratum(
            "transnasbench101", {"architecture": "64-2222-1_11_111"}
        )
    )

    assert first["variant"] == second["variant"] == "micro"
    assert first["base_channel"] == second["base_channel"] == 64
    assert first["macro_code_1_count"] == 4
    assert first["cell_op_1_count"] == 2
    assert first["cell_op_2_count"] == 4
    assert second["macro_code_2_count"] == 4
    assert second["cell_op_1_count"] == 6
    assert second["cell_op_2_count"] == 0
    assert first["cell_operation_count"] == second["cell_operation_count"] == 6
    assert first != second


def test_transnas_macro_stratum_records_base_channel_and_module_code_counts():
    stratum = json.loads(
        architecture_stratum(
            "transnasbench101", {"architecture": "32-12344-basic"}
        )
    )

    assert stratum == {
        "base_channel": 32,
        "macro_code_1_count": 1,
        "macro_code_2_count": 1,
        "macro_code_3_count": 1,
        "macro_code_4_count": 2,
        "macro_module_count": 5,
        "variant": "macro",
    }


def test_transnas_adapter_requires_exact_space_protocol(tmp_path):
    path = tmp_path / "micro.jsonl"
    _write_records(
        path,
        [_record(0, "64-41414-0_00_000", protocol="transnasbench101-macro-final")],
    )

    with pytest.raises(ValueError, match="Expected protocol"):
        TransNasBench101Adapter(str(path), space="micro", version="v10141024")

    missing_path = tmp_path / "missing-protocol.jsonl"
    missing_protocol = _record(
        0, "64-41414-0_00_000", protocol="transnasbench101-micro-final"
    )
    missing_protocol.pop("protocol")
    _write_records(missing_path, [missing_protocol])

    with pytest.raises(ValueError, match="Expected protocol"):
        TransNasBench101Adapter(
            str(missing_path), space="micro", version="v10141024"
        )


def test_transnas_manifest_carries_space_variant_and_file_provenance(tmp_path):
    path = tmp_path / "micro.jsonl"
    _write_records(
        path,
        [
            _record(0, "64-41414-0_00_000", protocol="transnasbench101-micro-final"),
            _record(1, "64-41414-0_00_001", protocol="transnasbench101-micro-final"),
        ],
    )
    adapter = TransNasBench101Adapter(str(path), space="micro", version="v10141024")

    manifest = create_sample_manifest(
        adapter.benchmark_id,
        adapter.version,
        adapter.iter_architectures(),
        count=1,
        seed=2026,
    )

    assert manifest["search_space_id"] == "transnas_micro"
    assert manifest["benchmark_variant"] == "micro"
    assert manifest["source_sha256"] == "a" * 64
    assert manifest["converted_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert adapter.metadata()["converted_file_sha256"] == manifest["converted_file_sha256"]


def test_valid_neg_loss_protocol_requires_explicit_maximize_direction():
    args = build_parser().parse_args(
        [
            "evaluate",
            "--benchmark",
            "transnasbench101",
            "--dataset",
            "room_layout",
            "--target-metric",
            "valid_neg_loss",
            "--target-direction",
            "maximize",
        ]
    )

    assert args.target_metric == "valid_neg_loss"
    assert args.target_direction == "maximize"


def test_transnas_manifest_reader_rejects_micro_macro_mismatch(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "transnasbench101",
                "benchmark_version": "v10141024",
                "search_space_id": "transnas_macro",
                "selected": [],
                "shards": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="search_space_id"):
        load_sample_indices(
            path,
            benchmark_id="transnasbench101",
            benchmark_version="v10141024",
            search_space_id="transnas_micro",
        )
