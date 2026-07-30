from __future__ import annotations

import json

import pytest

from zcp_test.cli import build_parser
from zcp_test.research import (
    architecture_stratum,
    create_sample_manifest,
    load_sample_indices,
)
from zcp_test.types import Architecture


def _architectures(count: int = 100):
    for index in range(count):
        operation = "skip_connect" if index < 80 else "nor_conv_3x3"
        encoding = f"|{operation}~0|+|{operation}~0|{operation}~1|+|{operation}~0|{operation}~1|{operation}~2|"
        yield Architecture(
            "nb201_topology",
            f"architecture-{index}",
            {"architecture": encoding},
            index,
        )


def test_feature_stratified_sample_is_deterministic_proportional_and_sharded(tmp_path):
    first = create_sample_manifest(
        "nasbench201", "1.1", _architectures(), fraction=0.1, seed=7, shards=4
    )
    second = create_sample_manifest(
        "nasbench201", "1.1", _architectures(), fraction=0.1, seed=7, shards=4
    )
    assert first == second
    assert first["population_size"] == 100
    assert first["sample_count"] == 10
    assert first["stratum_count"] == 2
    selected_strata = [record["stratum"] for record in first["selected"]]
    assert sorted(selected_strata.count(value) for value in set(selected_strata)) == [2, 8]
    shard_indices = [set(shard["benchmark_indices"]) for shard in first["shards"]]
    assert sum(map(len, shard_indices)) == 10
    assert set.union(*shard_indices) == {
        record["benchmark_index"] for record in first["selected"]
    }
    assert all(left.isdisjoint(right) for index, left in enumerate(shard_indices) for right in shard_indices[index + 1 :])

    path = tmp_path / "sample.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    indices, loaded = load_sample_indices(
        path, benchmark_id="nasbench201", benchmark_version="1.1", shard_index=2
    )
    assert indices == first["shards"][2]["benchmark_indices"]
    assert loaded["strategy"] == "proportional_feature_stratified"


def test_sample_manifest_rejects_protocol_mismatch_and_invalid_sizes(tmp_path):
    manifest = create_sample_manifest(
        "nasbench201", "1.1", _architectures(10), count=2, seed=1
    )
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_id"):
        load_sample_indices(path, benchmark_id="nats_tss", benchmark_version="1.1")
    with pytest.raises(ValueError, match="benchmark_version"):
        load_sample_indices(path, benchmark_id="nasbench201", benchmark_version="2.0")
    with pytest.raises(ValueError, match="exactly one"):
        create_sample_manifest("nasbench201", "1.1", _architectures(10), count=2, fraction=0.1)


def test_benchmark_specific_strata_cover_size_dag_darts_transnas_and_vit():
    nb101 = architecture_stratum(
        "nasbench101",
        {
            "matrix": [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
            "operations": ["input", "conv3x3-bn-relu", "output"],
        },
    )
    nats = architecture_stratum("nats_sss", {"architecture": "8:16:24:32:40"})
    darts = architecture_stratum(
        "nasbench301_surrogate",
        {
            "normal": [["skip_connect", 0]] * 8,
            "reduce": [["sep_conv_3x3", 0]] * 8,
        },
    )
    transnas = architecture_stratum(
        "transnasbench101", {"architecture": "64-41414-0_00_000"}
    )
    vit = architecture_stratum(
        "vitbench101", {"depth": 12, "hidden_dim": 192}
    )
    assert len({nb101, nats, darts, transnas, vit}) == 5


def test_sampling_cli_parser_and_evaluate_shard_options():
    parser = build_parser()
    sample = parser.parse_args(
        [
            "benchmark",
            "sample",
            "nasbench201",
            "--fraction",
            "0.01",
            "--shards",
            "4",
            "--output",
            "sample.json",
        ]
    )
    assert sample.fraction == pytest.approx(0.01)
    assert sample.shards == 4
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--benchmark",
            "nasbench201",
            "--sample-manifest",
            "sample.json",
            "--sample-shard",
            "2",
        ]
    )
    assert evaluate.sample_shard == 2


def test_benchmark_sample_cli_writes_atomic_manifest(monkeypatch, tmp_path, capsys):
    import zcp_test.cli as cli

    class Adapter:
        benchmark_id = "nasbench201"

        def metadata(self):
            return {"version": "1.1"}

        def iter_architectures(self):
            return _architectures(20)

    monkeypatch.setattr(cli, "load_builtin_spaces", lambda: None)
    monkeypatch.setattr(cli, "load_builtin_benchmarks", lambda: None)
    monkeypatch.setattr(cli, "_resolve_benchmark_path", lambda args: "verified.pth")
    monkeypatch.setattr(cli.BENCHMARKS, "create", lambda name, **kwargs: Adapter())
    output = tmp_path / "samples" / "nb201.json"

    cli.main(
        [
            "benchmark",
            "sample",
            "nasbench201",
            "--trusted",
            "--count",
            "5",
            "--seed",
            "3",
            "--shards",
            "2",
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 5
    assert summary["output"] == str(output)
    assert manifest["benchmark_id"] == "nasbench201"
    assert [shard["sample_count"] for shard in manifest["shards"]] == [3, 2]
    assert not output.with_suffix(".json.tmp").exists()
