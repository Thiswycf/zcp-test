import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from zcp_test.data import setup as data_setup
from zcp_test.data.bootstrap import BootstrapAsset
from zcp_test.data.assets import DataAsset, DataRegistry
from zcp_test.cli import command_data


@pytest.fixture
def tiny_benchmark(monkeypatch, tmp_path):
    payload = b"tiny raw benchmark"
    asset = BootstrapAsset(
        asset_id="tiny_asset",
        version="test-1",
        path="tiny/raw.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    runtime = tmp_path / "tiny/converted/data.jsonl"

    monkeypatch.setattr(data_setup, "BUILTIN_ASSETS", {asset.asset_id: asset})
    monkeypatch.setattr(data_setup, "BENCHMARK_ASSETS", {"tiny": (asset.asset_id,)})
    monkeypatch.setattr(data_setup, "BENCHMARK_SIZES", {"tiny": len(payload)})
    monkeypatch.setattr(
        data_setup,
        "_runtime_paths",
        lambda root, benchmark: (Path(root) / "tiny/converted/data.jsonl",),
    )
    return asset, payload, runtime


def test_data_checklist_transitions_missing_conversion_required_ready(
    tmp_path, tiny_benchmark
):
    asset, payload, runtime = tiny_benchmark

    missing = data_setup.data_checklist(tmp_path)[0]
    assert missing["benchmark_id"] == "tiny"
    assert missing["state"] == "missing"
    assert missing["remediation"] is not None

    raw = asset.installed_path(tmp_path)
    raw.parent.mkdir(parents=True)
    raw.write_bytes(payload)
    conversion_required = data_setup.data_checklist(tmp_path)[0]
    assert conversion_required["state"] == "conversion_required"
    assert conversion_required["raw_paths"] == [str(raw)]
    assert conversion_required["runtime_paths"] == [str(runtime)]

    runtime.parent.mkdir(parents=True)
    runtime.write_text('{"record_kind":"fixture"}\n', encoding="utf-8")
    ready = data_setup.data_checklist(tmp_path)[0]
    assert ready["state"] == "ready"
    assert ready["remediation"] is None


def test_data_checklist_accepts_valid_external_catalog_runtime(tmp_path, tiny_benchmark):
    _, _, _ = tiny_benchmark
    external = tmp_path / "external/data.jsonl"
    external.parent.mkdir(parents=True)
    external.write_text('{}\n', encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    DataRegistry(catalog).register(DataAsset("tiny", str(external), "test-1", trusted=True))

    record = data_setup.data_checklist(tmp_path / "empty-root", catalog)[0]

    assert record["state"] == "ready"
    assert record["catalog_state"] == "external_ready"
    assert record["location"] == "catalog_external"
    assert record["runtime_paths"] == [str(external)]


def test_external_catalog_honors_declared_checksum_and_bootstrap_skips_ready_asset(
    monkeypatch, tmp_path, tiny_benchmark
):
    _, _, _ = tiny_benchmark
    external = tmp_path / "external/data.jsonl"
    external.parent.mkdir(parents=True)
    external.write_text('{}\n', encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    DataRegistry(catalog).register(
        DataAsset("tiny", str(external), "test-1", sha256="0" * 64, trusted=True)
    )
    assert data_setup.data_checklist(tmp_path / "empty-root", catalog)[0]["state"] == "missing"

    DataRegistry(catalog).register(
        DataAsset(
            "tiny",
            str(external),
            "test-1",
            sha256=hashlib.sha256(external.read_bytes()).hexdigest(),
            trusted=True,
        ),
        replace=True,
    )
    monkeypatch.setattr(
        data_setup,
        "bootstrap_data",
        lambda *args, **kwargs: pytest.fail("ready external asset must not be downloaded"),
    )

    result = data_setup.bootstrap_benchmarks(
        tmp_path / "empty-root", ["tiny"], catalog=catalog
    )

    assert result["ok"] is True
    assert result["checklist"][0]["catalog_state"] == "external_ready"


def test_benchmark_group_expansion_preserves_order_and_deduplicates_groups():
    expanded = data_setup._expand_benchmarks(
        ["vitbench101", "nasbench101", "vitbench101"]
    )

    assert expanded == (
        "vitbench101_autoformer_main",
        "vitbench101_autoformer_ext",
        "vitbench101_pit",
        "nasbench101",
    )
    with pytest.raises(KeyError, match="unknown"):
        data_setup._expand_benchmarks(["unknown"])


def test_export_and_verify_manifest_detects_digest_changes(tmp_path, tiny_benchmark):
    _, _, runtime = tiny_benchmark
    runtime.parent.mkdir(parents=True)
    runtime.write_text('{"score":1}\n', encoding="utf-8")
    manifest_path = tmp_path / "transfer/manifest.json"

    exported = data_setup.export_data_manifest(tmp_path, manifest_path, ["tiny"])
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["records"][0]["path"] == "tiny/converted/data.jsonl"
    assert payload["records"][0]["kind"] == "file"
    verified = data_setup.verify_data_manifest(tmp_path, exported)
    assert verified["valid"] is True
    assert verified["records"][0]["actual_sha256"] == payload["records"][0]["sha256"]

    runtime.write_text('{"score":2}\n', encoding="utf-8")
    tampered = data_setup.verify_data_manifest(tmp_path, exported)
    assert tampered["valid"] is False
    assert tampered["records"][0]["exists"] is True
    assert tampered["records"][0]["actual_sha256"] != payload["records"][0]["sha256"]


def test_export_manifest_rejects_runtime_path_outside_data_root(monkeypatch, tmp_path):
    root = tmp_path / "data"
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(data_setup, "_runtime_paths", lambda root, benchmark: (outside,))

    with pytest.raises(ValueError):
        data_setup.export_data_manifest(root, tmp_path / "manifest.json", ["tiny"])


@pytest.mark.parametrize("unsafe_path", ["../outside.bin", "/tmp/outside.bin"])
def test_verify_manifest_rejects_unsafe_paths(tmp_path, unsafe_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "benchmark_id": "tiny",
                        "path": unsafe_path,
                        "kind": "file",
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe data manifest path"):
        data_setup.verify_data_manifest(tmp_path / "data", manifest)


def test_noninteractive_bootstrap_requires_yes(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    args = Namespace(
        action="bootstrap",
        root=str(tmp_path),
        benchmarks="nasbench101",
        all=False,
        yes=False,
        catalog=str(tmp_path / "catalog.json"),
    )

    with pytest.raises(RuntimeError, match="requires --yes"):
        command_data(args)
