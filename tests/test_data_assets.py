import hashlib
import json

import pytest

from zcp_test.data import DataAsset, DataRegistry, JsonlTable, vitbench101_release_parser
from zcp_test.data.jsonl import convert_trusted_torch_records, write_jsonl_atomic


def test_registry_round_trip_and_checksum(tmp_path):
    data = tmp_path / "asset.bin"
    data.write_bytes(b"benchmark")
    checksum = hashlib.sha256(b"benchmark").hexdigest()
    registry = DataRegistry(tmp_path / "registry.json")
    registry.register(DataAsset("nb", str(data), "1", sha256=checksum, trusted=True))
    assert registry.get("nb").version == "1"
    assert registry.verify("nb")["valid"] is True


def test_registry_rejects_duplicate(tmp_path):
    registry = DataRegistry(tmp_path / "registry.json")
    asset = DataAsset("nb", "/missing", "1")
    registry.register(asset)
    with pytest.raises(KeyError):
        registry.register(asset)


def test_jsonl_runtime_is_strict(tmp_path):
    path = write_jsonl_atomic([{"record_kind": "benchmark_architecture", "value": 1}], tmp_path / "table.jsonl")
    assert JsonlTable(path, expected_kind="benchmark_architecture").load()[0]["value"] == 1
    path.write_text(json.dumps([1, 2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        JsonlTable(path).load()


def test_torch_conversion_requires_explicit_trust(tmp_path):
    with pytest.raises(PermissionError):
        convert_trusted_torch_records(
            tmp_path / "untrusted.pth", tmp_path / "out.jsonl", lambda _: [], trusted=False
        )


def test_torch_conversion_refuses_unsafe_pickle_fallback(monkeypatch, tmp_path):
    import torch

    source = tmp_path / "trusted.pth"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("weights_only unsupported")),
    )

    with pytest.raises(RuntimeError, match="unsafe pickle fallback"):
        convert_trusted_torch_records(
            source, tmp_path / "out.jsonl", lambda _: [], trusted=True
        )


def test_vit_release_parser_keeps_training_protocols_distinct():
    records = list(vitbench101_release_parser([{"arch": {"depth": 12}, "c100_base_acc": 70, "c100_kd_acc": 77, "imagenet_super_acc": 74}]))
    names = {metric["metric_name"] for metric in records[0][1]}
    assert names == {"accuracy_vanilla", "accuracy_kd", "accuracy_inherited"}
