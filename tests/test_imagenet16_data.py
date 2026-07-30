import hashlib
import json
import pickle

import numpy as np
import pytest
import torch

from zcp_test.data import imagenet16
from zcp_test.data.assets import DataAsset, DataRegistry, sha256_file
from zcp_test.data.imagenet16 import (
    SafeImageNet16,
    convert_imagenet16_120,
    verify_safe_imagenet16,
)
from zcp_test.inputs import make_input_batch


def _write_batch(path, labels):
    images = np.arange(len(labels) * 3 * 16 * 16, dtype=np.uint8).reshape(len(labels), -1)
    path.write_bytes(pickle.dumps({"data": images, "labels": labels}, protocol=2))
    return hashlib.md5(path.read_bytes()).hexdigest()


def _source(monkeypatch, tmp_path):
    source = tmp_path / "ImageNet16"
    source.mkdir()
    train_md5 = _write_batch(source / "train", [1, 120, 121])
    valid_md5 = _write_batch(source / "valid", [2, 999])
    monkeypatch.setattr(
        imagenet16,
        "IMAGENET16_FILES",
        {"train": (("train", train_md5),), "valid": (("valid", valid_md5),)},
    )
    return source


def test_imagenet16_conversion_requires_trust(monkeypatch, tmp_path):
    source = _source(monkeypatch, tmp_path)

    with pytest.raises(PermissionError, match="trusted"):
        convert_imagenet16_120(source, tmp_path / "safe")


def test_imagenet16_safe_conversion_dataset_and_input(monkeypatch, tmp_path):
    source = _source(monkeypatch, tmp_path)
    destination = tmp_path / "safe"

    manifest = convert_imagenet16_120(source, destination, trusted=True)

    assert manifest == destination / "manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["splits"] == {"train": 2, "valid": 1}
    assert verify_safe_imagenet16(destination)["valid"] is True
    table = SafeImageNet16(destination, train=True)
    assert len(table) == 2
    assert table[0][0].size == (16, 16)
    assert [table[index][1] for index in range(2)] == [0, 119]
    batch = make_input_batch(
        "dataset", "ImageNet16-120", 2, 16, 120, 7, torch.device("cpu"), str(destination)
    )
    assert batch.inputs.shape == (2, 3, 16, 16)
    assert batch.labels.shape == (2,)
    assert batch.protocol["transform"].endswith("imagenet16-120-official-normalize")
    assert convert_imagenet16_120(source, destination, trusted=True) == manifest


def test_imagenet16_runtime_rejects_tampered_shard(monkeypatch, tmp_path):
    source = _source(monkeypatch, tmp_path)
    destination = tmp_path / "safe"
    manifest = convert_imagenet16_120(source, destination, trusted=True)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    shard = destination / document["shards"][0]["images"]["path"]
    shard.write_bytes(b"tampered")

    assert verify_safe_imagenet16(destination)["valid"] is False
    with pytest.raises(ValueError, match="Unsafe or corrupt"):
        SafeImageNet16(destination)


def test_imagenet16_runtime_rejects_manifest_path_escape(monkeypatch, tmp_path):
    source = _source(monkeypatch, tmp_path)
    destination = tmp_path / "safe"
    manifest = convert_imagenet16_120(source, destination, trusted=True)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["shards"][0]["images"]["path"] = "../outside.npy"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    verification = verify_safe_imagenet16(destination)
    assert verification["valid"] is False
    assert verification["reason"] == "images_path_escape"
    with pytest.raises(ValueError, match="Unsafe or corrupt"):
        SafeImageNet16(destination)


def test_imagenet16_registry_verify_checks_runtime_shards(monkeypatch, tmp_path):
    source = _source(monkeypatch, tmp_path)
    destination = tmp_path / "safe"
    manifest = convert_imagenet16_120(source, destination, trusted=True)
    registry = DataRegistry(tmp_path / "catalog.json")
    registry.register(
        DataAsset(
            "dataset_imagenet16_120",
            str(manifest),
            "npy-shards-v1",
            sha256=sha256_file(manifest),
            protocol="imagenet16-120-official-md5-safe-conversion-v1",
        )
    )
    assert registry.verify("dataset_imagenet16_120")["valid"] is True

    document = json.loads(manifest.read_text(encoding="utf-8"))
    shard = destination / document["shards"][0]["images"]["path"]
    shard.write_bytes(b"tampered")
    verification = registry.verify("dataset_imagenet16_120")
    assert verification["valid"] is False
    assert verification["runtime_integrity"]["reason"] == "images_integrity_failed"


def test_imagenet16_replace_preserves_destination_until_conversion_succeeds(
    monkeypatch, tmp_path
):
    source = _source(monkeypatch, tmp_path)
    destination = tmp_path / "safe"
    manifest = convert_imagenet16_120(source, destination, trusted=True)
    original = manifest.read_bytes()
    monkeypatch.setattr(imagenet16, "_load_trusted_batch", lambda _path: (_ for _ in ()).throw(RuntimeError("stop")))

    with pytest.raises(RuntimeError, match="stop"):
        convert_imagenet16_120(source, destination, trusted=True, replace=True)

    assert manifest.read_bytes() == original
    assert verify_safe_imagenet16(destination)["valid"] is True


def test_imagenet16_restricted_unpickler_rejects_non_numpy_global(monkeypatch, tmp_path):
    source = tmp_path / "ImageNet16"
    source.mkdir()
    payload = pickle.dumps({"data": eval, "labels": [1]}, protocol=2)
    path = source / "train"
    path.write_bytes(payload)
    monkeypatch.setattr(
        imagenet16,
        "IMAGENET16_FILES",
        {"train": (("train", hashlib.md5(payload).hexdigest()),), "valid": ()},
    )

    with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
        convert_imagenet16_120(source, tmp_path / "safe", trusted=True)
