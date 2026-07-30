import builtins
import json
import pickle
from pathlib import Path

import pytest

from zcp_test.data import converters
from zcp_test.doctor import diagnostics
from zcp_test.legacy import import_pickle


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_doctor_reports_packages_torch_and_catalog(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    report = diagnostics(catalog)
    assert report["python"]
    assert set(report["packages"]) == {
        "torch", "torchvision", "numpy", "scipy", "nats_bench", "nasbench301", "h5py", "timm"
    }
    assert report["torch"]["version"]
    assert report["data_catalog"] == {"path": str(catalog), "exists": True}


def test_doctor_handles_missing_torch(monkeypatch):
    original_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    assert diagnostics()["torch"] is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([{"score": 1}, 2], [{"legacy_index": 0, "score": 1}, {"legacy_index": 1, "value": 2}]),
        ({"a": 1}, [{"legacy_index": 0, "key": "a", "value": 1}]),
        ("value", [{"legacy_index": 0, "value": "value"}]),
    ],
)
def test_legacy_pickle_shapes_require_trust(tmp_path, payload, expected):
    source = tmp_path / "legacy.pkl"
    destination = tmp_path / "legacy.jsonl"
    source.write_bytes(pickle.dumps(payload))
    with pytest.raises(PermissionError, match="trusted"):
        import_pickle(source, destination)
    assert import_pickle(source, destination, trusted=True) == len(expected)
    assert _jsonl(destination) == expected


def test_vitbench_release_parser_preserves_metric_protocols():
    payload = [{
        "arch": {"hidden_dim": 192},
        "c100_base_acc": 80,
        "c100_kd_acc": 82,
        "imagenet_super_acc": 75,
    }]
    specification, metrics = next(converters.vitbench101_release_parser(payload))
    assert specification == {"hidden_dim": 192}
    assert {(row["dataset"], row["metric_name"]) for row in metrics} == {
        ("cifar100", "accuracy_vanilla"),
        ("cifar100", "accuracy_kd"),
        ("imagenet1k", "accuracy_inherited"),
    }
    with pytest.raises(ValueError, match="list"):
        list(converters.vitbench101_release_parser({}))
    with pytest.raises(ValueError, match="Invalid"):
        list(converters.vitbench101_release_parser([{"arch": "not-a-mapping"}]))


def test_convert_trusted_benchmark_validates_parser_contract(monkeypatch, tmp_path):
    source = tmp_path / "source.pth"
    source.write_bytes(b"trusted fixture")
    monkeypatch.setattr(converters, "sha256_file", lambda path: "fixture-sha")

    captured = {}

    def fake_convert(source_path, destination, convert, trusted):
        captured["rows"] = list(convert("payload"))
        captured["trusted"] = trusted
        return Path(destination)

    monkeypatch.setattr(converters, "convert_trusted_torch_records", fake_convert)
    destination = tmp_path / "runtime.jsonl"
    result = converters.convert_trusted_benchmark(
        source,
        destination,
        benchmark_id="benchmark",
        search_space_id="space",
        benchmark_version="1",
        protocol="protocol",
        parser=lambda payload: [({"x": 1}, [{"value": 2}])],
        trusted=True,
    )
    assert result == destination
    assert captured["trusted"] is True
    assert captured["rows"][0]["source_sha256"] == "fixture-sha"
    assert captured["rows"][0]["metrics"] == [{"value": 2}]

    with pytest.raises(ValueError, match="specification mapping"):
        converters.convert_trusted_benchmark(
            source,
            destination,
            benchmark_id="benchmark",
            search_space_id="space",
            benchmark_version="1",
            protocol="protocol",
            parser=lambda payload: [("invalid", [])],
            trusted=True,
        )


def test_convert_vitbench_slice_and_transnas_parser(monkeypatch, tmp_path):
    captured = {}

    def fake_convert(source, destination, **kwargs):
        captured.update(kwargs)
        captured["rows"] = list(kwargs["parser"](captured.pop("payload", {})))
        return Path(destination)

    monkeypatch.setattr(converters, "convert_trusted_benchmark", fake_convert)
    with pytest.raises(ValueError, match="Unknown"):
        converters.convert_vitbench101("source", "destination", slice_id="unknown", parser=lambda _: [], trusted=True)

    captured["payload"] = {
        "data": {
            "micro": {
                "arch-1": {
                    "classification": {
                        "total_epochs": 12,
                        "metrics": {
                            "valid_accuracy": [0.1, 0.8],
                            "empty": [],
                            "invalid": [float("nan")],
                        },
                    }
                }
            }
        }
    }
    result = converters.convert_transnasbench101(
        "source", tmp_path / "transnas.jsonl", space="micro", trusted=True
    )
    assert result.name == "transnas.jsonl"
    specification, metrics = captured["rows"][0]
    assert specification == {"architecture": "arch-1"}
    assert metrics == [{
        "dataset": "classification",
        "split": "valid",
        "metric_name": "valid_accuracy",
        "epoch_budget": 12,
        "value": 0.8,
    }]
    with pytest.raises(ValueError, match="micro.*macro"):
        converters.convert_transnasbench101("source", "destination", space="invalid", trusted=True)

    captured["payload"] = {}
    with pytest.raises(ValueError, match="does not contain"):
        converters.convert_transnasbench101("source", "destination", space="macro", trusted=True)
