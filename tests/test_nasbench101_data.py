import base64
import json
import struct

import pytest

import zcp_test.data.nasbench101 as nb101
from zcp_test.benchmarks.nasbench101 import NasBench101Adapter
from zcp_test.data.nb101_proto import ProtobufDecodeError, parse_model_metrics
from zcp_test.data.vendor.model_metrics_pb2 import ModelMetrics
from zcp_test.spaces.nb101 import Nb101Space
from zcp_test.types import MetricSpec


def _varint(value):
    output = bytearray()
    while value > 0x7F:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _evaluation(epoch, time, train, valid, test):
    return b"".join(
        bytes([field_number << 3 | 1]) + struct.pack("<d", value)
        for field_number, value in enumerate((epoch, time, train, valid, test), 1)
    )


def _model_metrics(test_accuracy=0.91):
    evaluations = [
        _evaluation(0, 0, 0.1, 0.1, 0.1),
        _evaluation(54, 5, 0.8, 0.75, 0.74),
        _evaluation(108, 10, 0.95, 0.92, test_accuracy),
    ]
    message = bytearray()
    for evaluation in evaluations:
        message.extend(b"\x0a" + _varint(len(evaluation)) + evaluation)
    message.extend(b"\x10" + _varint(12345))
    message.extend(b"\x19" + struct.pack("<d", 12.5))
    message.extend(b"\x20\x07")
    return bytes(message)


def _record(module_hash, test_accuracy=0.91, epochs=108):
    payload = [
        module_hash,
        epochs,
        "010001000",
        "input,conv1x1-bn-relu,output",
        base64.b64encode(_model_metrics(test_accuracy)).decode("ascii"),
    ]
    return json.dumps(payload).encode("utf-8")


def test_tfrecord_framing_checks_crc(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    nb101.write_tfrecord([b"first", b"second"], path)

    framed = list(nb101.iter_tfrecord_records(path))
    assert [record.data for record in framed] == [b"first", b"second"]
    assert list(nb101.iter_tfrecord(path, start_offset=framed[1].offset)) == [b"second"]
    assert nb101.crc32c(b"123456789") == 0xE3069283

    damaged = bytearray(path.read_bytes())
    damaged[12] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(nb101.TFRecordError, match="data CRC"):
        list(nb101.iter_tfrecord(path))


def test_lightweight_model_metrics_parser():
    metrics = parse_model_metrics(_model_metrics())
    assert metrics.trainable_parameters == 12345
    assert metrics.total_time == 12.5
    assert metrics.evaluation_data[2].validation_accuracy == 0.92
    assert metrics.evaluation_data[2].test_accuracy == 0.91

    with pytest.raises(ProtobufDecodeError, match="Truncated"):
        parse_model_metrics(b"\x0a\x05\x09")

    generated = ModelMetrics.FromString(_model_metrics())
    assert generated.trainable_parameters == metrics.trainable_parameters
    assert generated.evaluation_data[2].test_accuracy == metrics.evaluation_data[2].test_accuracy


def test_official_base64_line_wrapping_is_accepted():
    encoded = base64.b64encode(_model_metrics()).decode("ascii")
    wrapped = "\n".join((encoded[:40], encoded[40:80], encoded[80:]))
    payload = json.loads(_record("hash-a"))
    payload[4] = wrapped

    record = nb101.parse_nasbench_record(json.dumps(payload).encode("utf-8"))

    assert record.metrics.evaluation_data[2].test_accuracy == 0.91


def test_convert_to_shards_manifest_and_hash_offsets(tmp_path):
    source = tmp_path / "fixture.tfrecord"
    nb101.write_tfrecord(
        [_record("hash-a", 0.91), _record("hash-a", 0.93), _record("hash-b", 0.89)],
        source,
    )

    manifest_path = nb101.convert_nasbench101(
        source, tmp_path / "converted", records_per_shard=1, commit_every=1
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["architecture_count"] == 2
    assert [shard["record_count"] for shard in manifest["shards"]] == [1, 1]
    assert manifest["format_version"] == 2
    assert (manifest_path.parent / "hash-index.json").is_file()
    assert (manifest_path.parent / "offsets.bin").stat().st_size == 2 * struct.calcsize("<IQI")
    assert not (manifest_path.parent / ".nasbench101-conversion.sqlite3").exists()

    record = nb101.read_indexed_record(manifest_path.parent, "hash-a")
    assert record["specification"]["operations"][0] == "input"
    final_test = [
        metric
        for metric in record["metrics"]
        if metric["split"] == "test" and metric["metric_name"] == "final_accuracy"
    ]
    assert [(metric["seed"], metric["value"]) for metric in final_test] == [
        (0, 0.91),
        (1, 0.93),
    ]


def test_conversion_resumes_from_committed_tfrecord_offset(tmp_path, monkeypatch):
    source = tmp_path / "fixture.tfrecord"
    nb101.write_tfrecord([_record("hash-a"), _record("hash-b")], source)
    destination = tmp_path / "converted"
    original_parser = nb101.parse_nasbench_record
    calls = 0

    def interrupt_once(data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original_parser(data)

    monkeypatch.setattr(nb101, "parse_nasbench_record", interrupt_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        nb101.convert_nasbench101(source, destination, commit_every=1)
    assert (destination / ".nasbench101-conversion.sqlite3").exists()

    monkeypatch.setattr(nb101, "parse_nasbench_record", original_parser)
    manifest_path = nb101.convert_nasbench101(source, destination, commit_every=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["architecture_count"] == 2


def test_converted_manifest_supports_unified_and_native_queries(tmp_path):
    specification = {
        "matrix": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "operations": ["input", "conv1x1-bn-relu", "output"],
    }
    module_hash = Nb101Space().canonicalize(specification).architecture_id
    source = tmp_path / "fixture.tfrecord"
    nb101.write_tfrecord([_record(module_hash, 0.91), _record(module_hash, 0.93)], source)
    manifest = nb101.convert_nasbench101(source, tmp_path / "converted", commit_every=1)
    adapter = NasBench101Adapter(str(manifest), version="full")
    architecture = next(iter(adapter.iter_architectures()))

    metric = adapter.query_metrics(
        architecture,
        MetricSpec("cifar10", "test", "final_accuracy", 108, seed_reduction="mean"),
    )
    native = adapter.query_native(specification, epochs=108)
    native_from_architecture = adapter.query_native(architecture, epochs=108)

    assert metric["final_accuracy"] == pytest.approx(0.92)
    assert native["test_accuracy"] == pytest.approx(0.92)
    assert native_from_architecture == native
    assert native["trainable_parameters"] == 12345
    assert adapter.is_valid(specification)


def test_nb101_query_requires_explicit_budget_when_multiple_match(tmp_path):
    specification = {
        "matrix": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "operations": ["input", "conv1x1-bn-relu", "output"],
    }
    module_hash = Nb101Space().canonicalize(specification).architecture_id
    source = tmp_path / "fixture.tfrecord"
    nb101.write_tfrecord(
        [_record(module_hash, 0.5, epochs=4), _record(module_hash, 0.9, epochs=108)], source
    )
    manifest = nb101.convert_nasbench101(source, tmp_path / "converted", commit_every=1)
    adapter = NasBench101Adapter(str(manifest), version="full")
    architecture = next(adapter.iter_architectures())

    with pytest.raises(ValueError, match="multiple epoch budgets"):
        adapter.query_metrics(
            architecture,
            MetricSpec("cifar10", "test", "final_accuracy", seed_reduction="mean"),
        )
