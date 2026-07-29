from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from zcp_test.data.nb101_proto import ModelMetrics, parse_model_metrics

try:
    import google_crc32c
except ImportError:  # pragma: no cover - exercised in minimal parser environments
    google_crc32c = None


class TFRecordError(ValueError):
    """Raised when TFRecord framing or a checksum is invalid."""


@dataclass(frozen=True)
class TFRecord:
    data: bytes
    offset: int
    next_offset: int


@dataclass(frozen=True)
class NasbenchRecord:
    module_hash: str
    epochs: int
    matrix: tuple[tuple[int, ...], ...]
    operations: tuple[str, ...]
    metrics: ModelMetrics


_CRC32C_TABLE: tuple[int, ...] | None = None


def crc32c(data: bytes) -> int:
    """Return the Castagnoli CRC-32C used by TFRecord."""
    if google_crc32c is not None:
        return int(google_crc32c.value(data))
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        polynomial = 0x82F63B78
        table = []
        for initial in range(256):
            value = initial
            for _ in range(8):
                value = (value >> 1) ^ polynomial if value & 1 else value >> 1
            table.append(value)
        _CRC32C_TABLE = tuple(table)
    checksum = 0xFFFFFFFF
    for byte in data:
        checksum = _CRC32C_TABLE[(checksum ^ byte) & 0xFF] ^ (checksum >> 8)
    return checksum ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    checksum = crc32c(data)
    return (((checksum >> 15) | (checksum << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord_records(
    source: str | Path | BinaryIO,
    *,
    verify_crc: bool = True,
    start_offset: int = 0,
    max_record_size: int | None = 64 * 1024 * 1024,
) -> Iterator[TFRecord]:
    """Yield framed TFRecord payloads while retaining restart-safe offsets."""
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if max_record_size is not None and max_record_size < 0:
        raise ValueError("max_record_size must be non-negative or None")
    close_handle = not hasattr(source, "read")
    handle = open(Path(source).expanduser(), "rb") if close_handle else source
    try:
        handle.seek(start_offset)
        while True:
            offset = handle.tell()
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise TFRecordError(f"Truncated TFRecord length at byte {offset}")
            length_crc_bytes = handle.read(4)
            if len(length_crc_bytes) != 4:
                raise TFRecordError(f"Truncated TFRecord length CRC at byte {offset}")
            if verify_crc and struct.unpack("<I", length_crc_bytes)[0] != masked_crc32c(
                length_bytes
            ):
                raise TFRecordError(f"Invalid TFRecord length CRC at byte {offset}")
            length = struct.unpack("<Q", length_bytes)[0]
            if max_record_size is not None and length > max_record_size:
                raise TFRecordError(
                    f"TFRecord payload at byte {offset} exceeds {max_record_size} bytes"
                )
            data = handle.read(length)
            if len(data) != length:
                raise TFRecordError(f"Truncated TFRecord payload at byte {offset}")
            data_crc_bytes = handle.read(4)
            if len(data_crc_bytes) != 4:
                raise TFRecordError(f"Truncated TFRecord data CRC at byte {offset}")
            if verify_crc and struct.unpack("<I", data_crc_bytes)[0] != masked_crc32c(data):
                raise TFRecordError(f"Invalid TFRecord data CRC at byte {offset}")
            yield TFRecord(data, offset, handle.tell())
    finally:
        if close_handle:
            handle.close()


def iter_tfrecord(
    source: str | Path | BinaryIO,
    *,
    verify_crc: bool = True,
    start_offset: int = 0,
    max_record_size: int | None = 64 * 1024 * 1024,
) -> Iterator[bytes]:
    for record in iter_tfrecord_records(
        source,
        verify_crc=verify_crc,
        start_offset=start_offset,
        max_record_size=max_record_size,
    ):
        yield record.data


def write_tfrecord(records: Iterable[bytes], destination: str | Path | BinaryIO) -> Path | None:
    """Write standard TFRecord framing; useful for small fixtures and exports."""
    close_handle = not hasattr(destination, "write")
    path = Path(destination).expanduser() if close_handle else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("wb")
    else:
        handle = destination
    try:
        for data in records:
            if not isinstance(data, bytes):
                raise TypeError("TFRecord payloads must be bytes")
            length_bytes = struct.pack("<Q", len(data))
            handle.write(length_bytes)
            handle.write(struct.pack("<I", masked_crc32c(length_bytes)))
            handle.write(data)
            handle.write(struct.pack("<I", masked_crc32c(data)))
        handle.flush()
    finally:
        if close_handle:
            handle.close()
    return path


def parse_nasbench_record(data: bytes) -> NasbenchRecord:
    """Parse one NAS-Bench-101 JSON tuple and its embedded ModelMetrics."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("NAS-Bench-101 record is not valid UTF-8 JSON") from error
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError("NAS-Bench-101 record must be a five-element JSON list")
    module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = payload
    if not isinstance(module_hash, str) or not module_hash:
        raise ValueError("NAS-Bench-101 module hash must be a non-empty string")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("NAS-Bench-101 epoch budget must be a positive integer")
    if not isinstance(raw_adjacency, str) or not isinstance(raw_operations, str):
        raise ValueError("NAS-Bench-101 adjacency and operations must be strings")
    operations = tuple(raw_operations.split(","))
    dimension = len(operations)
    if dimension == 0 or len(raw_adjacency) != dimension * dimension:
        raise ValueError("NAS-Bench-101 adjacency size does not match operations")
    if any(value not in "01" for value in raw_adjacency):
        raise ValueError("NAS-Bench-101 adjacency must contain only zero and one")
    matrix = tuple(
        tuple(int(value) for value in raw_adjacency[row * dimension : (row + 1) * dimension])
        for row in range(dimension)
    )
    if not isinstance(raw_metrics, str):
        raise ValueError("NAS-Bench-101 ModelMetrics must be base64 text")
    try:
        compact_metrics = "".join(raw_metrics.split())
        metrics_data = base64.b64decode(compact_metrics, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("NAS-Bench-101 ModelMetrics is not valid base64") from error
    return NasbenchRecord(
        module_hash, epochs, matrix, operations, parse_model_metrics(metrics_data)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS architectures (
          module_hash TEXT PRIMARY KEY, matrix TEXT NOT NULL, operations TEXT NOT NULL,
          first_offset INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_records (
          source_offset INTEGER PRIMARY KEY, module_hash TEXT NOT NULL, epochs INTEGER NOT NULL,
          metrics BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS source_records_hash ON source_records(module_hash, source_offset);
        """
    )
    return connection


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata"))


def _set_metadata(connection: sqlite3.Connection, **values: Any) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in values.items()],
    )


def _ingest(
    source: Path, state_path: Path, source_sha256: str, *, verify_crc: bool, commit_every: int
) -> sqlite3.Connection:
    connection = _connect_state(state_path)
    identity = {
        "source_path": str(source),
        "source_size": str(source.stat().st_size),
        "source_sha256": source_sha256,
    }
    existing = _metadata(connection)
    for key, value in identity.items():
        if key in existing and existing[key] != value:
            connection.close()
            raise ValueError(f"Conversion state belongs to a different source ({key})")
    _set_metadata(connection, **identity)
    connection.commit()
    start_offset = int(existing.get("next_offset", 0))
    pending = 0
    try:
        for framed in iter_tfrecord_records(
            source, verify_crc=verify_crc, start_offset=start_offset
        ):
            parsed = parse_nasbench_record(framed.data)
            matrix_json = json.dumps(parsed.matrix, separators=(",", ":"))
            operations_json = json.dumps(parsed.operations, separators=(",", ":"))
            current = connection.execute(
                "SELECT matrix, operations FROM architectures WHERE module_hash = ?",
                (parsed.module_hash,),
            ).fetchone()
            if current is not None and current != (matrix_json, operations_json):
                raise ValueError(f"Conflicting specifications for module hash {parsed.module_hash}")
            connection.execute(
                "INSERT OR IGNORE INTO architectures VALUES (?, ?, ?, ?)",
                (parsed.module_hash, matrix_json, operations_json, framed.offset),
            )
            raw_metrics = base64.b64decode(
                "".join(json.loads(framed.data.decode("utf-8"))[4].split()), validate=True
            )
            connection.execute(
                "INSERT OR IGNORE INTO source_records VALUES (?, ?, ?, ?)",
                (framed.offset, parsed.module_hash, parsed.epochs, raw_metrics),
            )
            _set_metadata(connection, next_offset=framed.next_offset)
            pending += 1
            if pending >= commit_every:
                connection.commit()
                pending = 0
        _set_metadata(connection, ingestion_complete=1, next_offset=source.stat().st_size)
        connection.commit()
        return connection
    except BaseException:
        connection.rollback()
        connection.close()
        raise


def _normalized_metrics(rows: list[tuple[int, bytes]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    repeats: dict[int, int] = {}
    for epochs, raw_metrics in rows:
        seed = repeats.get(epochs, 0)
        repeats[epochs] = seed + 1
        metrics = parse_model_metrics(raw_metrics)
        evaluations = metrics.evaluation_data
        points = (("halfway", 1), ("final", 2))
        for point, evaluation_index in points:
            if evaluation_index >= len(evaluations):
                continue
            evaluation = evaluations[evaluation_index]
            metric_name = f"{point}_accuracy"
            evaluation_metadata: dict[str, Any] = {"evaluation_point": point}
            if evaluation.current_epoch is not None:
                evaluation_metadata["current_epoch"] = evaluation.current_epoch
            if evaluation.checkpoint_path is not None:
                evaluation_metadata["checkpoint_path"] = evaluation.checkpoint_path
            for split, value in (
                ("train", evaluation.train_accuracy),
                ("valid", evaluation.validation_accuracy),
                ("test", evaluation.test_accuracy),
            ):
                if value is not None:
                    output.append(
                        {
                            "dataset": "cifar10",
                            "split": split,
                            "metric_name": metric_name,
                            "epoch_budget": epochs,
                            "seed": seed,
                            "value": value,
                            **evaluation_metadata,
                        }
                    )
            if evaluation.training_time is not None:
                output.append(
                    {
                        "dataset": "cifar10",
                        "split": "benchmark",
                            "metric_name": f"{point}_training_time",
                        "epoch_budget": epochs,
                        "seed": seed,
                        "value": evaluation.training_time,
                        **evaluation_metadata,
                    }
                )
        if metrics.trainable_parameters is not None:
            output.append(
                {
                    "dataset": "cifar10",
                    "split": "benchmark",
                    "metric_name": "trainable_parameters",
                    "epoch_budget": epochs,
                    "seed": seed,
                    "value": metrics.trainable_parameters,
                }
            )
        if metrics.total_time is not None:
            output.append(
                {
                    "dataset": "cifar10",
                    "split": "benchmark",
                    "metric_name": "total_time",
                    "epoch_budget": epochs,
                    "seed": seed,
                    "value": metrics.total_time,
                }
            )
    return output


def _write_conversion(
    connection: sqlite3.Connection,
    destination: Path,
    source: Path,
    source_sha256: str,
    *,
    records_per_shard: int,
    benchmark_version: str,
) -> Path:
    for temporary in destination.glob(".*.tmp"):
        temporary.unlink()
    manifest_path = destination / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    index_temporary = destination / ".hash-index.json.tmp"
    offsets_temporary = destination / ".offsets.bin.tmp"
    shard_summaries = []
    architecture_count = connection.execute("SELECT COUNT(*) FROM architectures").fetchone()[0]
    shard_handle = None
    shard_path: Path | None = None
    shard_count = 0
    shard_number = -1
    try:
        with (
            index_temporary.open("w", encoding="utf-8", newline="\n") as index_handle,
            offsets_temporary.open("wb") as offsets_handle,
        ):
            index_handle.write('{"schema_version":1,"hash_to_index":{')
            first_index_entry = True
            architectures = connection.execute(
                "SELECT module_hash, matrix, operations FROM architectures ORDER BY first_offset"
            )
            for benchmark_index, (module_hash, matrix_json, operations_json) in enumerate(
                architectures
            ):
                if shard_handle is None or shard_count >= records_per_shard:
                    if shard_handle is not None and shard_path is not None:
                        shard_handle.close()
                        final_path = destination / shard_path.name[1:-4]
                        shard_path.replace(final_path)
                        shard_summaries.append(_shard_summary(final_path, shard_count))
                    shard_number += 1
                    shard_count = 0
                    shard_path = destination / f".architectures-{shard_number:05d}.jsonl.tmp"
                    shard_handle = shard_path.open("wb")
                rows = list(
                    connection.execute(
                        "SELECT epochs, metrics FROM source_records WHERE module_hash = ? "
                        "ORDER BY source_offset",
                        (module_hash,),
                    )
                )
                record = {
                    "record_kind": "benchmark_architecture",
                    "benchmark_id": "nasbench101",
                    "search_space_id": "nb101_dag",
                    "benchmark_version": benchmark_version,
                    "benchmark_index": benchmark_index,
                    "protocol": "official-tfrecord",
                    "source_sha256": source_sha256,
                    "module_hash": module_hash,
                    "specification": {
                        "matrix": json.loads(matrix_json),
                        "operations": json.loads(operations_json),
                    },
                    "metrics": _normalized_metrics(rows),
                }
                encoded = (
                    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    + "\n"
                ).encode()
                offset = shard_handle.tell()
                shard_handle.write(encoded)
                if not first_index_entry:
                    index_handle.write(",")
                first_index_entry = False
                index_handle.write(json.dumps(module_hash))
                index_handle.write(":")
                index_handle.write(str(benchmark_index))
                offsets_handle.write(struct.pack("<IQI", shard_number, offset, len(encoded)))
                shard_count += 1
            index_handle.write("}}\n")
    finally:
        if shard_handle is not None and not shard_handle.closed:
            shard_handle.close()
    if shard_path is not None:
        final_path = destination / shard_path.name[1:-4]
        shard_path.replace(final_path)
        shard_summaries.append(_shard_summary(final_path, shard_count))
    index_path = destination / "hash-index.json"
    index_temporary.replace(index_path)
    offsets_path = destination / "offsets.bin"
    offsets_temporary.replace(offsets_path)
    for stale in destination.glob("architectures-*.jsonl"):
        if stale.name not in {entry["path"] for entry in shard_summaries}:
            stale.unlink()
    manifest = {
        "format": "zcp-test-nasbench101-sharded-jsonl",
        "format_version": 2,
        "benchmark_id": "nasbench101",
        "search_space_id": "nb101_dag",
        "benchmark_version": benchmark_version,
        "architecture_count": architecture_count,
        "records_per_shard": records_per_shard,
        "source": {"path": str(source), "size": source.stat().st_size, "sha256": source_sha256},
        "index": {
            "path": index_path.name,
            "kind": "module_hash_to_architecture_index",
            "sha256": _sha256_file(index_path),
        },
        "offsets": {
            "path": offsets_path.name,
            "kind": "little_endian_uint32_uint64_uint32",
            "entry_size": struct.calcsize("<IQI"),
            "sha256": _sha256_file(offsets_path),
        },
        "shards": shard_summaries,
    }
    manifest_temporary = destination / ".manifest.json.tmp"
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)
    return manifest_path


def _shard_summary(path: Path, record_count: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "record_count": record_count,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def convert_nasbench101(
    source: str | Path,
    destination: str | Path,
    *,
    records_per_shard: int = 10_000,
    benchmark_version: str = "full",
    verify_crc: bool = True,
    commit_every: int = 1_000,
) -> Path:
    """Stream an official TFRecord into resumable sharded JSONL output.

    The returned path is the manifest. A temporary SQLite database stores only
    conversion state and is removed after the manifest is published.
    """
    if records_per_shard <= 0 or commit_every <= 0:
        raise ValueError("records_per_shard and commit_every must be positive")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"NAS-Bench-101 TFRecord does not exist: {source_path}")
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(source_path)
    state_path = destination_path / ".nasbench101-conversion.sqlite3"
    connection = _ingest(
        source_path,
        state_path,
        source_sha256,
        verify_crc=verify_crc,
        commit_every=commit_every,
    )
    try:
        manifest_path = _write_conversion(
            connection,
            destination_path,
            source_path,
            source_sha256,
            records_per_shard=records_per_shard,
            benchmark_version=benchmark_version,
        )
    finally:
        connection.close()
    state_path.unlink(missing_ok=True)
    Path(str(state_path) + "-wal").unlink(missing_ok=True)
    Path(str(state_path) + "-shm").unlink(missing_ok=True)
    return manifest_path


def read_indexed_record(destination: str | Path, module_hash: str) -> dict[str, Any]:
    """Read one converted architecture through the manifest hash/offset index."""
    directory = Path(destination).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((directory / manifest["index"]["path"]).read_text(encoding="utf-8"))
    try:
        benchmark_index = int(index["hash_to_index"][module_hash])
    except KeyError as error:
        raise KeyError(f"Unknown NAS-Bench-101 module hash: {module_hash}") from error
    entry_size = int(manifest["offsets"]["entry_size"])
    with (directory / manifest["offsets"]["path"]).open("rb") as handle:
        handle.seek(benchmark_index * entry_size)
        shard_number, offset, length = struct.unpack("<IQI", handle.read(entry_size))
    shard = directory / manifest["shards"][shard_number]["path"]
    with shard.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.read(length))


NasBench101Record = NasbenchRecord
convert_nasbench101_tfrecord = convert_nasbench101


__all__ = [
    "NasbenchRecord",
    "NasBench101Record",
    "TFRecord",
    "TFRecordError",
    "convert_nasbench101",
    "convert_nasbench101_tfrecord",
    "crc32c",
    "iter_tfrecord",
    "iter_tfrecord_records",
    "masked_crc32c",
    "parse_nasbench_record",
    "read_indexed_record",
    "write_tfrecord",
]
