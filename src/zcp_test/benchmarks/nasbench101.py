from __future__ import annotations

import json
import random
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from zcp_test.benchmarks.base import BenchmarkAdapter
from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder
from zcp_test.spaces.nb101 import Nb101Space
from zcp_test.types import Architecture, MetricSpec


class NasBench101Adapter(BenchmarkAdapter):
    benchmark_id = "nasbench101"
    search_space_id = "nb101_dag"

    def __init__(self, path: str, *, version: str | None = None, model_builder=None) -> None:
        source = Path(path).expanduser().resolve()
        self._model_builder = model_builder or default_model_builder
        self._space = Nb101Space()
        self._legacy: JsonlBenchmarkAdapter | None = None
        if source.is_file() and source.name != "manifest.json":
            self._legacy = JsonlBenchmarkAdapter(
                str(source),
                benchmark_id=self.benchmark_id,
                search_space_id=self.search_space_id,
                version=version,
                model_builder=self._model_builder,
            )
            self.path = source
            self.version = self._legacy.version
            return
        manifest_path = source / "manifest.json" if source.is_dir() else source
        if not manifest_path.is_file():
            raise FileNotFoundError(f"NAS-Bench-101 manifest does not exist: {manifest_path}")
        self.directory = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "zcp-test-nasbench101-sharded-jsonl":
            raise ValueError(f"Unsupported NAS-Bench-101 manifest: {manifest_path}")
        if self.manifest.get("benchmark_id") != self.benchmark_id:
            raise ValueError("NAS-Bench-101 manifest benchmark_id mismatch")
        self.version = str(self.manifest["benchmark_version"])
        if version and version != self.version:
            raise ValueError(f"Expected NAS-Bench-101 version {version!r}, found {self.version!r}")
        self.path = manifest_path
        self._index: list[dict[str, Any]] = []
        self._by_hash: dict[str, dict[str, Any]] = {}
        index_path = self.directory / self.manifest["index"]["path"]
        if int(self.manifest.get("format_version", 1)) >= 2:
            hash_to_index = json.loads(index_path.read_text(encoding="utf-8"))["hash_to_index"]
            entry_size = int(self.manifest["offsets"]["entry_size"])
            offsets_path = self.directory / self.manifest["offsets"]["path"]
            offsets_data = offsets_path.read_bytes()
            if len(offsets_data) != len(hash_to_index) * entry_size:
                raise ValueError("NAS-Bench-101 offsets size does not match hash index")
            entries: list[dict[str, Any] | None] = [None] * len(hash_to_index)
            for module_hash, benchmark_index_value in hash_to_index.items():
                benchmark_index = int(benchmark_index_value)
                shard_number, offset, length = struct.unpack_from(
                    "<IQI", offsets_data, benchmark_index * entry_size
                )
                entry = {
                    "module_hash": module_hash,
                    "shard": self.manifest["shards"][shard_number]["path"],
                    "offset": offset,
                    "length": length,
                    "benchmark_index": benchmark_index,
                }
                entries[benchmark_index] = entry
                self._by_hash[module_hash] = entry
            if any(entry is None for entry in entries):
                raise ValueError("NAS-Bench-101 offsets contain missing architecture indices")
            self._index = [entry for entry in entries if entry is not None]
        else:
            with index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = json.loads(line)
                    self._index.append(entry)
                    self._by_hash[str(entry["module_hash"])] = entry
        if len(self._index) != int(self.manifest["architecture_count"]):
            raise ValueError("NAS-Bench-101 index count does not match manifest")

    def _record(self, module_hash: str) -> dict[str, Any]:
        try:
            entry = self._by_hash[module_hash]
        except KeyError as error:
            raise KeyError(f"Unknown NAS-Bench-101 module hash: {module_hash}") from error
        with (self.directory / entry["shard"]).open("rb") as handle:
            handle.seek(int(entry["offset"]))
            payload = handle.read(int(entry["length"]))
        record = json.loads(payload)
        if record.get("module_hash") != module_hash:
            raise ValueError(f"NAS-Bench-101 index mismatch for {module_hash}")
        return record

    def _architecture(self, record: Mapping[str, Any]) -> Architecture:
        architecture = self._space.canonicalize(record["specification"])
        if architecture.architecture_id != record.get("module_hash", architecture.architecture_id):
            raise ValueError("NAS-Bench-101 official module hash does not match specification")
        return Architecture(
            self.search_space_id,
            architecture.architecture_id,
            architecture.spec,
            int(record["benchmark_index"]),
        )

    def metadata(self) -> Mapping[str, Any]:
        if self._legacy:
            return self._legacy.metadata()
        return {
            "benchmark_id": self.benchmark_id,
            "search_space_id": self.search_space_id,
            "version": self.version,
            "path": str(self.path),
            "format": self.manifest["format"],
            "architecture_count": len(self._index),
            "source": self.manifest.get("source"),
            "model_fidelity": self._space.model_fidelity,
            "numerical_equivalence": "not_tensorflow_bitwise_equivalent",
        }

    def capabilities(self) -> Mapping[str, Any]:
        if self._legacy:
            return self._legacy.capabilities()
        return {
            "datasets": ["cifar10"],
            "splits": ["train", "valid", "test", "benchmark"],
            "metric_names": [
                "final_accuracy",
                "halfway_accuracy",
                "final_training_time",
                "halfway_training_time",
                "trainable_parameters",
                "total_time",
            ],
            "epoch_budgets": [4, 12, 36, 108],
            "seeds": [0, 1, 2],
            "model_building": True,
            "safe_runtime_format": True,
        }

    def iter_architectures(self, start: int = 0, end: int | None = None) -> Iterable[Architecture]:
        if self._legacy:
            yield from self._legacy.iter_architectures(start, end)
            return
        stop = len(self._index) if end is None else min(end, len(self._index))
        if start < 0 or start > stop:
            raise ValueError("Invalid NAS-Bench-101 architecture range")
        for entry in self._index[start:stop]:
            yield self._architecture(self._record(str(entry["module_hash"])))

    def sample_architecture(self, seed: int | None = None) -> Architecture:
        if self._legacy:
            return self._legacy.sample_architecture(seed)
        entry = random.Random(seed).choice(self._index)
        return self._architecture(self._record(str(entry["module_hash"])))

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        return self._canonical_architecture(specification).spec

    def architecture_id(self, specification: Any) -> str:
        return self._canonical_architecture(specification).architecture_id

    def _canonical_architecture(self, specification: Any) -> Architecture:
        if isinstance(specification, Architecture):
            if specification.search_space_id != self.search_space_id:
                raise ValueError("NAS-Bench-101 architecture belongs to a different search space")
            specification = specification.spec
        return self._space.canonicalize(specification)

    def is_valid(self, specification: Any) -> bool:
        try:
            return self.architecture_id(specification) in (
                self._by_hash if not self._legacy else {item.architecture_id for item in self._legacy.iter_architectures()}
            )
        except (TypeError, ValueError):
            return False

    def hash_spec(self, specification: Any) -> str:
        return self.architecture_id(specification)

    def build_model(self, architecture: Architecture, dataset: str) -> Any:
        if architecture.search_space_id != self.search_space_id:
            raise ValueError("NAS-Bench-101 architecture belongs to a different search space")
        return self._model_builder(architecture, dataset)

    def get_metrics_from_spec(self, specification: Any) -> tuple[Mapping[str, Any], Mapping[int, list[Mapping[str, Any]]]]:
        if self._legacy:
            raise NotImplementedError("Native metric view requires converted NAS-Bench-101 data")
        module_hash = self.architecture_id(specification)
        record = self._record(module_hash)
        fixed = {
            "module_adjacency": record["specification"]["matrix"],
            "module_operations": record["specification"]["operations"],
        }
        computed: dict[int, list[dict[str, Any]]] = {}
        for metric in record.get("metrics", []):
            budget = int(metric["epoch_budget"])
            seed = int(metric["seed"])
            while len(computed.setdefault(budget, [])) <= seed:
                computed[budget].append({})
            key = f"{metric['split']}_{metric['metric_name']}"
            computed[budget][seed][key] = metric["value"]
            if metric["metric_name"] == "trainable_parameters":
                fixed["trainable_parameters"] = metric["value"]
        return fixed, computed

    def _metric_values(self, record: Mapping[str, Any], metric: MetricSpec) -> list[float]:
        values = []
        matched_budgets: set[int] = set()
        for item in record.get("metrics", []):
            if item.get("dataset") != metric.dataset or item.get("split") != metric.split:
                continue
            if item.get("metric_name") != metric.metric_name:
                continue
            if metric.epoch_budget is not None and item.get("epoch_budget") != metric.epoch_budget:
                continue
            if metric.seed is not None and item.get("seed") != metric.seed:
                continue
            if item.get("epoch_budget") is not None:
                matched_budgets.add(int(item["epoch_budget"]))
            values.append(float(item["value"]))
        if metric.epoch_budget is None and len(matched_budgets) > 1:
            raise ValueError(
                "NAS-Bench-101 metric matches multiple epoch budgets; specify epoch_budget"
            )
        return values

    def query_metrics(self, architecture: Architecture, metric: MetricSpec) -> Mapping[str, float]:
        if self._legacy:
            return self._legacy.query_metrics(architecture, metric)
        self.validate_metric(metric)
        values = self._metric_values(self._record(architecture.architecture_id), metric)
        if not values:
            raise KeyError(f"NAS-Bench-101 metric not found: {metric}")
        if metric.seed is not None:
            if len(values) != 1:
                raise ValueError("NAS-Bench-101 repeat index matched multiple values")
            value = values[0]
        elif metric.seed_reduction == "mean":
            value = sum(values) / len(values)
        elif metric.seed_reduction == "min":
            value = min(values)
        elif metric.seed_reduction == "max":
            value = max(values)
        else:
            raise ValueError(f"Unsupported seed reduction: {metric.seed_reduction}")
        return {metric.metric_name: value}

    def query_native(
        self,
        specification: Any,
        *,
        epochs: int = 108,
        repeat_index: int | None = None,
        seed_reduction: str = "mean",
        stop_halfway: bool = False,
    ) -> Mapping[str, float]:
        architecture = self._canonical_architecture(specification)
        point = "halfway" if stop_halfway else "final"
        result: dict[str, float] = {}
        for split, native_name in (("train", "train_accuracy"), ("valid", "validation_accuracy"), ("test", "test_accuracy")):
            value = self.query_metrics(
                architecture,
                MetricSpec("cifar10", split, f"{point}_accuracy", epochs, repeat_index, seed_reduction),
            )
            result[native_name] = value[f"{point}_accuracy"]
        time_value = self.query_metrics(
            architecture,
            MetricSpec("cifar10", "benchmark", f"{point}_training_time", epochs, repeat_index, seed_reduction),
        )
        result["training_time"] = time_value[f"{point}_training_time"]
        parameter_value = self.query_metrics(
            architecture,
            MetricSpec(
                "cifar10",
                "benchmark",
                "trainable_parameters",
                epochs,
                repeat_index,
                seed_reduction,
            ),
        )
        result["trainable_parameters"] = parameter_value["trainable_parameters"]
        return result


__all__ = ["NasBench101Adapter"]
