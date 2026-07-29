from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zcp_test.benchmarks.base import BenchmarkAdapter
from zcp_test.benchmarks.common import (
    ModelBuilder,
    architecture_from_spec,
    bounded_range,
    require_architecture_space,
    sample_index,
)
from zcp_test.data.jsonl import JsonlTable
from zcp_test.types import Architecture, MetricSpec


class JsonlBenchmarkAdapter(BenchmarkAdapter):
    """Runtime adapter for normalized, safe benchmark JSONL files."""

    def __init__(
        self,
        path: str,
        *,
        benchmark_id: str,
        search_space_id: str,
        version: str | None = None,
        model_builder: ModelBuilder | None = None,
        required_protocol: str | None = None,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.search_space_id = search_space_id
        self.path = JsonlTable(path, expected_kind="benchmark_architecture").path
        self._records = JsonlTable(self.path, expected_kind="benchmark_architecture").load()
        self._model_builder = model_builder
        self._required_protocol = required_protocol
        self._validate_records(version)
        self.version = version or str(self._records[0]["benchmark_version"])
        self._architectures = [self._to_architecture(record) for record in self._records]
        self._by_id = {architecture.architecture_id: record for architecture, record in zip(
            self._architectures, self._records, strict=True
        )}

    def _validate_records(self, requested_version: str | None) -> None:
        if not self._records:
            raise ValueError(f"Benchmark JSONL is empty: {self.path}")
        indices: set[int] = set()
        for line_index, record in enumerate(self._records):
            if record.get("benchmark_id") != self.benchmark_id:
                raise ValueError(f"benchmark_id mismatch in record {line_index}")
            if record.get("search_space_id") != self.search_space_id:
                raise ValueError(f"search_space_id mismatch in record {line_index}")
            if not isinstance(record.get("specification"), dict):
                raise ValueError(f"Missing specification in record {line_index}")
            if not isinstance(record.get("metrics", []), list):
                raise ValueError(f"metrics must be a list in record {line_index}")
            version = str(record.get("benchmark_version", ""))
            if requested_version and version != requested_version:
                raise ValueError(f"Expected version {requested_version!r}, found {version!r}")
            index = int(record.get("benchmark_index", line_index))
            if index in indices:
                raise ValueError(f"Duplicate benchmark_index {index}")
            indices.add(index)
            if self._required_protocol and record.get("protocol") != self._required_protocol:
                raise ValueError(
                    f"Expected protocol {self._required_protocol!r}, found {record.get('protocol')!r}"
                )

    def _to_architecture(self, record: Mapping[str, Any]) -> Architecture:
        return architecture_from_spec(
            self.search_space_id,
            self.canonicalize(record["specification"]),
            int(record.get("benchmark_index", 0)),
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "search_space_id": self.search_space_id,
            "version": self.version,
            "path": str(self.path),
            "format": "jsonl",
            "architecture_count": len(self._architectures),
        }

    def capabilities(self) -> Mapping[str, Any]:
        metrics = [metric for record in self._records for metric in record.get("metrics", [])]
        return {
            "datasets": sorted({metric["dataset"] for metric in metrics}),
            "splits": sorted({metric["split"] for metric in metrics}),
            "metric_names": sorted({metric["metric_name"] for metric in metrics}),
            "epoch_budgets": sorted(
                {metric["epoch_budget"] for metric in metrics if metric.get("epoch_budget") is not None}
            ),
            "model_building": self._model_builder is not None,
            "safe_runtime_format": True,
        }

    def iter_architectures(self, start: int = 0, end: int | None = None) -> Iterable[Architecture]:
        for index in bounded_range(len(self._architectures), start, end):
            yield self._architectures[index]

    def sample_architecture(self, seed: int | None = None) -> Architecture:
        return self._architectures[sample_index(len(self._architectures), seed)]

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if not isinstance(specification, Mapping):
            raise TypeError("Architecture specification must be a mapping")
        return dict(specification)

    def architecture_id(self, specification: Any) -> str:
        return architecture_from_spec(self.search_space_id, self.canonicalize(specification)).architecture_id

    def build_model(self, architecture: Architecture, dataset: str) -> Any:
        require_architecture_space(architecture, self.search_space_id)
        if self._model_builder is None:
            raise NotImplementedError(f"{self.benchmark_id} has no configured model builder")
        return self._model_builder(architecture, dataset)

    def query_metrics(self, architecture: Architecture, metric: MetricSpec) -> Mapping[str, float]:
        require_architecture_space(architecture, self.search_space_id)
        self.validate_metric(metric)
        try:
            record = self._by_id[architecture.architecture_id]
        except KeyError as error:
            raise KeyError(f"Architecture is not present in {self.benchmark_id}") from error
        matches = []
        for candidate in record.get("metrics", []):
            if candidate.get("dataset") != metric.dataset:
                continue
            if candidate.get("split") != metric.split:
                continue
            if candidate.get("metric_name") != metric.metric_name:
                continue
            if metric.epoch_budget is not None and candidate.get("epoch_budget") != metric.epoch_budget:
                continue
            if metric.seed is not None and candidate.get("seed") != metric.seed:
                continue
            matches.append(candidate)
        if not matches:
            raise KeyError(f"No metric matching explicit protocol: {metric}")
        matched_budgets = {
            int(candidate["epoch_budget"])
            for candidate in matches
            if candidate.get("epoch_budget") is not None
        }
        if metric.epoch_budget is None and len(matched_budgets) > 1:
            raise ValueError("Metric matches multiple epoch budgets; specify epoch_budget")
        values = [float(candidate["value"]) for candidate in matches]
        if metric.seed is None and len(values) > 1:
            if metric.seed_reduction == "mean":
                value = sum(values) / len(values)
            elif metric.seed_reduction == "min":
                value = min(values)
            elif metric.seed_reduction == "max":
                value = max(values)
            else:
                raise ValueError(f"Unsupported seed reduction: {metric.seed_reduction}")
        elif len(values) == 1:
            value = values[0]
        else:
            raise ValueError("Multiple metric rows matched an explicitly requested seed")
        return {metric.metric_name: value}
