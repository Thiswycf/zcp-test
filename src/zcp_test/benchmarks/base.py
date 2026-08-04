from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Mapping

from zcp_test.types import Architecture, MetricSpec


class BenchmarkAdapter(ABC):
    benchmark_id: str
    search_space_id: str

    @abstractmethod
    def metadata(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def capabilities(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def iter_architectures(self, start: int = 0, end: int | None = None) -> Iterable[Architecture]: ...

    @abstractmethod
    def sample_architecture(self, seed: int | None = None) -> Architecture: ...

    @abstractmethod
    def canonicalize(self, specification: Any) -> Mapping[str, Any]: ...

    @abstractmethod
    def architecture_id(self, specification: Any) -> str: ...

    @abstractmethod
    def build_model(self, architecture: Architecture, dataset: str) -> Any: ...

    @abstractmethod
    def query_metrics(self, architecture: Architecture, metric: MetricSpec) -> Mapping[str, float]: ...

    def validate_metric(self, metric: MetricSpec) -> None:
        """Validate an explicit metric request before querying a benchmark."""
        capabilities = self.capabilities()
        datasets = capabilities.get("datasets")
        if datasets and metric.dataset not in datasets:
            raise ValueError(f"Dataset {metric.dataset!r} is not supported by {self.benchmark_id}")
        splits = capabilities.get("splits")
        if splits and metric.split not in splits:
            raise ValueError(f"Split {metric.split!r} is not supported by {self.benchmark_id}")
        metric_names = capabilities.get("metric_names")
        if metric_names and metric.metric_name not in metric_names:
            raise ValueError(
                f"Metric {metric.metric_name!r} is not supported by {self.benchmark_id}"
            )
        budgets = capabilities.get("epoch_budgets")
        if budgets and metric.epoch_budget is not None and metric.epoch_budget not in budgets:
            raise ValueError(
                f"Epoch budget {metric.epoch_budget!r} is not supported by {self.benchmark_id}"
            )
        reductions = capabilities.get("seed_reductions")
        if (
            reductions
            and metric.seed is None
            and metric.seed_reduction not in reductions
        ):
            raise ValueError(
                f"Seed reduction {metric.seed_reduction!r} is not supported by "
                f"{self.benchmark_id}"
            )
        version = self.metadata().get("version")
        if metric.benchmark_version and version and metric.benchmark_version != version:
            raise ValueError(
                f"Requested benchmark version {metric.benchmark_version!r}, loaded {version!r}"
            )
