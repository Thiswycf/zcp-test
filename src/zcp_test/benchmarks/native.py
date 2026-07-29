from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from zcp_test.benchmarks.base import BenchmarkAdapter
from zcp_test.benchmarks.common import ModelBuilder, architecture_from_spec, bounded_range, ensure_path, require_architecture_space, sample_index
from zcp_test.types import Architecture, MetricSpec

ApiFactory = Callable[[Path], Any]


class IndexedNativeAdapter(BenchmarkAdapter):
    default_epoch_budget: int | None = None

    def __init__(self, path: str, *, version: str, api_factory: ApiFactory | None = None, model_builder: ModelBuilder | None = None) -> None:
        self.path = ensure_path(path, kind=f"{self.benchmark_id} data")
        self.version = version
        self._api_factory = api_factory
        self._model_builder = model_builder
        self._api_instance: Any | None = None

    @property
    def api(self) -> Any:
        if self._api_instance is None:
            self._api_instance = (self._api_factory or self._default_api_factory)(self.path)
        return self._api_instance

    def _default_api_factory(self, path: Path) -> Any:
        raise NotImplementedError

    def _architecture_spec(self, index: int) -> Mapping[str, Any]:
        return self.canonicalize({"architecture": str(self.api[index])})

    def metadata(self) -> Mapping[str, Any]:
        return {"benchmark_id": self.benchmark_id, "search_space_id": self.search_space_id, "version": self.version, "path": str(self.path), "format": "native_api"}

    def iter_architectures(self, start: int = 0, end: int | None = None) -> Iterable[Architecture]:
        for index in bounded_range(len(self.api), start, end):
            yield architecture_from_spec(self.search_space_id, self._architecture_spec(index), index)

    def sample_architecture(self, seed: int | None = None) -> Architecture:
        index = sample_index(len(self.api), seed)
        return architecture_from_spec(self.search_space_id, self._architecture_spec(index), index)

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if isinstance(specification, str):
            specification = {"architecture": specification}
        if not isinstance(specification, Mapping) or not isinstance(specification.get("architecture"), str):
            raise ValueError("Architecture specification must contain an architecture string")
        return {"architecture": specification["architecture"].strip()}

    def architecture_id(self, specification: Any) -> str:
        return architecture_from_spec(self.search_space_id, self.canonicalize(specification)).architecture_id

    def build_model(self, architecture: Architecture, dataset: str) -> Any:
        require_architecture_space(architecture, self.search_space_id)
        if self._model_builder:
            return self._model_builder(architecture, dataset)
        config = self.api.get_net_config(self._required_index(architecture), dataset)
        try:
            from xautodl.models import get_cell_based_tiny_net
        except ImportError as error:
            raise RuntimeError("Model construction requires xautodl or a model_builder") from error
        return get_cell_based_tiny_net(config)

    def _required_index(self, architecture: Architecture) -> int:
        require_architecture_space(architecture, self.search_space_id)
        if architecture.benchmark_index is None:
            raise ValueError("Native benchmark query requires benchmark_index")
        if architecture.benchmark_index < 0 or architecture.benchmark_index >= len(self.api):
            raise IndexError(f"Invalid benchmark index: {architecture.benchmark_index}")
        expected = self.architecture_id(self._architecture_spec(architecture.benchmark_index))
        if architecture.architecture_id != expected:
            raise ValueError("Architecture ID does not match its benchmark index")
        return architecture.benchmark_index

    @staticmethod
    def _extract_metric(info: Mapping[str, Any], metric: MetricSpec) -> float:
        aliases = {
            "accuracy": [f"{metric.split}-accuracy", f"{metric.split}_accuracy", "accuracy"],
            "loss": [f"{metric.split}-loss", f"{metric.split}_loss", "loss"],
            "time": [f"{metric.split}-all-time", f"{metric.split}-per-time", "time", "latency"],
            "latency": ["latency", "time"],
        }
        for key in [metric.metric_name, *aliases.get(metric.metric_name, [])]:
            if key in info:
                return float(info[key])
        raise KeyError(f"Metric {metric.metric_name!r} for split {metric.split!r} not in {sorted(info)}")
