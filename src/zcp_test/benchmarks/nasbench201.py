from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zcp_test.benchmarks.native import IndexedNativeAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder
from zcp_test.types import Architecture, MetricSpec


class NasBench201Adapter(IndexedNativeAdapter):
    benchmark_id = "nasbench201"
    search_space_id = "nb201_topology"
    default_epoch_budget = 200

    def __init__(self, path: str, *, version: str, api_factory=None, model_builder=None) -> None:
        super().__init__(path, version=version, api_factory=api_factory, model_builder=model_builder or default_model_builder)

    def _default_api_factory(self, path: Path) -> Any:
        try:
            from zcp_test.vendor.nas_201_api import NASBench201API
        except ImportError as error:
            raise RuntimeError("Install NAS-Bench-201 API or pass api_factory") from error
        return NASBench201API(str(path), verbose=False)

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "datasets": ["cifar10-valid", "cifar10", "cifar100", "ImageNet16-120"],
            "splits": ["train", "valid", "test"],
            "metric_names": ["accuracy", "loss", "time"],
            "epoch_budgets": [12, 200],
            "seed_reductions": ["mean"],
            "supports_explicit_seed": True,
            "model_building": True,
            "benchmark_source": "nasbench201",
        }

    def _architecture_spec(self, index: int) -> Mapping[str, Any]:
        value = self.api.arch(index) if hasattr(self.api, "arch") else self.api[index]
        return self.canonicalize(str(value))

    def query_metrics(self, architecture: Architecture, metric: MetricSpec) -> Mapping[str, float]:
        self.validate_metric(metric)
        index = self._required_index(architecture)
        random_seed: bool | int = metric.seed if metric.seed is not None else False
        info = self.api.get_more_info(
            index,
            metric.dataset,
            iepoch=None,
            hp=str(metric.epoch_budget or self.default_epoch_budget),
            is_random=random_seed,
        )
        return {metric.metric_name: self._extract_metric(info, metric)}
