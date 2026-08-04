from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zcp_test.benchmarks.native import IndexedNativeAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder
from zcp_test.types import Architecture, MetricSpec


class NatsAdapter(IndexedNativeAdapter):
    api_type: str
    supported_budgets: tuple[int, ...]

    def __init__(self, path: str, *, version: str, api_factory=None, model_builder=None) -> None:
        super().__init__(path, version=version, api_factory=api_factory, model_builder=model_builder or default_model_builder)

    def _default_api_factory(self, path: Path) -> Any:
        try:
            from nats_bench import create
        except ImportError as error:
            raise RuntimeError("NATS-Bench adapter requires nats-bench") from error
        return create(str(path), self.api_type, fast_mode=True, verbose=False)

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "datasets": ["cifar10-valid", "cifar10", "cifar100", "ImageNet16-120"],
            "splits": ["train", "valid", "test"],
            "metric_names": ["accuracy", "loss", "time", "latency"],
            "epoch_budgets": list(self.supported_budgets),
            "seed_reductions": ["mean", "min", "max"],
            "supports_explicit_seed": True,
            "model_building": True,
            "benchmark_source": f"nats_{self.api_type}",
        }

    def query_metrics(self, architecture: Architecture, metric: MetricSpec) -> Mapping[str, float]:
        self.validate_metric(metric)
        index = self._required_index(architecture)
        reduction = metric.seed_reduction
        if reduction not in {"mean", "min", "max"}:
            raise ValueError(f"Unsupported NATS seed reduction: {reduction!r}")
        hp = str(metric.epoch_budget or self.default_epoch_budget)
        if metric.seed is not None or reduction == "mean":
            random_seed: bool | int = metric.seed if metric.seed is not None else False
            info = self.api.get_more_info(
                index,
                metric.dataset,
                iepoch=None,
                hp=hp,
                is_random=random_seed,
            )
            return {metric.metric_name: self._extract_metric(info, metric)}

        query_metadata = getattr(self.api, "query_meta_info_by_index", None)
        if not callable(query_metadata):
            raise RuntimeError(
                f"NATS seed reduction {reduction!r} requires API seed enumeration via "
                "query_meta_info_by_index"
            )
        metadata = query_metadata(index, hp=hp)
        get_dataset_seeds = getattr(metadata, "get_dataset_seeds", None)
        if not callable(get_dataset_seeds):
            raise RuntimeError(
                f"NATS seed reduction {reduction!r} requires metadata.get_dataset_seeds"
            )
        seeds = tuple(get_dataset_seeds(metric.dataset))
        if not seeds:
            raise RuntimeError(
                f"NATS seed reduction {reduction!r} found no official seeds for "
                f"{metric.dataset!r}"
            )
        values = [
            self._extract_metric(
                self.api.get_more_info(
                    index,
                    metric.dataset,
                    iepoch=None,
                    hp=hp,
                    is_random=seed,
                ),
                metric,
            )
            for seed in seeds
        ]
        aggregate = min(values) if reduction == "min" else max(values)
        return {metric.metric_name: aggregate}


class NatsTssAdapter(NatsAdapter):
    benchmark_id = "nats_tss"
    search_space_id = "nb201_topology"
    api_type = "tss"
    default_epoch_budget = 200
    supported_budgets = (12, 200)


class NatsSssAdapter(NatsAdapter):
    benchmark_id = "nats_sss"
    search_space_id = "nats_size"
    api_type = "sss"
    default_epoch_budget = 90
    supported_budgets = (12, 90)
