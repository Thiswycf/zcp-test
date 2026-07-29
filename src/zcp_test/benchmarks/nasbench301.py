from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder
from zcp_test.benchmarks.common import ensure_path, require_architecture_space
from zcp_test.types import Architecture, MetricSpec


class NasBench301SurrogateAdapter(JsonlBenchmarkAdapter):
    benchmark_id = "nasbench301_surrogate"
    search_space_id = "darts"

    def __init__(
        self,
        path: str,
        *,
        architecture_path: str,
        version: str = "1.0",
        ensemble_loader=None,
        model_builder=None,
    ) -> None:
        self.ensemble_path = ensure_path(path, kind="NAS-Bench-301 ensemble")
        self._ensemble_loader = ensemble_loader
        self._ensemble = None
        super().__init__(
            architecture_path,
            benchmark_id=self.benchmark_id,
            search_space_id=self.search_space_id,
            version=version,
            model_builder=model_builder or default_model_builder,
        )

    @property
    def ensemble(self):
        if self._ensemble is None:
            if self._ensemble_loader:
                self._ensemble = self._ensemble_loader(self.ensemble_path)
            else:
                try:
                    import nasbench301
                except ImportError as error:
                    raise RuntimeError("Install the nb301 extra to query the surrogate") from error
                self._ensemble = nasbench301.load_ensemble(str(self.ensemble_path))
        return self._ensemble

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if not isinstance(specification, Mapping):
            raise TypeError("DARTS genotype must be a mapping")
        required = {"normal", "normal_concat", "reduce", "reduce_concat"}
        if not required.issubset(specification):
            raise ValueError(f"DARTS genotype is missing: {sorted(required - set(specification))}")
        result = {key: specification[key] for key in sorted(required)}
        for cell in ("normal", "reduce"):
            if not isinstance(result[cell], list):
                raise ValueError(f"DARTS {cell} edges must be a list")
            result[cell] = [[str(edge[0]), int(edge[1])] for edge in result[cell]]
        for concat in ("normal_concat", "reduce_concat"):
            result[concat] = [int(value) for value in result[concat]]
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            **super().metadata(),
            "surrogate_path": str(self.ensemble_path),
            "ground_truth_kind": "surrogate_prediction",
        }

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "datasets": ["cifar10"],
            "splits": ["test"],
            "metric_names": ["accuracy", "runtime"],
            "epoch_budgets": [],
            "surrogate": True,
            "surrogate_noise_default": False,
            "model_building": self._model_builder is not None,
        }

    def query_metrics(self, architecture: Architecture, metric: MetricSpec) -> Mapping[str, float]:
        require_architecture_space(architecture, self.search_space_id)
        self.validate_metric(metric)
        genotype_type = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
        spec = self.canonicalize(architecture.spec)
        genotype = genotype_type(
            normal=[tuple(edge) for edge in spec["normal"]],
            normal_concat=spec["normal_concat"],
            reduce=[tuple(edge) for edge in spec["reduce"]],
            reduce_concat=spec["reduce_concat"],
        )
        value = self.ensemble.predict(
            config=genotype,
            representation="genotype",
            with_noise=metric.surrogate_noise,
        )
        return {metric.metric_name: float(value)}
