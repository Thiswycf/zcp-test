from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping
import sys
import types
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder
from zcp_test.benchmarks.common import ensure_path, require_architecture_space
from zcp_test.spaces.darts import DartsSpace, canonicalize_genotype
from zcp_test.types import Architecture, MetricSpec


def _import_inference_only_nb301():
    """Import the official API without its unused PyG training stack."""
    placeholders = (
        ("nasbench301.surrogate_models.bananas.bananas", "BANANASModel"),
        ("nasbench301.surrogate_models.gnn.gnn", "GNNSurrogateModel"),
    )
    for module_name, class_name in placeholders:
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            setattr(module, class_name, type(class_name, (), {}))
            sys.modules[module_name] = module
    if "IPython" not in sys.modules:
        module = types.ModuleType("IPython")
        module.embed = lambda *args, **kwargs: None
        module.get_ipython = lambda: None
        module.version_info = (0, 0)
        sys.modules["IPython"] = module
    if "pathvalidate" not in sys.modules:
        module = types.ModuleType("pathvalidate")
        module.sanitize_filename = lambda value: value
        sys.modules["pathvalidate"] = module
    import nasbench301

    return nasbench301


class NasBench301SurrogateAdapter(JsonlBenchmarkAdapter):
    benchmark_id = "nasbench301_surrogate"
    search_space_id = "darts"

    def __init__(
        self,
        path: str,
        *,
        architecture_path: str | None = None,
        runtime_path: str | None = None,
        version: str = "1.0",
        ensemble_loader=None,
        model_builder=None,
    ) -> None:
        self.ensemble_path = ensure_path(path, kind="NAS-Bench-301 ensemble")
        self.runtime_ensemble_path = (
            ensure_path(runtime_path, kind="NAS-Bench-301 runtime ensemble")
            if runtime_path
            else None
        )
        self._ensemble_loader = ensemble_loader
        self._ensembles: dict[str, Any] = {}
        self._space = DartsSpace()
        self.version = version
        self._model_builder = model_builder or default_model_builder
        self._generated = architecture_path is None
        if architecture_path is not None:
            super().__init__(
                architecture_path,
                benchmark_id=self.benchmark_id,
                search_space_id=self.search_space_id,
                version=version,
                model_builder=self._model_builder,
            )

    def _load_ensemble(self, kind: str):
        if kind not in self._ensembles:
            path = self.ensemble_path if kind == "accuracy" else self.runtime_ensemble_path
            if path is None:
                raise ValueError(
                    "NAS-Bench-301 runtime queries require a separate runtime_path ensemble"
                )
            if self._ensemble_loader:
                self._ensembles[kind] = self._ensemble_loader(path)
            else:
                try:
                    nasbench301 = _import_inference_only_nb301()
                except ImportError as error:
                    raise RuntimeError("Install the nb301 extra to query the surrogate") from error
                self._ensembles[kind] = nasbench301.load_ensemble(str(path))
        return self._ensembles[kind]

    @property
    def ensemble(self):
        return self._load_ensemble("accuracy")

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if not isinstance(specification, Mapping):
            raise TypeError("DARTS genotype must be a mapping")
        required = {"normal", "normal_concat", "reduce", "reduce_concat"}
        if not required.issubset(specification):
            raise ValueError(f"DARTS genotype is missing: {sorted(required - set(specification))}")
        return canonicalize_genotype(specification)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "search_space_id": self.search_space_id,
            "version": self.version,
            "surrogate_path": str(self.ensemble_path),
            "runtime_surrogate_path": str(self.runtime_ensemble_path)
            if self.runtime_ensemble_path
            else None,
            "ground_truth_kind": "surrogate_prediction",
            "architecture_source": "deterministic_darts_sampling"
            if self._generated
            else "registered_jsonl",
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

    def iter_architectures(self, start: int = 0, end: int | None = None):
        if not self._generated:
            yield from super().iter_architectures(start, end)
            return
        if start < 0 or end is None or end < start:
            raise ValueError("Generated NAS-Bench-301 iteration requires a finite valid range")
        for seed in range(start, end):
            yield self._space.sample(seed)

    def sample_architecture(self, seed: int | None = None):
        if not self._generated:
            return super().sample_architecture(seed)
        return self._space.sample(seed)

    def architecture_id(self, specification: Any) -> str:
        return self._space.canonicalize(self.canonicalize(specification)).architecture_id

    def build_model(self, architecture: Architecture, dataset: str) -> Any:
        require_architecture_space(architecture, self.search_space_id)
        return self._model_builder(architecture, dataset)

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
        ensemble = self._load_ensemble("runtime" if metric.metric_name == "runtime" else "accuracy")
        value = ensemble.predict(
            config=genotype,
            representation="genotype",
            with_noise=metric.surrogate_noise,
        )
        return {metric.metric_name: float(value)}
