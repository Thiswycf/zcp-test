from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping


class ScoreDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class RecordStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"


class ModelFidelity(str, Enum):
    REFERENCE_MODEL = "reference_model"
    REFERENCE_TOPOLOGY_PYTORCH_PORT = "reference_topology_pytorch_port"
    SURROGATE_PREDICTION = "surrogate_prediction"
    INHERITED_SUPERNET = "inherited_supernet"
    PROXY_APPROXIMATION = "proxy_approximation"
    METRIC_ONLY = "metric_only"


@dataclass(frozen=True)
class Architecture:
    search_space_id: str
    architecture_id: str
    spec: Mapping[str, Any]
    benchmark_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSpec:
    dataset: str
    split: str
    metric_name: str
    epoch_budget: int | None = None
    seed: int | None = None
    seed_reduction: str = "mean"
    benchmark_version: str | None = None
    surrogate_noise: bool = False


@dataclass(frozen=True)
class ProxyCapability:
    proxy_id: str
    version: str = "1"
    model_families: tuple[str, ...] = ("cnn",)
    requires_data: bool = True
    requires_inputs: bool = True
    requires_labels: bool = False
    requires_loss_fn: bool = False
    supports_cpu: bool = True
    direction: ScoreDirection = ScoreDirection.MAXIMIZE
    components: tuple[str, ...] = ("score",)
    primary_component: str = "score"
    dependencies: tuple[str, ...] = ()
    implementation_fidelity: str = "unverified"
    source: str | None = None
    alias_of: str | None = None
    resource_direction: ScoreDirection | None = None
    source_commit: str | None = None
    license: str | None = None
    official_code_available: bool | None = None
    protocol_domain: str | None = None
    default_batches: int = 1
    default_repetitions: int = 1
    requires_edge_activations: bool = False
    requires_topology: bool = False
    formal_use: str = "proxy_score"


@dataclass(frozen=True)
class ProxyContext:
    inputs: Any = None
    labels: Any = None
    loss_fn: Any = None
    seed: int = 0
    device: str | None = None
    model_family: str = "cnn"
    benchmark_id: str | None = None
    search_space_id: str | None = None
    input_fingerprint: str | None = None
    batch_fingerprints: tuple[str, ...] = ()
    proxy_batches: int = 1
    proxy_repetitions: int = 1
    batch_provider: Callable[[], Iterator[tuple[Any, Any | None]]] | None = None
    edge_activations: Any = None


@dataclass(frozen=True)
class ProxyOutput:
    score: float
    primary_component: str = "score"
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass
class ScoreResult:
    score: float | None = None
    primary_component: str = "score"
    components: dict[str, float] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.OK
    error_type: str | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    peak_memory_mb: float | None = None
    proxy_version: str | None = None
    direction: ScoreDirection = ScoreDirection.MAXIMIZE
    implementation_fidelity: str = "unverified"
    source: str | None = None
    alias_of: str | None = None
    resource_direction: ScoreDirection | None = None
    source_commit: str | None = None
    license: str | None = None
    protocol_domain: str | None = None
    formal_use: str | None = None

    @property
    def values(self) -> dict[str, float]:
        """Compatibility view for callers written against schema 1.x."""
        return self.components
