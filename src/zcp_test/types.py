from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ScoreDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class RecordStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"


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
    requires_labels: bool = False
    supports_cpu: bool = True
    direction: ScoreDirection = ScoreDirection.MAXIMIZE
    components: tuple[str, ...] = ("score",)
    primary_component: str = "score"
    dependencies: tuple[str, ...] = ()


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

    @property
    def values(self) -> dict[str, float]:
        """Compatibility view for callers written against schema 1.x."""
        return self.components
