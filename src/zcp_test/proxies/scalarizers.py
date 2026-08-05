from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ScalarizedScores:
    values: tuple[float, ...]
    selector: str
    scalarizer_id: str
    scalarizer_version: str
    cohort_digest: str | None = None
    cohort_size: int | None = None
    tie_method: str | None = None


def cohort_digest(architecture_ids: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(str(value) for value in architecture_ids), separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_pointwise(row: Mapping[str, object], selector: str = "primary") -> float:
    if selector == "primary":
        value = row.get("score")
    elif selector.startswith("component:"):
        name = selector.split(":", 1)[1]
        if not name:
            raise ValueError("component selector requires a component name")
        components = row.get("components")
        if not isinstance(components, Mapping) or name not in components:
            raise ValueError(f"score row is missing component {name!r}")
        value = components[name]
    else:
        raise ValueError(f"selector {selector!r} is not pointwise")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError("selected score is not finite")
    return result


def aggregate_rank(
    rows: Sequence[Mapping[str, float]],
    component_names: Sequence[str],
    *,
    method: str,
    architecture_ids: Sequence[str],
) -> ScalarizedScores:
    import scipy.stats

    if not rows or len(rows) != len(architecture_ids):
        raise ValueError("rank aggregation requires one architecture ID per non-empty row")
    names = tuple(component_names)
    if not names:
        raise ValueError("rank aggregation requires at least one component")
    count = len(rows)
    totals = [0.0] * count
    for name in names:
        values = [float(row[name]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"component {name!r} contains a non-finite value")
        ranks = scipy.stats.rankdata(values, method="average")
        for index, rank in enumerate(ranks):
            percentile = float(rank) / count
            if method == "az_nas_log_rank":
                totals[index] += math.log(percentile)
            elif method == "mean_percentile_rank":
                totals[index] += percentile / len(names)
            else:
                raise ValueError(f"unknown rank aggregator {method!r}")
    return ScalarizedScores(
        values=tuple(totals),
        selector=f"aggregate:{method}",
        scalarizer_id=method,
        scalarizer_version="cohort-average-ties-v1",
        cohort_digest=cohort_digest(architecture_ids),
        cohort_size=count,
        tie_method="average",
    )
