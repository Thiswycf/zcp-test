from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from zcp_test.reporting.analysis import correlation_table, read_scores
from zcp_test.types import Architecture, MetricSpec


def _finite_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return None
    result = (
        stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic
        if method == "spearman"
        else stats.kendalltau(paired.iloc[:, 0], paired.iloc[:, 1], variant="b").statistic
    )
    return float(result) if np.isfinite(result) else None


def _top_k_set(values: pd.Series, k: int) -> set[str]:
    return set(values.nlargest(min(k, len(values))).index.astype(str))


def _rank_stability(targets: pd.DataFrame, top_k: Iterable[int]) -> pd.DataFrame:
    pivot = targets.pivot(index="architecture_id", columns="epoch_budget", values="target_value")
    records: list[dict[str, Any]] = []
    budgets = sorted(int(value) for value in pivot.columns)
    for left_index, left_budget in enumerate(budgets):
        for right_budget in budgets[left_index + 1 :]:
            paired = pivot[[left_budget, right_budget]].dropna()
            record: dict[str, Any] = {
                "left_budget": left_budget,
                "right_budget": right_budget,
                "sample_count": len(paired),
                "spearman": _finite_correlation(paired[left_budget], paired[right_budget], "spearman"),
                "kendall_tau_b": _finite_correlation(
                    paired[left_budget], paired[right_budget], "kendall_tau_b"
                ),
            }
            for value in top_k:
                left = _top_k_set(paired[left_budget], value)
                right = _top_k_set(paired[right_budget], value)
                union = left | right
                record[f"top_{value}_jaccard"] = len(left & right) / len(union) if union else None
            records.append(record)
    return pd.DataFrame.from_records(records)


def nasbench101_budget_study(
    source: Any,
    adapter: Any,
    *,
    budgets: Iterable[int] = (4, 12, 36, 108),
    dataset: str = "cifar10",
    split: str = "valid",
    metric_name: str = "final_accuracy",
    repeat_index: int | None = None,
    seed_reduction: str = "mean",
    target_direction: str = "maximize",
    bootstrap_samples: int = 0,
    top_k: Iterable[int] = (5, 10, 50),
) -> dict[str, pd.DataFrame]:
    frame = read_scores(source)
    required = {"architecture_id", "architecture", "proxy_id", "component", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"NAS-Bench-101 budget study requires score fields: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("NAS-Bench-101 budget study requires non-empty scores")
    selected_budgets = tuple(dict.fromkeys(int(value) for value in budgets))
    supported = set(adapter.capabilities().get("epoch_budgets", ()))
    if not selected_budgets or any(value not in supported for value in selected_budgets):
        raise ValueError(
            f"Unsupported NAS-Bench-101 budgets {selected_budgets}; available: {sorted(supported)}"
        )
    if seed_reduction not in {"mean", "min", "max"}:
        raise ValueError("seed_reduction must be mean, min, or max")
    if target_direction not in {"maximize", "minimize"}:
        raise ValueError("target_direction must be maximize or minimize")

    score_groups = ["architecture_id", "proxy_id", "component", "direction"]
    scores = (
        frame.dropna(subset=["architecture_id", "proxy_id", "score"])
        .groupby(score_groups, dropna=False, as_index=False)
        .agg(score=("score", "mean"), observation_count=("score", "size"))
    )
    specifications = (
        frame.dropna(subset=["architecture_id"])
        .drop_duplicates("architecture_id")
        .set_index("architecture_id")["architecture"]
    )
    target_rows: list[dict[str, Any]] = []
    for architecture_id, specification in specifications.items():
        canonical = adapter.canonicalize(specification)
        expected_id = adapter.architecture_id(canonical)
        if str(architecture_id) != expected_id:
            raise ValueError(
                f"Score architecture ID {architecture_id} does not match NAS-Bench-101 hash {expected_id}"
            )
        architecture = Architecture(adapter.search_space_id, expected_id, canonical)
        for budget in selected_budgets:
            metric = MetricSpec(
                dataset,
                split,
                metric_name,
                budget,
                repeat_index,
                seed_reduction,
                adapter.metadata().get("version"),
            )
            value = float(adapter.query_metrics(architecture, metric)[metric_name])
            target_rows.append(
                {
                    "architecture_id": expected_id,
                    "epoch_budget": budget,
                    "target_value": -value if target_direction == "minimize" else value,
                    "raw_target_value": value,
                    "target_direction": target_direction,
                    "dataset": dataset,
                    "target_split": split,
                    "target_metric": metric_name,
                    "seed": repeat_index,
                    "seed_reduction": seed_reduction,
                }
            )
    targets = pd.DataFrame.from_records(target_rows)
    detailed = scores.merge(targets, on="architecture_id", validate="many_to_many")
    correlations = correlation_table(
        detailed,
        group_by=("proxy_id", "component", "epoch_budget"),
        bootstrap_samples=bootstrap_samples,
    )
    stability = _rank_stability(targets, top_k)
    return {"detailed": detailed, "correlations": correlations, "rank_stability": stability}


__all__ = ["nasbench101_budget_study"]
