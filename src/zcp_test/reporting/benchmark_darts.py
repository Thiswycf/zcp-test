from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from zcp_test.reporting.benchmark_studies import (
    _VALID_DIRECTIONS,
    _architecture_value,
    _numeric_feature_correlations,
    _safe_feature_name,
    _study_group_fields,
)
from zcp_test.spaces.darts import canonicalize_genotype


def _specification(value: Any, row_label: Any, architecture_column: str) -> dict[str, Any]:
    value = _architecture_value(value, architecture_column)
    if not isinstance(value, Mapping):
        raise ValueError(f"NAS-Bench-301 row {row_label!r} architecture must be a mapping")
    return canonicalize_genotype(value)


def _feature_record(specification: Mapping[str, Any]) -> dict[str, float | int]:
    record: dict[str, float | int] = {}
    per_cell: dict[str, Counter[str]] = {}
    for cell in ("normal", "reduce"):
        edges = specification[cell]
        operations = Counter(str(edge[0]) for edge in edges)
        per_cell[cell] = operations
        record[f"darts_{cell}_unique_operation_count"] = len(operations)
        record[f"darts_{cell}_input_edge_count"] = sum(int(edge[1]) < 2 for edge in edges)
        record[f"darts_{cell}_intermediate_edge_count"] = sum(int(edge[1]) >= 2 for edge in edges)
        record[f"darts_{cell}_mean_source_index"] = float(
            np.mean([int(edge[1]) for edge in edges])
        )
        spans = [edge_index // 2 + 2 - int(edge[1]) for edge_index, edge in enumerate(edges)]
        record[f"darts_{cell}_mean_edge_span"] = float(np.mean(spans))
        record[f"darts_{cell}_max_edge_span"] = int(max(spans))
        for operation, count in sorted(operations.items()):
            record[f"darts_{cell}_op_count__{_safe_feature_name(operation)}"] = count
        for source in range(5):
            record[f"darts_{cell}_source_count__{source}"] = sum(
                int(edge[1]) == source for edge in edges
            )
    operations = sorted(set(per_cell["normal"]) | set(per_cell["reduce"]))
    for operation in operations:
        name = _safe_feature_name(operation)
        normal = per_cell["normal"].get(operation, 0)
        reduce = per_cell["reduce"].get(operation, 0)
        record[f"darts_total_op_count__{name}"] = normal + reduce
        record[f"darts_cell_balance__{name}"] = normal - reduce
        record[f"darts_cell_interaction__{name}"] = normal * reduce
    return record


def _featured_frame(frame: pd.DataFrame, architecture_column: str) -> pd.DataFrame:
    records = [
        _feature_record(_specification(value, row_label, architecture_column))
        for row_label, value in frame[architecture_column].items()
    ]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def _edge_table(frame: pd.DataFrame, architecture_column: str) -> pd.DataFrame:
    protocol_fields = _study_group_fields(frame, ("proxy_id", "component"))
    records: list[dict[str, Any]] = []
    for row_label, row in frame.iterrows():
        specification = _specification(row[architecture_column], row_label, architecture_column)
        direction = str(row["direction"]).casefold()
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"NAS-Bench-301 has invalid proxy direction {direction!r}")
        score = float(row["score"])
        if direction == "minimize":
            score = -score
        target = float(row["target_value"])
        if str(row.get("target_direction", "maximize")).casefold() == "minimize":
            target = -target
        for cell in ("normal", "reduce"):
            for edge_index, (operation, source) in enumerate(specification[cell]):
                node = edge_index // 2 + 2
                records.append(
                    {
                        **{field: row[field] for field in protocol_fields},
                        "architecture_id": row["architecture_id"],
                        "cell": cell,
                        "node": node,
                        "edge_slot": edge_index % 2,
                        "source": int(source),
                        "source_class": "cell_input" if int(source) < 2 else "intermediate",
                        "edge_span": node - int(source),
                        "operation": str(operation),
                        "target_value": target,
                        "adjusted_score": score,
                    }
                )
    return pd.DataFrame.from_records(records)


def _interaction_effects(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    protocol_fields = [
        field
        for field in _study_group_fields(edges, ("proxy_id", "component"))
        if field in edges
    ]
    groups = [*protocol_fields, "cell", "node", "source_class", "operation"]
    effects = (
        edges.groupby(groups, as_index=False, dropna=False, sort=True)
        .agg(
            sample_count=("architecture_id", "nunique"),
            target_mean=("target_value", "mean"),
            target_median=("target_value", "median"),
            score_mean=("adjusted_score", "mean"),
            score_median=("adjusted_score", "median"),
            mean_edge_span=("edge_span", "mean"),
        )
    )
    baselines = (
        edges.groupby(
            [*protocol_fields, "cell", "node", "source_class"],
            as_index=False,
            dropna=False,
        )
        .agg(
            interaction_target_mean=("target_value", "mean"),
            interaction_score_mean=("adjusted_score", "mean"),
        )
    )
    effects = effects.merge(
        baselines,
        on=[*protocol_fields, "cell", "node", "source_class"],
        validate="many_to_one",
    )
    effects["target_delta_from_interaction_mean"] = (
        effects["target_mean"] - effects["interaction_target_mean"]
    )
    effects["score_delta_from_interaction_mean"] = (
        effects["score_mean"] - effects["interaction_score_mean"]
    )
    return effects


def nasbench301_darts_study(
    frame: pd.DataFrame,
    *,
    architecture_column: str = "architecture",
    methods: Sequence[str] = ("spearman", "kendall_tau_b", "pearson"),
) -> dict[str, pd.DataFrame]:
    """Study operation/topology interactions in NAS-Bench-301 DARTS genotypes."""
    required = {
        architecture_column,
        "architecture_id",
        "proxy_id",
        "component",
        "score",
        "target_value",
        "direction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"NAS-Bench-301 DARTS study requires fields: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("NAS-Bench-301 DARTS study requires non-empty scores")
    protocol_fields = _study_group_fields(frame, ("proxy_id", "component"))
    if frame.duplicated([*protocol_fields, "architecture_id"], keep=False).any():
        raise ValueError(
            "NAS-Bench-301 DARTS study requires one row per protocol/proxy/architecture"
        )
    featured = _featured_frame(frame, architecture_column)
    feature_columns = [column for column in featured if column.startswith("darts_")]
    correlations = _numeric_feature_correlations(
        featured,
        feature_columns=feature_columns,
        study="NAS-Bench-301 DARTS study",
        methods=methods,
    )
    edges = _edge_table(frame, architecture_column)
    interactions = _interaction_effects(edges)
    architectures = featured.drop_duplicates("architecture_id", keep="first")
    architecture_columns = ["architecture_id", architecture_column, *feature_columns]
    return {
        "architectures": architectures[architecture_columns].reset_index(drop=True),
        "edges": edges,
        "correlations": correlations,
        "operation_topology_interactions": interactions,
    }


__all__ = ["nasbench301_darts_study"]
