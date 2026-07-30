from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from zcp_test.reporting.analysis import correlation_table, read_scores
from zcp_test.types import Architecture, MetricSpec


_SUCCESS_STATUSES = {"ok", "success", "completed"}
_SCORE_PROTOCOL_FIELDS = (
    "benchmark_id",
    "benchmark_version",
    "benchmark_variant",
    "benchmark_protocol",
    "search_space_id",
    "proxy_version",
    "input_source",
    "input_fingerprint",
    "model_fidelity",
    "seed",
)
_TARGET_PROTOCOL_FIELDS = (
    "dataset",
    "target_split",
    "target_metric",
    "target_direction",
    "target_seed",
    "target_seed_reduction",
)
_STRUCTURE_FEATURES = (
    "vertices",
    "edges",
    "longest_path_depth",
    "conv3_count",
    "conv1_count",
    "maxpool_count",
)


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


def _available_fields(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [field for field in candidates if field in frame and frame[field].notna().any()]


def _top_k_set(values: pd.Series, k: int) -> set[str]:
    table = pd.DataFrame(
        {"architecture_id": values.index.astype(str), "value": pd.to_numeric(values, errors="coerce")}
    ).reset_index(drop=True).dropna(subset=["value"])
    table = table.sort_values(
        ["value", "architecture_id"], ascending=[False, True], kind="mergesort"
    )
    return set(table.head(min(k, len(table)))["architecture_id"])


def _rank_stability(targets: pd.DataFrame, top_k: Iterable[int]) -> pd.DataFrame:
    pivot = targets.pivot(index="architecture_id", columns="epoch_budget", values="ranking_target")
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


def _top_k_retrieval(detailed: pd.DataFrame, top_k: Iterable[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    protocol_candidates = (*_SCORE_PROTOCOL_FIELDS, *_TARGET_PROTOCOL_FIELDS)
    group_columns = ["proxy_id", "component", "direction", "epoch_budget"]
    group_columns.extend(
        field
        for field in _available_fields(detailed, protocol_candidates)
        if field not in group_columns
    )
    for key, group in detailed.groupby(group_columns, dropna=False, sort=True):
        identifiers = dict(zip(group_columns, key, strict=True))
        direction = str(identifiers["direction"]).casefold()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"Invalid proxy direction in NAS-Bench-101 scores: {direction}")
        ranked = group.dropna(subset=["architecture_id", "score", "target_value"]).copy()
        ranked["adjusted_score"] = pd.to_numeric(ranked["score"], errors="coerce")
        if direction == "minimize":
            ranked["adjusted_score"] = -ranked["adjusted_score"]
        ranked = ranked.dropna(subset=["adjusted_score"]).drop_duplicates("architecture_id")
        if ranked.empty:
            continue
        target = ranked.set_index("architecture_id")["ranking_target"]
        score = ranked.set_index("architecture_id")["adjusted_score"]
        for requested_k in top_k:
            effective_k = min(int(requested_k), len(ranked))
            if effective_k <= 0:
                raise ValueError("top_k values must be positive")
            selected_ids = _top_k_set(score, effective_k)
            oracle_ids = _top_k_set(target, effective_k)
            overlap = len(selected_ids & oracle_ids)
            union = len(selected_ids | oracle_ids)
            selected_target = target.loc[list(selected_ids)]
            oracle_target = target.loc[list(oracle_ids)]
            records.append(
                {
                    **identifiers,
                    "requested_k": int(requested_k),
                    "effective_k": effective_k,
                    "architecture_count": len(ranked),
                    "overlap_count": overlap,
                    "precision_at_k": overlap / effective_k,
                    "jaccard_at_k": overlap / union if union else 1.0,
                    "selected_target_mean": float(selected_target.mean()),
                    "oracle_target_mean": float(oracle_target.mean()),
                    "mean_regret": float(oracle_target.mean() - selected_target.mean()),
                    "selected_target_best": float(selected_target.max()),
                    "oracle_target_best": float(target.max()),
                    "best_regret": float(target.max() - selected_target.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def _aggregate_scores(
    frame: pd.DataFrame, protocol_fields: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.dropna(subset=["architecture_id", "proxy_id"]).copy()
    working["score"] = pd.to_numeric(working["score"], errors="coerce")
    working["_successful"] = (
        working["status"].fillna("ok").astype(str).str.casefold().isin(_SUCCESS_STATUSES)
        & np.isfinite(working["score"])
    )
    score_groups = [
        "architecture_id",
        "proxy_id",
        "component",
        "direction",
        *protocol_fields,
    ]
    records: list[dict[str, Any]] = []
    for key, group in working.groupby(score_groups, dropna=False, sort=True):
        identifiers = dict(zip(score_groups, key, strict=True))
        successful = group[group["_successful"]]
        records.append(
            {
                **identifiers,
                "score": float(successful["score"].mean()) if not successful.empty else np.nan,
                "status": "ok" if not successful.empty else "failed",
                "observation_count": len(group),
                "successful_observation_count": len(successful),
                "failed_observation_count": len(group) - len(successful),
            }
        )
    scores = pd.DataFrame.from_records(records)

    coverage_groups = ["proxy_id", "component", "direction", *protocol_fields]
    coverage_records: list[dict[str, Any]] = []
    for key, group in working.groupby(coverage_groups, dropna=False, sort=True):
        identifiers = dict(zip(coverage_groups, key, strict=True))
        successful_ids = set(group.loc[group["_successful"], "architecture_id"].astype(str))
        all_ids = set(group["architecture_id"].astype(str))
        failed_ids = set(group.loc[~group["_successful"], "architecture_id"].astype(str))
        coverage_records.append(
            {
                **identifiers,
                "total_count": len(all_ids),
                "successful_count": len(successful_ids),
                "failed_count": len(failed_ids),
                "coverage": len(successful_ids) / len(all_ids) if all_ids else None,
                "total_observation_count": len(group),
                "failed_observation_count": int((~group["_successful"]).sum()),
            }
        )
    return scores, pd.DataFrame.from_records(coverage_records)


def _longest_path_depth(matrix: list[list[int]]) -> int:
    depths = [-1] * len(matrix)
    depths[0] = 0
    for target in range(1, len(matrix)):
        predecessors = [
            depths[source]
            for source in range(target)
            if matrix[source][target] and depths[source] >= 0
        ]
        if predecessors:
            depths[target] = max(predecessors) + 1
    if depths[-1] < 0:
        raise ValueError("Canonical NAS-Bench-101 graph has no input-output path")
    return depths[-1]


def _architecture_features(
    canonical_specs: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for architecture_id, specification in sorted(canonical_specs.items()):
        matrix = [[int(value) for value in row] for row in specification["matrix"]]
        operations = [str(value) for value in specification["operations"]]
        records.append(
            {
                "architecture_id": architecture_id,
                "vertices": len(matrix),
                "edges": sum(sum(row) for row in matrix),
                "longest_path_depth": _longest_path_depth(matrix),
                "conv3_count": operations[1:-1].count("conv3x3-bn-relu"),
                "conv1_count": operations[1:-1].count("conv1x1-bn-relu"),
                "maxpool_count": operations[1:-1].count("maxpool3x3"),
            }
        )
    return pd.DataFrame.from_records(records)


def _analysis_group_fields(detailed: pd.DataFrame) -> list[str]:
    fields = ["proxy_id", "component", "direction", "epoch_budget"]
    fields.extend(
        field
        for field in _available_fields(
            detailed, (*_SCORE_PROTOCOL_FIELDS, *_TARGET_PROTOCOL_FIELDS)
        )
        if field not in fields
    )
    return fields


def _direction_adjusted_scores(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    directions = adjusted["direction"].astype(str).str.casefold()
    invalid = sorted(set(directions) - {"maximize", "minimize"})
    if invalid:
        raise ValueError(f"Invalid proxy directions in NAS-Bench-101 scores: {invalid}")
    adjusted["adjusted_score"] = pd.to_numeric(adjusted["score"], errors="coerce")
    adjusted.loc[directions.eq("minimize"), "adjusted_score"] *= -1
    return adjusted


def _feature_analyses(
    detailed: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = _direction_adjusted_scores(
        detailed.merge(features, on="architecture_id", validate="many_to_one")
    )
    group_fields = _analysis_group_fields(joined)
    long = joined.melt(
        id_vars=[
            *group_fields,
            "architecture_id",
            "adjusted_score",
            "ranking_target",
        ],
        value_vars=list(_STRUCTURE_FEATURES),
        var_name="feature",
        value_name="feature_value",
    )
    strata = (
        long.groupby([*group_fields, "feature", "feature_value"], dropna=False, sort=True)
        .agg(
            architecture_count=("architecture_id", "nunique"),
            score_mean=("adjusted_score", "mean"),
            score_median=("adjusted_score", "median"),
            target_mean=("ranking_target", "mean"),
            target_median=("ranking_target", "median"),
        )
        .reset_index()
    )

    controlled_records: list[dict[str, Any]] = []
    structure_key = joined[list(_STRUCTURE_FEATURES)].apply(tuple, axis=1)
    joined = joined.assign(_structure_key=structure_key)
    for key, group in joined.groupby(group_fields, dropna=False, sort=True):
        identifiers = dict(zip(group_fields, key, strict=True))
        valid = group.dropna(subset=["adjusted_score", "ranking_target"]).copy()
        sizes = valid.groupby("_structure_key")["architecture_id"].transform("size")
        valid = valid[sizes >= 2]
        if valid.empty:
            score_residual = pd.Series(dtype=float)
            target_residual = pd.Series(dtype=float)
            informative_strata = 0
        else:
            score_residual = valid["adjusted_score"] - valid.groupby("_structure_key")[
                "adjusted_score"
            ].transform("mean")
            target_residual = valid["ranking_target"] - valid.groupby("_structure_key")[
                "ranking_target"
            ].transform("mean")
            informative_strata = int(valid["_structure_key"].nunique())
        controlled_records.append(
            {
                **identifiers,
                "control_features": ",".join(_STRUCTURE_FEATURES),
                "sample_count": len(valid),
                "informative_stratum_count": informative_strata,
                "spearman": _finite_correlation(
                    score_residual, target_residual, "spearman"
                ),
                "kendall_tau_b": _finite_correlation(
                    score_residual, target_residual, "kendall_tau_b"
                ),
            }
        )
    return strata, pd.DataFrame.from_records(controlled_records)


def _local_edit_neighbors(
    canonical_specs: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    operation_buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    edge_buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    frozen: dict[str, tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]] = {}
    for architecture_id, specification in sorted(canonical_specs.items()):
        matrix = tuple(tuple(int(value) for value in row) for row in specification["matrix"])
        operations = tuple(str(value) for value in specification["operations"])
        frozen[architecture_id] = (matrix, operations)
        for position in range(1, len(operations) - 1):
            masked = (*operations[:position], "*", *operations[position + 1 :])
            operation_buckets[(matrix, position, masked)].append(architecture_id)
        for source in range(len(matrix)):
            for target in range(source + 1, len(matrix)):
                flattened = tuple(
                    matrix[row][column]
                    for row in range(len(matrix))
                    for column in range(row + 1, len(matrix))
                    if (row, column) != (source, target)
                )
                edge_buckets[(operations, source, target, flattened)].append(architecture_id)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def add_pair(left: str, right: str, edit_type: str, edit_location: str) -> None:
        pair = tuple(sorted((left, right)))
        record = {
            "architecture_id_left": pair[0],
            "architecture_id_right": pair[1],
            "edit_type": edit_type,
            "edit_location": edit_location,
        }
        existing = pairs.get(pair)
        if existing is not None and existing != record:
            raise ValueError("NAS-Bench-101 pair matched multiple one-edit definitions")
        pairs[pair] = record

    for (matrix, position, _), architecture_ids in operation_buckets.items():
        for left, right in combinations(sorted(set(architecture_ids)), 2):
            left_matrix, left_operations = frozen[left]
            right_matrix, right_operations = frozen[right]
            differences = [
                index
                for index, values in enumerate(zip(left_operations, right_operations, strict=True))
                if values[0] != values[1]
            ]
            if left_matrix == right_matrix == matrix and differences == [position]:
                add_pair(left, right, "operation", f"vertex:{position}")

    for (operations, source, target, _), architecture_ids in edge_buckets.items():
        for left, right in combinations(sorted(set(architecture_ids)), 2):
            left_matrix, left_operations = frozen[left]
            right_matrix, right_operations = frozen[right]
            differences = [
                (row, column)
                for row in range(len(left_matrix))
                for column in range(row + 1, len(left_matrix))
                if left_matrix[row][column] != right_matrix[row][column]
            ]
            if left_operations == right_operations == operations and differences == [(source, target)]:
                add_pair(left, right, "edge", f"edge:{source}->{target}")

    columns = (
        "architecture_id_left",
        "architecture_id_right",
        "edit_type",
        "edit_location",
    )
    return pd.DataFrame.from_records(
        [pairs[key] for key in sorted(pairs)], columns=columns
    )


def _neighborhood_analysis(
    detailed: pd.DataFrame, neighbors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = _direction_adjusted_scores(detailed)
    group_fields = _analysis_group_fields(adjusted)
    difference_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for key, group in adjusted.groupby(group_fields, dropna=False, sort=True):
        identifiers = dict(zip(group_fields, key, strict=True))
        if group["architecture_id"].duplicated().any():
            raise ValueError(
                "NAS-Bench-101 neighborhood analysis requires one row per protocol architecture"
            )
        indexed = group.set_index("architecture_id")
        group_differences: list[dict[str, Any]] = []
        for pair in neighbors.itertuples(index=False):
            if pair.architecture_id_left not in indexed.index or pair.architecture_id_right not in indexed.index:
                continue
            left = indexed.loc[pair.architecture_id_left]
            right = indexed.loc[pair.architecture_id_right]
            score_difference = float(right["adjusted_score"] - left["adjusted_score"])
            target_difference = float(right["ranking_target"] - left["ranking_target"])
            if not np.isfinite(score_difference) or not np.isfinite(target_difference):
                continue
            record = {
                **identifiers,
                "architecture_id_left": pair.architecture_id_left,
                "architecture_id_right": pair.architecture_id_right,
                "edit_type": pair.edit_type,
                "edit_location": pair.edit_location,
                "score_difference": score_difference,
                "target_difference": target_difference,
            }
            group_differences.append(record)
            difference_records.append(record)
        differences = pd.DataFrame.from_records(group_differences)
        if differences.empty:
            score_differences = pd.Series(dtype=float)
            target_differences = pd.Series(dtype=float)
            comparable = pd.Series(dtype=bool)
            agreement = None
            edit_counts: dict[str, int] = {}
        else:
            score_differences = differences["score_difference"]
            target_differences = differences["target_difference"]
            comparable = score_differences.ne(0) & target_differences.ne(0)
            agreement = (
                float(
                    np.mean(
                        np.sign(score_differences[comparable])
                        == np.sign(target_differences[comparable])
                    )
                )
                if comparable.any()
                else None
            )
            edit_counts = differences["edit_type"].value_counts().to_dict()
        summary_records.append(
            {
                **identifiers,
                "pair_count": len(differences),
                "operation_pair_count": int(edit_counts.get("operation", 0)),
                "edge_pair_count": int(edit_counts.get("edge", 0)),
                "spearman": _finite_correlation(
                    score_differences, target_differences, "spearman"
                ),
                "kendall_tau_b": _finite_correlation(
                    score_differences, target_differences, "kendall_tau_b"
                ),
                "direction_comparable_pair_count": int(comparable.sum()),
                "direction_agreement_rate": agreement,
            }
        )
    return (
        pd.DataFrame.from_records(difference_records),
        pd.DataFrame.from_records(summary_records),
    )


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
    frame = read_scores(source, include_failed=True)
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

    score_protocol_fields = _available_fields(frame, _SCORE_PROTOCOL_FIELDS)
    scores, score_coverage = _aggregate_scores(frame, score_protocol_fields)
    specification_rows = frame.dropna(subset=["architecture_id", "architecture"])[
        ["architecture_id", "architecture"]
    ].copy()
    conflicts = specification_rows.assign(
        _stable_spec=specification_rows["architecture"].map(
            lambda value: repr(value) if not isinstance(value, dict) else repr(sorted(value.items()))
        )
    ).groupby("architecture_id")["_stable_spec"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("NAS-Bench-101 architecture_id maps to multiple specifications")
    specifications = specification_rows.drop_duplicates("architecture_id").set_index(
        "architecture_id"
    )["architecture"]
    canonical_specs: dict[str, Mapping[str, Any]] = {}
    target_rows: list[dict[str, Any]] = []
    for architecture_id, specification in specifications.items():
        canonical = adapter.canonicalize(specification)
        expected_id = adapter.architecture_id(canonical)
        if str(architecture_id) != expected_id:
            raise ValueError(
                f"Score architecture ID {architecture_id} does not match NAS-Bench-101 hash {expected_id}"
            )
        canonical_specs[expected_id] = canonical
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
                    "target_value": value,
                    "ranking_target": -value if target_direction == "minimize" else value,
                    "raw_target_value": value,
                    "target_direction": target_direction,
                    "dataset": dataset,
                    "target_split": split,
                    "target_metric": metric_name,
                    "target_seed": repeat_index,
                    "target_seed_reduction": seed_reduction,
                }
            )
    targets = pd.DataFrame.from_records(target_rows)
    detailed = scores.merge(targets, on="architecture_id", validate="many_to_many")
    target_protocol_fields = _available_fields(detailed, _TARGET_PROTOCOL_FIELDS)
    correlation_groups = [
        "proxy_id",
        "component",
        "direction",
        "epoch_budget",
        *score_protocol_fields,
        *target_protocol_fields,
    ]
    correlations = correlation_table(
        detailed,
        group_by=correlation_groups,
        bootstrap_samples=bootstrap_samples,
    )
    successful = detailed[
        detailed["status"].fillna("ok").astype(str).str.casefold().isin(_SUCCESS_STATUSES)
        & np.isfinite(pd.to_numeric(detailed["score"], errors="coerce"))
    ].reset_index(drop=True)
    features = _architecture_features(canonical_specs)
    feature_strata, controlled = _feature_analyses(successful, features)
    neighbors = _local_edit_neighbors(canonical_specs)
    neighborhood_differences, neighborhood_correlations = _neighborhood_analysis(
        successful, neighbors
    )
    return {
        "detailed": detailed,
        "score_coverage": score_coverage,
        "correlations": correlations,
        "rank_stability": _rank_stability(targets, top_k),
        "top_k_retrieval": _top_k_retrieval(successful, top_k),
        "architecture_features": features,
        "feature_strata": feature_strata,
        "structure_controlled_correlations": controlled,
        "edit_neighbors": neighbors,
        "neighborhood_differences": neighborhood_differences,
        "neighborhood_correlations": neighborhood_correlations,
    }


__all__ = ["nasbench101_budget_study"]
