from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from zcp_test.reporting.analysis import PROTOCOL_FIELDS, ScoreSource, read_scores


PathLike = str | Path

PROXY_SPEC_FIELDS = ("proxy_id", "component", "proxy_version")
STRICT_PROTOCOL_FIELDS = tuple(
    field
    for field in dict.fromkeys(
        (
            *PROTOCOL_FIELDS,
            "target_seed",
            "input_source",
            "input_fingerprint",
            "seed",
            "model_fidelity",
            "run_id",
            "source_run",
        )
    )
    if field
    not in {
        "proxy_implementation_fidelity",
        "proxy_alias_of",
        "run_id",
        "source_run",
    }
)
TIE_STRATEGY = "score_desc_then_architecture_id_asc_ordinal"
FUSION_SPLIT_STRATEGY = "architecture_id_ascending_prefix_validation"
OBSERVATION_NOTE = "observational association; not a causal estimate"


def _active_fields(frame: pd.DataFrame, candidates: Sequence[str]) -> tuple[str, ...]:
    return tuple(field for field in candidates if field in frame and frame[field].notna().any())


def _proxy_label(row: Mapping[str, Any]) -> str:
    proxy_id = str(row["proxy_id"])
    component = str(row.get("component", "default"))
    version = row.get("proxy_version")
    suffix = "" if pd.isna(version) else f"@{version}"
    return f"{proxy_id}/{component}{suffix}"


def _group_items(frame: pd.DataFrame, fields: Sequence[str]):
    if not fields:
        yield (), frame
        return
    yield from frame.groupby(list(fields), dropna=False, sort=True)


def _group_values(fields: Sequence[str], key: Any) -> dict[str, Any]:
    values = key if isinstance(key, tuple) else (key,)
    return dict(zip(fields, values, strict=True))


def _prepare(source: ScoreSource) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = read_scores(source)
    required = ("architecture_id", "proxy_id", "score")
    missing = [field for field in required if field not in frame or frame[field].isna().all()]
    if missing:
        raise ValueError(f"Missing score fields: {', '.join(missing)}")
    frame = frame.copy()
    frame["architecture_id"] = frame["architecture_id"].astype(str)
    frame["component"] = frame["component"].fillna("default").astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")

    direction = frame["direction"].fillna("maximize").astype(str).str.casefold()
    invalid_direction = sorted(set(direction) - {"maximize", "minimize"})
    if invalid_direction:
        raise ValueError(f"Unknown proxy direction(s): {', '.join(invalid_direction)}")
    frame["direction"] = direction
    frame["adjusted_score"] = frame["score"].where(direction.eq("maximize"), -frame["score"])

    target_direction = frame["target_direction"].fillna("maximize").astype(str).str.casefold()
    invalid_target_direction = sorted(set(target_direction) - {"maximize", "minimize"})
    if invalid_target_direction:
        raise ValueError(
            f"Unknown target direction(s): {', '.join(invalid_target_direction)}"
        )
    frame["target_direction"] = target_direction
    frame["adjusted_target"] = frame["target_value"].where(
        target_direction.eq("maximize"), -frame["target_value"]
    )

    protocol_fields = _active_fields(frame, STRICT_PROTOCOL_FIELDS)
    proxy_fields = _active_fields(frame, PROXY_SPEC_FIELDS)
    duplicate_fields = [*protocol_fields, *proxy_fields, "architecture_id"]
    duplicates = frame.duplicated(duplicate_fields, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, duplicate_fields].iloc[0].to_dict()
        raise ValueError(
            "Duplicate architecture×proxy observation within one protocol: "
            f"{example}"
        )

    consistency_fields = [*protocol_fields, *proxy_fields]
    for _, group in _group_items(frame, consistency_fields):
        if group["direction"].nunique(dropna=False) > 1:
            raise ValueError("A proxy specification has inconsistent directions within a protocol")

    target_key = [*protocol_fields, "architecture_id"]
    finite_targets = frame[np.isfinite(frame["adjusted_target"])].copy()
    if not finite_targets.empty:
        target_counts = finite_targets.groupby(target_key, dropna=False)["adjusted_target"].nunique()
        if target_counts.gt(1).any():
            raise ValueError("Conflicting target values for one architecture within a protocol")

    frame["proxy_label"] = frame.apply(_proxy_label, axis=1)
    return frame, protocol_fields


def _correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    finite = np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(right.to_numpy(dtype=float))
    left_values = left.to_numpy(dtype=float)[finite]
    right_values = right.to_numpy(dtype=float)[finite]
    if (
        len(left_values) < 2
        or np.unique(left_values).size < 2
        or np.unique(right_values).size < 2
    ):
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(left_values, right_values).statistic)
    if method in ("kendall", "kendall_tau_b"):
        return float(stats.kendalltau(left_values, right_values, variant="b").statistic)
    if method == "pearson":
        return float(stats.pearsonr(left_values, right_values).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def _stable_order(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return frame.sort_values(
        [value, "architecture_id"], ascending=[False, True], kind="mergesort"
    )


def _top_ids(frame: pd.DataFrame, value: str, k: int) -> set[str]:
    finite = frame[np.isfinite(frame[value])]
    return set(_stable_order(finite, value).head(k)["architecture_id"])


def _ordinal_rank_score(frame: pd.DataFrame, value: str) -> pd.Series:
    ordered = _stable_order(frame, value)
    count = len(ordered)
    scores = np.ones(count) if count == 1 else 1.0 - np.arange(count) / (count - 1)
    return pd.Series(scores, index=ordered.index).reindex(frame.index)


def _csv_wide(
    frame: pd.DataFrame,
    *,
    index: Sequence[str],
    columns: str,
    values: str,
) -> pd.DataFrame:
    metadata = frame[list(index)].drop_duplicates(ignore_index=True)
    metadata["_matrix_row"] = np.arange(len(metadata))
    keyed = frame.merge(metadata, on=list(index), how="left", validate="many_to_one")
    values_wide = keyed.pivot(index="_matrix_row", columns=columns, values=values)
    values_wide = values_wide.rename_axis(columns=None).reset_index()
    return metadata.merge(values_wide, on="_matrix_row", validate="one_to_one").drop(
        columns="_matrix_row"
    )


def proxy_target_protocol_matrix(
    source: ScoreSource,
    *,
    methods: Sequence[str] = ("spearman", "kendall_tau_b", "pearson"),
) -> dict[str, pd.DataFrame]:
    """Build protocol-separated proxy-target correlations in long and CSV-wide forms."""
    frame, protocol_fields = _prepare(source)
    observation_columns = list(
        dict.fromkeys(
            [
                *protocol_fields,
                "architecture_id",
                "proxy_id",
                "component",
                "proxy_version",
                "proxy_label",
                "direction",
                "adjusted_score",
                "target_value",
                "target_direction",
                "adjusted_target",
            ]
        )
    )
    observations = frame[observation_columns].copy().reset_index(drop=True)
    architecture_matrix = _csv_wide(
        observations,
        index=[*protocol_fields, "architecture_id", "adjusted_target"],
        columns="proxy_label",
        values="adjusted_score",
    )
    records: list[dict[str, Any]] = []
    proxy_fields = _active_fields(frame, PROXY_SPEC_FIELDS)
    group_fields = [*protocol_fields, *proxy_fields]
    target_universe = (
        frame[np.isfinite(frame["adjusted_target"])]
        .groupby(list(protocol_fields), dropna=False)["architecture_id"]
        .nunique()
        if protocol_fields
        else pd.Series({(): frame.loc[np.isfinite(frame["adjusted_target"]), "architecture_id"].nunique()})
    )
    for key, group in _group_items(frame, group_fields):
        group_values = _group_values(group_fields, key)
        finite = group[np.isfinite(group["adjusted_score"]) & np.isfinite(group["adjusted_target"])]
        protocol_key = tuple(group_values[field] for field in protocol_fields)
        if len(protocol_fields) == 1:
            protocol_key = protocol_key[0]
        universe_count = int(target_universe.get(protocol_key, 0))
        base = {
            **group_values,
            "proxy_label": group["proxy_label"].iloc[0],
            "sample_count": int(len(finite)),
            "target_architecture_count": universe_count,
            "coverage": len(finite) / universe_count if universe_count else float("nan"),
            "tie_strategy": TIE_STRATEGY,
        }
        for method in methods:
            records.append(
                {
                    **base,
                    "method": method,
                    "correlation": _correlation(
                        finite["adjusted_target"], finite["adjusted_score"], method
                    ),
                }
            )
    long = pd.DataFrame.from_records(records)
    if long.empty:
        matrix = pd.DataFrame(columns=[*protocol_fields, "method"])
    else:
        matrix = _csv_wide(
            long,
            index=[*protocol_fields, "method"],
            columns="proxy_label",
            values="correlation",
        )
    return {
        "long": long,
        "matrix": matrix,
        "observations": observations,
        "architecture_matrix": architecture_matrix,
    }


def _proxy_frames(protocol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(label): group[["architecture_id", "adjusted_score", "adjusted_target"]].copy()
        for label, group in protocol.groupby("proxy_label", sort=True, dropna=False)
    }


def _joined_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    joined = left.merge(
        right,
        on="architecture_id",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    joined = joined[np.isfinite(joined["adjusted_score_left"]) & np.isfinite(joined["adjusted_score_right"])]
    joined["adjusted_target"] = joined["adjusted_target_left"].combine_first(
        joined["adjusted_target_right"]
    )
    return joined.sort_values("architecture_id", kind="mergesort").reset_index(drop=True)


def proxy_proxy_correlations(source: ScoreSource) -> pd.DataFrame:
    """Compare proxies only after an inner join on architecture within each protocol."""
    frame, protocol_fields = _prepare(source)
    records: list[dict[str, Any]] = []
    for key, protocol in _group_items(frame, protocol_fields):
        protocol_values = _group_values(protocol_fields, key)
        proxies = _proxy_frames(protocol)
        for left_label, right_label in itertools.combinations(sorted(proxies), 2):
            left, right = proxies[left_label], proxies[right_label]
            joined = _joined_pair(left, right)
            left_count = int(np.isfinite(left["adjusted_score"]).sum())
            right_count = int(np.isfinite(right["adjusted_score"]).sum())
            left_ids = set(left.loc[np.isfinite(left["adjusted_score"]), "architecture_id"])
            right_ids = set(right.loc[np.isfinite(right["adjusted_score"]), "architecture_id"])
            union_count = len(left_ids | right_ids)
            records.append(
                {
                    **protocol_values,
                    "proxy_left": left_label,
                    "proxy_right": right_label,
                    "left_sample_count": left_count,
                    "right_sample_count": right_count,
                    "common_sample_count": int(len(joined)),
                    "union_sample_count": union_count,
                    "common_coverage": len(joined) / union_count if union_count else float("nan"),
                    "left_coverage": len(joined) / left_count if left_count else float("nan"),
                    "right_coverage": len(joined) / right_count if right_count else float("nan"),
                    "spearman": _correlation(
                        joined["adjusted_score_left"], joined["adjusted_score_right"], "spearman"
                    ),
                    "kendall_tau_b": _correlation(
                        joined["adjusted_score_left"], joined["adjusted_score_right"], "kendall_tau_b"
                    ),
                    "pearson": _correlation(
                        joined["adjusted_score_left"], joined["adjusted_score_right"], "pearson"
                    ),
                    "tie_strategy": TIE_STRATEGY,
                }
            )
    return pd.DataFrame.from_records(records)


def proxy_proxy_top_k(
    source: ScoreSource, *, k: int | Sequence[int] = (1, 5, 10)
) -> pd.DataFrame:
    """Report pairwise top-k set agreement using stable architecture-id tie breaks."""
    requested = [k] if isinstance(k, int) else list(k)
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("k must contain positive integers")
    frame, protocol_fields = _prepare(source)
    records: list[dict[str, Any]] = []
    for key, protocol in _group_items(frame, protocol_fields):
        protocol_values = _group_values(protocol_fields, key)
        proxies = _proxy_frames(protocol)
        for left_label, right_label in itertools.combinations(sorted(proxies), 2):
            joined = _joined_pair(proxies[left_label], proxies[right_label])
            for requested_k in requested:
                effective_k = min(requested_k, len(joined))
                left_ids = _top_ids(joined, "adjusted_score_left", effective_k)
                right_ids = _top_ids(joined, "adjusted_score_right", effective_k)
                intersection = len(left_ids & right_ids)
                union = len(left_ids | right_ids)
                records.append(
                    {
                        **protocol_values,
                        "proxy_left": left_label,
                        "proxy_right": right_label,
                        "k": requested_k,
                        "effective_k": effective_k,
                        "common_sample_count": int(len(joined)),
                        "intersection_count": intersection,
                        "union_count": union,
                        "jaccard": intersection / union if union else float("nan"),
                        "tie_strategy": TIE_STRATEGY,
                    }
                )
    return pd.DataFrame.from_records(records)


def _residual_correlation(joined: pd.DataFrame) -> float:
    finite = joined[
        np.isfinite(joined["adjusted_target"])
        & np.isfinite(joined["adjusted_score_left"])
        & np.isfinite(joined["adjusted_score_right"])
    ]
    if len(finite) < 3 or finite["adjusted_target"].nunique() < 2:
        return float("nan")
    design = np.column_stack([np.ones(len(finite)), finite["adjusted_target"].to_numpy()])
    left_residual = finite["adjusted_score_left"].to_numpy() - design @ np.linalg.lstsq(
        design, finite["adjusted_score_left"].to_numpy(), rcond=None
    )[0]
    right_residual = finite["adjusted_score_right"].to_numpy() - design @ np.linalg.lstsq(
        design, finite["adjusted_score_right"].to_numpy(), rcond=None
    )[0]
    return _correlation(pd.Series(left_residual), pd.Series(right_residual), "pearson")


def _fusion_metrics(
    joined: pd.DataFrame,
    validation_fraction: float,
    *,
    target_split: str | None,
) -> dict[str, Any]:
    normalized_split = (target_split or "").strip().casefold()
    if normalized_split not in {"valid", "validation", "val"}:
        return {
            "fusion_weight_left": float("nan"),
            "fusion_validation_count": 0,
            "fusion_evaluation_count": 0,
            "fusion_validation_spearman": float("nan"),
            "fusion_evaluation_spearman": float("nan"),
            "best_single_evaluation_spearman": float("nan"),
            "fusion_gain_over_best_single": float("nan"),
            "fusion_status": "unsupported_target_split",
        }
    finite = joined[
        np.isfinite(joined["adjusted_target"])
        & np.isfinite(joined["adjusted_score_left"])
        & np.isfinite(joined["adjusted_score_right"])
    ].sort_values("architecture_id", kind="mergesort").copy()
    if len(finite) < 4:
        return {
            "fusion_weight_left": float("nan"),
            "fusion_validation_count": 0,
            "fusion_evaluation_count": 0,
            "fusion_validation_spearman": float("nan"),
            "fusion_evaluation_spearman": float("nan"),
            "best_single_evaluation_spearman": float("nan"),
            "fusion_gain_over_best_single": float("nan"),
            "fusion_status": "insufficient_samples",
        }
    validation_count = min(len(finite) - 2, max(2, int(math.ceil(len(finite) * validation_fraction))))
    validation = finite.iloc[:validation_count].copy()
    evaluation = finite.iloc[validation_count:].copy()
    validation["left_rank_score"] = _ordinal_rank_score(validation, "adjusted_score_left")
    validation["right_rank_score"] = _ordinal_rank_score(validation, "adjusted_score_right")
    evaluation["left_rank_score"] = _ordinal_rank_score(evaluation, "adjusted_score_left")
    evaluation["right_rank_score"] = _ordinal_rank_score(evaluation, "adjusted_score_right")
    candidates = []
    for weight in np.linspace(0.0, 1.0, 5):
        fused = weight * validation["left_rank_score"] + (1.0 - weight) * validation["right_rank_score"]
        candidates.append((
            _correlation(validation["adjusted_target"], fused, "spearman"),
            float(weight),
        ))
    valid_candidates = [item for item in candidates if math.isfinite(item[0])]
    if not valid_candidates:
        selected_weight = 0.5
        validation_score = float("nan")
    else:
        validation_score, selected_weight = max(
            valid_candidates, key=lambda item: (item[0], -abs(item[1] - 0.5), -item[1])
        )
    fused_evaluation = (
        selected_weight * evaluation["left_rank_score"]
        + (1.0 - selected_weight) * evaluation["right_rank_score"]
    )
    fusion_score = _correlation(evaluation["adjusted_target"], fused_evaluation, "spearman")
    single_scores = [
        _correlation(evaluation["adjusted_target"], evaluation[column], "spearman")
        for column in ("adjusted_score_left", "adjusted_score_right")
    ]
    finite_single = [score for score in single_scores if math.isfinite(score)]
    best_single = max(finite_single) if finite_single else float("nan")
    gain = fusion_score - best_single if math.isfinite(fusion_score) and math.isfinite(best_single) else float("nan")
    return {
        "fusion_weight_left": selected_weight,
        "fusion_validation_count": int(len(validation)),
        "fusion_evaluation_count": int(len(evaluation)),
        "fusion_validation_spearman": validation_score,
        "fusion_evaluation_spearman": fusion_score,
        "best_single_evaluation_spearman": best_single,
        "fusion_gain_over_best_single": gain,
        "fusion_status": "ok",
    }


def proxy_complementarity(
    source: ScoreSource,
    *,
    k: int | Sequence[int] = (1, 5, 10),
    validation_fraction: float = 0.5,
) -> pd.DataFrame:
    """Describe pairwise redundancy/complementarity without causal interpretation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    requested = [k] if isinstance(k, int) else list(k)
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("k must contain positive integers")
    frame, protocol_fields = _prepare(source)
    records: list[dict[str, Any]] = []
    for key, protocol in _group_items(frame, protocol_fields):
        protocol_values = _group_values(protocol_fields, key)
        proxies = _proxy_frames(protocol)
        for left_label, right_label in itertools.combinations(sorted(proxies), 2):
            joined = _joined_pair(proxies[left_label], proxies[right_label])
            joined = joined[np.isfinite(joined["adjusted_target"])].copy()
            residual = _residual_correlation(joined)
            fusion = _fusion_metrics(
                joined,
                validation_fraction,
                target_split=protocol_values.get("target_split"),
            )
            for requested_k in requested:
                effective_k = min(requested_k, len(joined))
                target_ids = _top_ids(joined, "adjusted_target", effective_k)
                left_ids = _top_ids(joined, "adjusted_score_left", effective_k)
                right_ids = _top_ids(joined, "adjusted_score_right", effective_k)
                left_recall = len(target_ids & left_ids) / effective_k if effective_k else float("nan")
                right_recall = len(target_ids & right_ids) / effective_k if effective_k else float("nan")
                union_recall = len(target_ids & (left_ids | right_ids)) / effective_k if effective_k else float("nan")
                records.append(
                    {
                        **protocol_values,
                        "proxy_left": left_label,
                        "proxy_right": right_label,
                        "k": requested_k,
                        "effective_k": effective_k,
                        "common_target_sample_count": int(len(joined)),
                        "residual_pearson": residual,
                        "left_top_k_recall": left_recall,
                        "right_top_k_recall": right_recall,
                        "top_k_union_recall": union_recall,
                        "top_k_union_marginal_gain": union_recall - max(left_recall, right_recall)
                        if effective_k
                        else float("nan"),
                        **fusion,
                        "tie_strategy": TIE_STRATEGY,
                        "fusion_split_strategy": FUSION_SPLIT_STRATEGY,
                        "interpretation": OBSERVATION_NOTE,
                    }
                )
    return pd.DataFrame.from_records(records)


def proxy_study(
    source: ScoreSource,
    *,
    k: int | Sequence[int] = (1, 5, 10),
    validation_fraction: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Return CSV-friendly tables for a complete multi-proxy observational study."""
    reusable_source = read_scores(source)
    target = proxy_target_protocol_matrix(reusable_source)
    return {
        "proxy_target_long": target["long"],
        "proxy_target_matrix": target["matrix"],
        "proxy_target_observations": target["observations"],
        "proxy_target_architecture_matrix": target["architecture_matrix"],
        "proxy_proxy_correlations": proxy_proxy_correlations(reusable_source),
        "proxy_proxy_top_k": proxy_proxy_top_k(reusable_source, k=k),
        "complementarity": proxy_complementarity(
            reusable_source, k=k, validation_fraction=validation_fraction
        ),
    }


def _save_figure(figure: Any, destination: PathLike | None) -> Any:
    if destination is None:
        return figure
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path


def _annotated_heatmap(table: pd.DataFrame, title: str, destination: PathLike | None) -> Any:
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("Cannot plot an empty heatmap")
    figure, axis = plt.subplots(
        figsize=(max(6, 0.8 * len(table.columns)), max(4, 0.55 * len(table.index)))
    )
    values = table.to_numpy(dtype=float)
    image = axis.imshow(values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(table.columns)), labels=table.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(table.index)), labels=table.index)
    axis.set_title(title)
    for row, column in itertools.product(range(values.shape[0]), range(values.shape[1])):
        if np.isfinite(values[row, column]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.85)
    return _save_figure(figure, destination)


def _protocol_label(fields: Sequence[str], values: Mapping[str, Any]) -> str:
    labels = []
    for field in fields:
        value = values[field]
        if field == "input_fingerprint":
            value = str(value)[:10]
        labels.append(f"{field}={value}")
    return " | ".join(labels) or "default"


def plot_proxy_target_heatmap(
    source: ScoreSource | Mapping[str, pd.DataFrame],
    destination: PathLike | None = None,
    *,
    method: str = "spearman",
) -> Any:
    result = source if isinstance(source, Mapping) else proxy_target_protocol_matrix(source)
    long = result["long"] if "long" in result else result["proxy_target_long"]
    selected = long[long["method"] == method].copy()
    protocol_fields = _active_fields(selected, STRICT_PROTOCOL_FIELDS)
    selected["protocol"] = selected.apply(
        lambda row: _protocol_label(
            protocol_fields, {field: row[field] for field in protocol_fields}
        ),
        axis=1,
    )
    table = selected.pivot(index="protocol", columns="proxy_label", values="correlation")
    return _annotated_heatmap(table, f"Proxy-target {method}", destination)


def plot_proxy_proxy_heatmap(
    source: ScoreSource | pd.DataFrame,
    destination: PathLike | None = None,
    *,
    method: str = "spearman",
) -> Any:
    correlations = source if isinstance(source, pd.DataFrame) else proxy_proxy_correlations(source)
    if method not in correlations:
        raise ValueError(f"Unknown proxy-proxy correlation method: {method}")
    labels = sorted(set(correlations["proxy_left"]) | set(correlations["proxy_right"]))
    protocol_fields = _active_fields(correlations, STRICT_PROTOCOL_FIELDS)
    protocol_groups = list(_group_items(correlations, protocol_fields))
    if len(protocol_groups) == 1:
        table = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
        for _, row in correlations.iterrows():
            table.loc[row["proxy_left"], row["proxy_right"]] = row[method]
            table.loc[row["proxy_right"], row["proxy_left"]] = row[method]
        return _annotated_heatmap(table, f"Proxy-proxy {method}", destination)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(protocol_groups),
        1,
        figsize=(max(8, 0.6 * len(labels)), max(6, 0.62 * len(labels) * len(protocol_groups))),
        squeeze=False,
    )
    image = None
    annotation_size = 6 if len(labels) > 15 else 8
    for index, (key, group) in enumerate(protocol_groups):
        values = _group_values(protocol_fields, key)
        table = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
        for _, row in group.iterrows():
            table.loc[row["proxy_left"], row["proxy_right"]] = row[method]
            table.loc[row["proxy_right"], row["proxy_left"]] = row[method]
        axis = axes[index, 0]
        array = table.to_numpy(dtype=float)
        image = axis.imshow(array, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_title(_protocol_label(protocol_fields, values), fontsize="small")
        for row, column in itertools.product(range(len(labels)), repeat=2):
            if np.isfinite(array[row, column]):
                axis.text(
                    column,
                    row,
                    f"{array[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                )
    figure.suptitle(f"Proxy-proxy {method}")
    if image is not None:
        figure.colorbar(image, ax=axes[:, 0].tolist(), shrink=0.5)
    figure.subplots_adjust(hspace=0.35, top=0.96)
    return _save_figure(figure, destination)


proxy_target_matrix = proxy_target_protocol_matrix
proxy_proxy_correlation_table = proxy_proxy_correlations
proxy_proxy_topk_table = proxy_proxy_top_k
complementarity_table = proxy_complementarity
run_proxy_study = proxy_study
