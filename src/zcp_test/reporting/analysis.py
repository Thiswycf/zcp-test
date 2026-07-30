from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PathLike = str | Path
ScoreSource = PathLike | pd.DataFrame | Iterable[Mapping[str, Any]]

_SCORE_FIELDS: dict[str, tuple[str, ...]] = {
    "architecture_id": (
        "architecture_id",
        "architecture.id",
        "architecture.uid",
        "model_id",
    ),
    "proxy_id": ("proxy_id", "proxy.id", "proxy.name", "metric.proxy_id"),
    "proxy_version": ("proxy_version", "proxy.version"),
    "direction": ("direction", "proxy.direction", "metric.direction"),
    "component": ("component", "proxy.component", "metric.component"),
    "score": ("score", "result.score", "proxy.score", "metric.value", "value"),
    "target_metric": (
        "target_metric",
        "target.metric",
        "target.name",
        "benchmark.metric",
    ),
    "target_split": ("target_split", "target.split"),
    "target_direction": ("target_direction", "target.direction"),
    "target_epoch_budget": ("target_epoch_budget", "target.epoch_budget"),
    "target_seed": ("target_seed", "target.seed"),
    "target_seed_reduction": ("target_seed_reduction", "target.seed_reduction"),
    "target_value": (
        "target_value",
        "target.value",
        "target.score",
        "benchmark.target_value",
        "accuracy",
    ),
    "status": ("status", "result.status"),
    "error_type": ("error_type", "error.type", "result.error_type"),
    "error_message": ("error_message", "error.message", "result.error_message"),
    "duration_seconds": ("duration_seconds", "duration", "result.duration_seconds"),
    "peak_memory_mb": ("peak_memory_mb", "memory.peak_mb", "result.peak_memory_mb"),
    "input_source": ("input_source", "input.source"),
    "input_fingerprint": ("input_fingerprint", "input.fingerprint"),
    "run_id": ("run_id", "run.id"),
    "dataset": ("dataset", "dataset.id", "dataset.name", "benchmark.dataset"),
    "search_space_id": ("search_space_id", "search_space.id", "space_id"),
    "benchmark_id": ("benchmark_id", "benchmark.id"),
    "benchmark_index": ("benchmark_index", "benchmark.index"),
    "benchmark_version": ("benchmark_version", "benchmark.version"),
    "benchmark_variant": ("benchmark_variant", "benchmark.variant"),
    "benchmark_protocol": ("benchmark_protocol", "benchmark.protocol"),
    "seed": ("seed", "evaluation.seed"),
}


def _nested_value(row: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _nested_value(row, key)
        if value is not None and not isinstance(value, (Mapping, list, tuple)):
            return value
    return None


def normalize_score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one score row with stable columns across legacy and nested schemas."""
    normalized = dict(row)
    for field, candidates in _SCORE_FIELDS.items():
        value = _first_value(row, candidates)
        if value is not None or field not in normalized:
            normalized[field] = value
    if normalized.get("component") is None:
        normalized["component"] = "default"
    if normalized.get("status") is None:
        normalized["status"] = "ok" if normalized.get("score") is not None else "error"
    return normalized


def _expand_score_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if row.get("component") is not None and row.get("score") is not None:
        return [normalize_score_row(row)]
    components = row.get("components")
    if not isinstance(components, Mapping) or not components:
        return [normalize_score_row(row)]
    expanded: list[dict[str, Any]] = []
    for component, score in components.items():
        component_row = dict(row)
        component_row["component"] = str(component)
        component_row["score"] = score
        expanded.append(normalize_score_row(component_row))
    return expanded


def _score_path(source: PathLike) -> Path:
    path = Path(source)
    return path / "scores.jsonl" if path.is_dir() else path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_bytes().splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, 1):
        trailing_fragment = line_number == len(lines) and not raw_line.endswith((b"\n", b"\r"))
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            if trailing_fragment:
                break
            raise ValueError(f"Invalid UTF-8 at {path}:{line_number}") from error
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if trailing_fragment:
                break
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def read_scores(source: ScoreSource, *, include_failed: bool = False) -> pd.DataFrame:
    """Read legacy flat or current nested score records into canonical columns."""
    if isinstance(source, pd.DataFrame):
        rows = source.to_dict(orient="records")
    elif isinstance(source, (str, Path)):
        rows = _read_jsonl(_score_path(source))
    else:
        rows = list(source)
        if rows and all(isinstance(item, (str, Path)) for item in rows):
            frames = []
            for item in rows:
                frame = read_scores(item, include_failed=include_failed)
                if len(rows) > 1:
                    frame = frame.copy()
                    frame["source_run"] = str(item)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized_rows = [normalized for row in rows for normalized in _expand_score_row(row)]
    frame = pd.DataFrame(normalized_rows)
    for field in _SCORE_FIELDS:
        if field not in frame:
            frame[field] = pd.Series(dtype="object")
    for field in ("score", "target_value", "duration_seconds"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if not include_failed and not frame.empty:
        frame = frame[frame["status"].fillna("ok").isin(("ok", "success", "completed"))]
    return frame.reset_index(drop=True)


load_scores = read_scores
load_score_frame = read_scores


def _direction_adjusted(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    if "direction" not in adjusted:
        return adjusted
    minimize = adjusted["direction"].astype(str).str.casefold().eq("minimize")
    adjusted.loc[minimize, "score"] = -adjusted.loc[minimize, "score"]
    if "target_direction" in adjusted:
        target_minimize = adjusted["target_direction"].astype(str).str.casefold().eq("minimize")
        adjusted.loc[target_minimize, "target_value"] = -adjusted.loc[
            target_minimize, "target_value"
        ]
    return adjusted


PROTOCOL_FIELDS = (
    "benchmark_id",
    "benchmark_version",
    "benchmark_variant",
    "benchmark_protocol",
    "search_space_id",
    "dataset",
    "target_metric",
    "target_split",
    "target_direction",
    "target_epoch_budget",
    "target_seed_reduction",
    "target_seed",
    "input_source",
    "input_fingerprint",
    "model_fidelity",
    "seed",
)


def protocol_group_fields(frame: pd.DataFrame, base: Sequence[str]) -> tuple[str, ...]:
    fields = list(base)
    for field in PROTOCOL_FIELDS:
        if field in frame and frame[field].notna().any() and field not in fields:
            fields.append(field)
    return tuple(fields)


def _paired_values(
    target: Sequence[float] | pd.Series,
    score: Sequence[float] | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(target, dtype=float)
    score_array = np.asarray(score, dtype=float)
    if target_array.shape != score_array.shape:
        raise ValueError("target and score lengths differ")
    finite = np.isfinite(target_array) & np.isfinite(score_array)
    return target_array[finite], score_array[finite]


def _correlation(target: np.ndarray, score: np.ndarray, method: str) -> float:
    if target.size < 2 or np.unique(target).size < 2 or np.unique(score).size < 2:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(target, score).statistic)
    if method in ("kendall", "kendall_tau_b"):
        return float(stats.kendalltau(target, score, variant="b").statistic)
    if method == "pearson":
        return float(stats.pearsonr(target, score).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def bootstrap_correlation(
    target: Sequence[float] | pd.Series,
    score: Sequence[float] | pd.Series,
    *,
    method: str = "spearman",
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Estimate a paired bootstrap percentile interval for a correlation."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    target_array, score_array = _paired_values(target, score)
    estimate = _correlation(target_array, score_array, method)
    if target_array.size < 2:
        values = np.asarray([], dtype=float)
    else:
        random = np.random.default_rng(seed)
        indices = random.integers(0, target_array.size, size=(samples, target_array.size))
        values = np.asarray(
            [_correlation(target_array[index], score_array[index], method) for index in indices]
        )
        values = values[np.isfinite(values)]
    alpha = (1.0 - confidence) / 2.0
    lower, upper = (
        np.quantile(values, (alpha, 1.0 - alpha)) if values.size else (float("nan"),) * 2
    )
    return {
        "method": method,
        "estimate": None if not math.isfinite(estimate) else estimate,
        "lower": None if not math.isfinite(float(lower)) else float(lower),
        "upper": None if not math.isfinite(float(upper)) else float(upper),
        "confidence": confidence,
        "sample_count": int(target_array.size),
        "bootstrap_samples": samples,
        "valid_bootstrap_samples": int(values.size),
    }


def correlation_table(
    source: ScoreSource,
    *,
    group_by: Sequence[str] | None = None,
    methods: Sequence[str] = ("spearman", "kendall_tau_b", "pearson"),
    bootstrap_samples: int = 0,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> pd.DataFrame:
    frame = _direction_adjusted(read_scores(source))
    if group_by is None:
        proxy_fields = ["proxy_id", "component"]
        if "proxy_version" in frame and frame["proxy_version"].notna().any():
            proxy_fields.append("proxy_version")
        group_by = protocol_group_fields(frame, proxy_fields)
    required = [*group_by, "target_value", "score"]
    missing = [field for field in required if field not in frame]
    if missing:
        raise ValueError(f"Missing score fields: {', '.join(missing)}")
    records: list[dict[str, Any]] = []
    groups = frame.groupby(list(group_by), dropna=False, sort=True) if group_by else [((), frame)]
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        target, score = _paired_values(group["target_value"], group["score"])
        record = dict(zip(group_by, keys, strict=True))
        record["sample_count"] = int(target.size)
        for method in methods:
            estimate = _correlation(target, score, method)
            record[method] = None if not math.isfinite(estimate) else estimate
            if bootstrap_samples:
                interval = bootstrap_correlation(
                    target,
                    score,
                    method=method,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed=seed,
                )
                record[f"{method}_lower"] = interval["lower"]
                record[f"{method}_upper"] = interval["upper"]
        records.append(record)
    return pd.DataFrame.from_records(records)


bootstrap_correlations = correlation_table


def _plot_frame(source: ScoreSource) -> pd.DataFrame:
    frame = read_scores(source)
    if frame.empty:
        raise ValueError("Cannot plot empty data")
    return frame


def _save_figure(figure: Any, destination: PathLike | None) -> Any:
    if destination is not None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight", metadata={"Creator": "zcp-test"})
    return figure


def plot_scatter(source: ScoreSource, destination: PathLike | None = None) -> Any:
    import matplotlib.pyplot as plt

    frame = _plot_frame(source).dropna(subset=["target_value", "score"])
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, group in frame.groupby(["proxy_id", "component"], dropna=False):
        axis.scatter(group["target_value"], group["score"], alpha=0.7, label=" / ".join(map(str, label)))
    axis.set(xlabel="Target", ylabel="Proxy score", title="Proxy score vs. target")
    axis.grid(alpha=0.25)
    if frame.groupby(["proxy_id", "component"]).ngroups > 1:
        axis.legend(fontsize="small")
    figure.tight_layout()
    return _save_figure(figure, destination)


def plot_rank(source: ScoreSource, destination: PathLike | None = None) -> Any:
    import matplotlib.pyplot as plt

    frame = _direction_adjusted(_plot_frame(source)).dropna(
        subset=["target_value", "score"]
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, group in frame.groupby(["proxy_id", "component"], dropna=False):
        target_rank = group["target_value"].rank(method="average", ascending=False)
        score_rank = group["score"].rank(method="average", ascending=False)
        axis.scatter(target_rank, score_rank, alpha=0.7, label=" / ".join(map(str, label)))
    axis.set(xlabel="Target rank", ylabel="Proxy rank", title="Rank agreement")
    axis.invert_xaxis()
    axis.invert_yaxis()
    axis.grid(alpha=0.25)
    if frame.groupby(["proxy_id", "component"]).ngroups > 1:
        axis.legend(fontsize="small")
    figure.tight_layout()
    return _save_figure(figure, destination)


def plot_heatmap(
    source: ScoreSource | pd.DataFrame,
    destination: PathLike | None = None,
    *,
    method: str = "spearman",
) -> Any:
    import matplotlib.pyplot as plt

    if isinstance(source, pd.DataFrame) and method in source.columns and "sample_count" in source:
        table = source.copy()
    else:
        table = correlation_table(source, methods=(method,))
    if table.empty:
        raise ValueError("Cannot plot empty correlation data")
    table["proxy"] = table["proxy_id"].astype(str) + " / " + table["component"].astype(str)
    protocol_fields = [
        field
        for field in PROTOCOL_FIELDS
        if field in table and table[field].notna().any()
    ]
    if not protocol_fields:
        matrix = table.set_index("proxy")[[method]].T
    else:
        table["protocol"] = table.apply(
            lambda row: " | ".join(f"{field}={row[field]}" for field in protocol_fields), axis=1
        )
        if table.duplicated(["proxy", "protocol"], keep=False).any():
            raise ValueError("Correlation heatmap has duplicate proxy/protocol rows")
        matrix = table.pivot(index="proxy", columns="protocol", values=method)
    figure, axis = plt.subplots(figsize=(max(6, 1.1 * matrix.shape[1]), max(3, 0.55 * matrix.shape[0])))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(matrix.shape[0]), matrix.index)
    axis.set_title(f"{method.replace('_', ' ').title()} correlation")
    for row in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row, column_index]
            if pd.notna(value):
                axis.text(column_index, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    return _save_figure(figure, destination)


def top_k_comparison(source: ScoreSource, *, k: int | Sequence[int] = 10) -> pd.DataFrame:
    frame = _direction_adjusted(_plot_frame(source)).dropna(
        subset=["target_value", "score"]
    )
    requested = [k] if isinstance(k, int) else list(k)
    records: list[dict[str, Any]] = []
    proxy_fields = ["proxy_id", "component"]
    if "proxy_version" in frame and frame["proxy_version"].notna().any():
        proxy_fields.append("proxy_version")
    group_fields = protocol_group_fields(frame, proxy_fields)
    for label, group in frame.groupby(list(group_fields), dropna=False):
        labels = label if isinstance(label, tuple) else (label,)
        group_identity = dict(zip(group_fields, labels, strict=True))
        for size in requested:
            actual_size = min(max(int(size), 0), len(group))
            target_top = set(group.nlargest(actual_size, "target_value").index)
            score_top = set(group.nlargest(actual_size, "score").index)
            overlap = len(target_top & score_top)
            records.append(
                {
                    **group_identity,
                    "k": actual_size,
                    "overlap": overlap,
                    "overlap_fraction": overlap / actual_size if actual_size else float("nan"),
                    "selected_target_mean": group.loc[list(score_top), "target_value"].mean()
                    if score_top
                    else float("nan"),
                    "optimal_target_mean": group.loc[list(target_top), "target_value"].mean()
                    if target_top
                    else float("nan"),
                }
            )
    return pd.DataFrame.from_records(records)


def rank_aggregation(source: ScoreSource, *, require_validation: bool = True) -> pd.DataFrame:
    frame = _direction_adjusted(read_scores(source))
    if require_validation and "target_split" in frame and frame["target_split"].notna().any():
        frame = frame[frame["target_split"].isin(("valid", "validation"))]
    if frame.empty:
        return pd.DataFrame(columns=["architecture_id", "aggregate_rank", "proxy_count"])
    frame = frame.dropna(subset=["architecture_id", "proxy_id", "score"]).copy()
    protocol_fields = protocol_group_fields(frame, ())
    proxy_fields = [*protocol_fields, "proxy_id", "component"]
    frame["normalized_rank"] = frame.groupby(proxy_fields, dropna=False)["score"].rank(
        method="average", pct=True, ascending=False
    )
    return (
        frame.groupby([*protocol_fields, "architecture_id"], as_index=False, dropna=False)
        .agg(aggregate_rank=("normalized_rank", "mean"), proxy_count=("proxy_id", "nunique"))
        .sort_values([*protocol_fields, "aggregate_rank", "architecture_id"])
    )


def proxy_cost_pareto(source: ScoreSource) -> pd.DataFrame:
    frame = _direction_adjusted(read_scores(source))
    records: list[dict[str, Any]] = []
    def median_numeric(group: pd.DataFrame, field: str) -> float:
        if field not in group:
            return float("nan")
        return float(pd.to_numeric(group[field], errors="coerce").median())

    group_fields = protocol_group_fields(frame, ("proxy_id", "component"))
    for key, group in frame.groupby(list(group_fields), dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        target, score = _paired_values(group["target_value"], group["score"])
        correlation = _correlation(target, score, "spearman")
        records.append(
            {
                **dict(zip(group_fields, values, strict=True)),
                "spearman": correlation,
                "median_duration_seconds": median_numeric(group, "duration_seconds"),
                "median_peak_memory_mb": median_numeric(group, "peak_memory_mb"),
            }
        )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        table["pareto"] = pd.Series(dtype=bool)
        return table
    protocol_only = [field for field in group_fields if field not in {"proxy_id", "component"}]
    table["pareto"] = False
    groups = table.groupby(protocol_only, dropna=False) if protocol_only else [((), table)]
    for _, protocol in groups:
        objectives = protocol[["spearman", "median_duration_seconds", "median_peak_memory_mb"]].copy()
        objectives["spearman"] = objectives["spearman"].fillna(float("-inf"))
        objectives[["median_duration_seconds", "median_peak_memory_mb"]] = objectives[
            ["median_duration_seconds", "median_peak_memory_mb"]
        ].fillna(float("inf"))
        for index, row in objectives.iterrows():
            dominated = (
                (objectives["spearman"] >= row["spearman"])
                & (objectives["median_duration_seconds"] <= row["median_duration_seconds"])
                & (objectives["median_peak_memory_mb"] <= row["median_peak_memory_mb"])
                & (
                    (objectives["spearman"] > row["spearman"])
                    | (objectives["median_duration_seconds"] < row["median_duration_seconds"])
                    | (objectives["median_peak_memory_mb"] < row["median_peak_memory_mb"])
                )
            ).any()
            table.loc[index, "pareto"] = not bool(dominated)
    return table


def sample_size_convergence(
    source: ScoreSource, *, sizes: Sequence[int] = (10, 25, 50, 100), seed: int = 0
) -> pd.DataFrame:
    frame = _direction_adjusted(read_scores(source)).dropna(subset=["target_value", "score"])
    records: list[dict[str, Any]] = []
    group_fields = protocol_group_fields(frame, ("proxy_id", "component"))
    for key, group in frame.groupby(list(group_fields), dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        group = group.sample(frac=1, random_state=seed)
        for size in sizes:
            sample = group.iloc[: min(size, len(group))]
            target, score = _paired_values(sample["target_value"], sample["score"])
            records.append(
                {
                    **dict(zip(group_fields, values, strict=True)),
                    "requested_size": size,
                    "sample_count": len(sample),
                    "spearman": _correlation(target, score, "spearman"),
                }
            )
    return pd.DataFrame.from_records(records)


def transfer_correlation_table(source: ScoreSource) -> pd.DataFrame:
    frame = read_scores(source).dropna(subset=["target_value", "score"])
    return correlation_table(frame)


def plot_top_k_compare(
    source: ScoreSource,
    destination: PathLike | None = None,
    *,
    k: int | Sequence[int] = (1, 5, 10),
) -> Any:
    import matplotlib.pyplot as plt

    table = top_k_comparison(source, k=k)
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, group in table.groupby(["proxy_id", "component"], dropna=False):
        axis.plot(group["k"], group["overlap_fraction"], marker="o", label=" / ".join(map(str, label)))
    axis.set(xlabel="k", ylabel="Top-k overlap", title="Top-k selection agreement", ylim=(0, 1.05))
    axis.grid(alpha=0.25)
    if table.groupby(["proxy_id", "component"]).ngroups > 1:
        axis.legend(fontsize="small")
    figure.tight_layout()
    return _save_figure(figure, destination)


plot_topk_compare = plot_top_k_compare


def plot_sensitivity(
    source: ScoreSource,
    destination: PathLike | None = None,
    *,
    parameter: str = "seed",
    value: str = "score",
) -> Any:
    import matplotlib.pyplot as plt

    frame = _plot_frame(source)
    if parameter not in frame or value not in frame:
        raise ValueError(f"Sensitivity fields are unavailable: {parameter}, {value}")
    frame = frame.dropna(subset=[parameter, value])
    if frame.empty:
        raise ValueError("Cannot plot empty sensitivity data")
    numeric_parameter = pd.to_numeric(frame[parameter], errors="coerce")
    frame[parameter] = (
        numeric_parameter
        if numeric_parameter.notna().all()
        else frame[parameter].map(lambda item: str(item))
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    aggregate = frame.groupby(["proxy_id", "component", parameter], dropna=False)[value].agg(["mean", "std"]).reset_index()
    for label, group in aggregate.groupby(["proxy_id", "component"], dropna=False):
        group = group.sort_values(parameter)
        axis.errorbar(group[parameter], group["mean"], yerr=group["std"].fillna(0), marker="o", label=" / ".join(map(str, label)))
    axis.set(xlabel=parameter, ylabel=value, title=f"Sensitivity to {parameter}")
    axis.grid(alpha=0.25)
    if aggregate.groupby(["proxy_id", "component"]).ngroups > 1:
        axis.legend(fontsize="small")
    figure.tight_layout()
    return _save_figure(figure, destination)


def _event_frame(source: PathLike | pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.DataFrame.from_records(_read_jsonl(Path(source)))
    records = list(source)
    if records and all(isinstance(item, (str, Path)) for item in records):
        frames = []
        for item in records:
            frame = pd.DataFrame.from_records(_read_jsonl(Path(item)))
            frame["source_run"] = str(item)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame.from_records(records)


def plot_search(source: PathLike | pd.DataFrame | Iterable[Mapping[str, Any]], destination: PathLike | None = None) -> Any:
    import matplotlib.pyplot as plt

    frame = _event_frame(source)
    if "record_kind" in frame:
        summaries = frame[frame["record_kind"] == "generation_summary"].copy()
        if not summaries.empty:
            available = [
                field
                for field in ("generation", "best_so_far", "mean_score", "q25", "q75")
                if field in summaries
            ]
            for field in available:
                summaries[field] = pd.to_numeric(summaries[field], errors="coerce")
            summaries = summaries.dropna(subset=["generation", "best_so_far"]).sort_values(
                "generation"
            )
            figure, axis = plt.subplots(figsize=(7, 5))
            generation = summaries["generation"].to_numpy(dtype=float)
            best_so_far = summaries["best_so_far"].to_numpy(dtype=float)
            axis.plot(generation, best_so_far, label="best_so_far")
            if "mean_score" in summaries:
                axis.plot(
                    generation,
                    summaries["mean_score"].to_numpy(dtype=float),
                    label="mean",
                )
            if {"q25", "q75"}.issubset(summaries):
                axis.fill_between(
                    generation,
                    summaries["q25"].to_numpy(dtype=float),
                    summaries["q75"].to_numpy(dtype=float),
                    alpha=0.2,
                    label="q25-q75",
                )
            axis.set(xlabel="generation", ylabel="score", title="Evolution search")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            return _save_figure(figure, destination)
    if frame.empty or not {"generation", "score"}.issubset(frame):
        raise ValueError("Search data requires generation and score")
    frame["generation"] = pd.to_numeric(frame["generation"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["generation", "score"]).sort_values("generation")
    summary = frame.groupby("generation")["score"].agg(["max", "mean"])
    summary["best_so_far"] = summary["max"].cummax()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(summary.index, summary["best_so_far"], marker="o", label="best so far")
    axis.plot(summary.index, summary["mean"], alpha=0.8, label="generation mean")
    axis.set(xlabel="Generation", ylabel="Score", title="Search progress")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return _save_figure(figure, destination)


def plot_training(source: PathLike | pd.DataFrame | Iterable[Mapping[str, Any]], destination: PathLike | None = None) -> Any:
    import matplotlib.pyplot as plt

    frame = _event_frame(source)
    if frame.empty or "epoch" not in frame:
        raise ValueError("Training data requires epoch")
    metrics = [field for field in ("train_top1", "valid_top1", "train_loss", "valid_loss", "learning_rate") if field in frame]
    if not metrics:
        raise ValueError("Training data contains no supported metrics")
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    for field in metrics:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["epoch"]).sort_values("epoch")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plotted = False
    for field in metrics:
        values = frame[["epoch", field]].dropna()
        if values.empty:
            continue
        axis = axes[1] if field == "learning_rate" or "loss" in field else axes[0]
        axis.plot(values["epoch"], values[field], marker="o", label=field)
        plotted = True
    axes[0].set(xlabel="Epoch", ylabel="Top-1 accuracy", title="Accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Loss / learning rate", title="Optimization")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize="small")
    if not plotted:
        raise ValueError("Training data contains no plottable values")
    figure.tight_layout()
    return _save_figure(figure, destination)


scatter_plot = plot_scatter
rank_plot = plot_rank
heatmap_plot = plot_heatmap
sensitivity_plot = plot_sensitivity
search_plot = plot_search
training_plot = plot_training
plot_score_scatter = plot_scatter
plot_rank_agreement = plot_rank
plot_correlation_heatmap = plot_heatmap
plot_topk_comparison = plot_top_k_compare
plot_proxy_sensitivity = plot_sensitivity
plot_search_progress = plot_search
plot_training_curves = plot_training


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _bundle_html(title: str, artifacts: Sequence[Path], output_directory: Path) -> str:
    sections = [f"<h1>{html.escape(title)}</h1>"]
    for artifact in artifacts:
        relative = artifact.relative_to(output_directory).as_posix()
        label = html.escape(artifact.stem.replace("_", " ").title())
        if artifact.suffix == ".svg":
            sections.append(f"<section><h2>{label}</h2><img src='{html.escape(relative)}' alt='{label}'></section>")
        elif artifact.suffix == ".csv":
            sections.append(f"<p><a href='{html.escape(relative)}'>{label} CSV</a></p>")
    style = "body{font:15px system-ui;max-width:1100px;margin:auto;padding:24px;color:#222}img{max-width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:24px}"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(sections)}</body></html>"


def build_report_bundle(
    scores: ScoreSource,
    output_directory: PathLike,
    *,
    search: PathLike | pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    training: PathLike | pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    title: str = "zcp-test analysis",
    bootstrap_samples: int = 1000,
    top_k: int | Sequence[int] = (1, 5, 10),
    sensitivity_parameter: str = "seed",
) -> dict[str, Any]:
    """Build a dependency-free CSV/PNG/SVG/HTML analysis directory."""
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frame = read_scores(scores, include_failed=True)
    artifacts: list[Path] = []
    scores_csv = output / "scores.csv"
    _write_csv(frame, scores_csv)
    artifacts.append(scores_csv)
    valid = frame[frame["status"].fillna("ok").isin(("ok", "success", "completed"))]
    paired = valid.dropna(subset=["score", "target_value"])
    if not paired.empty:
        correlations = correlation_table(
            paired,
            bootstrap_samples=bootstrap_samples,
        )
        correlation_csv = output / "correlations.csv"
        _write_csv(correlations, correlation_csv)
        artifacts.append(correlation_csv)
        top_k_table = top_k_comparison(paired, k=top_k)
        top_k_csv = output / "top_k.csv"
        _write_csv(top_k_table, top_k_csv)
        artifacts.append(top_k_csv)
        plots = {
            "scatter": lambda destination: plot_scatter(paired, destination),
            "rank": lambda destination: plot_rank(paired, destination),
            "heatmap": lambda destination: plot_heatmap(correlations, destination),
            "top_k": lambda destination: plot_top_k_compare(paired, destination, k=top_k),
        }
        if sensitivity_parameter in paired and paired[sensitivity_parameter].notna().any():
            plots["sensitivity"] = lambda destination: plot_sensitivity(
                paired, destination, parameter=sensitivity_parameter
            )
        for name, plotter in plots.items():
            for suffix in ("png", "svg"):
                target = output / f"{name}.{suffix}"
                figure = plotter(target)
                plt.close(figure)
                artifacts.append(target)
        from zcp_test.reporting.proxy_studies import (
            plot_proxy_proxy_heatmap,
            plot_proxy_target_heatmap,
            proxy_study,
        )

        multi_proxy = proxy_study(paired, k=top_k)
        for name, table in multi_proxy.items():
            path = output / f"{name}.csv"
            _write_csv(table, path)
            artifacts.append(path)
        for suffix in ("png", "svg"):
            target = output / f"proxy_target_protocol_heatmap.{suffix}"
            plot_proxy_target_heatmap(multi_proxy, target)
            plt.close()
            artifacts.append(target)
        if not multi_proxy["proxy_proxy_correlations"].empty:
            for suffix in ("png", "svg"):
                target = output / f"proxy_proxy_heatmap.{suffix}"
                plot_proxy_proxy_heatmap(multi_proxy["proxy_proxy_correlations"], target)
                plt.close()
                artifacts.append(target)
    for name, source, plotter in (
        ("search", search, plot_search),
        ("training", training, plot_training),
    ):
        if source is None:
            continue
        for suffix in ("png", "svg"):
            target = output / f"{name}.{suffix}"
            figure = plotter(source, target)
            plt.close(figure)
            artifacts.append(target)
    index = output / "index.html"
    index.write_text(_bundle_html(title, artifacts, output), encoding="utf-8")
    artifacts.append(index)
    return {
        "output_directory": str(output),
        "row_count": len(frame),
        "artifacts": [str(path) for path in artifacts],
    }


create_report_bundle = build_report_bundle
static_bundle = build_report_bundle
build_static_bundle = build_report_bundle
generate_bundle = build_report_bundle


__all__ = [
    "bootstrap_correlation",
    "bootstrap_correlations",
    "build_report_bundle",
    "build_static_bundle",
    "correlation_table",
    "create_report_bundle",
    "heatmap_plot",
    "generate_bundle",
    "load_score_frame",
    "load_scores",
    "normalize_score_row",
    "plot_heatmap",
    "plot_correlation_heatmap",
    "plot_proxy_sensitivity",
    "plot_rank",
    "plot_rank_agreement",
    "plot_scatter",
    "plot_score_scatter",
    "plot_search",
    "plot_search_progress",
    "plot_sensitivity",
    "plot_top_k_compare",
    "plot_topk_compare",
    "plot_topk_comparison",
    "plot_training",
    "plot_training_curves",
    "rank_plot",
    "read_scores",
    "scatter_plot",
    "search_plot",
    "sensitivity_plot",
    "static_bundle",
    "top_k_comparison",
    "proxy_cost_pareto",
    "rank_aggregation",
    "sample_size_convergence",
    "transfer_correlation_table",
    "training_plot",
]
