from __future__ import annotations

import html
import hashlib
import json
import math
from itertools import combinations
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
    "component": ("component", "primary_component", "proxy.component", "metric.component"),
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
    proxy_id = str(normalized.get("proxy_id", "")).casefold()
    proxy_version = normalized.get("proxy_version")
    if (
        proxy_id in {"params", "flops"}
        and proxy_version in {None, "", 1, "1"}
        and str(normalized.get("direction", "")).casefold() == "minimize"
    ):
        normalized["reported_proxy_version"] = str(proxy_version or "1")
        normalized["proxy_version"] = {
            "params": "count-v2",
            "flops": "thop-v2",
        }[proxy_id]
        normalized["reported_direction"] = normalized["direction"]
        normalized["direction"] = "maximize"
        normalized["resource_direction"] = "minimize"
        normalized["direction_migration"] = "legacy-resource-direction-to-accuracy-v2"
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


def validate_analysis_scores(
    frame: pd.DataFrame,
    action: str,
    *,
    sensitivity_parameter: str = "seed",
) -> None:
    if frame.empty:
        raise ValueError("Analysis requires at least one successful score record")
    required = ("architecture_id", "proxy_id", "component", "score", "target_value")
    unusable = [
        field
        for field in required
        if field not in frame or frame[field].isna().all()
    ]
    if unusable:
        raise ValueError(
            f"{action} analysis requires non-empty fields: {', '.join(unusable)}"
        )
    finite_scores = pd.to_numeric(frame["score"], errors="coerce")
    finite_targets = pd.to_numeric(frame["target_value"], errors="coerce")
    if not np.isfinite(finite_scores).any() or not np.isfinite(finite_targets).any():
        raise ValueError(f"{action} analysis requires finite score and target_value pairs")
    if action == "sensitivity":
        if sensitivity_parameter not in frame:
            raise ValueError(
                f"Sensitivity parameter field is unavailable: {sensitivity_parameter}"
            )
        values = frame[sensitivity_parameter].dropna()
        if values.nunique() < 2:
            raise ValueError(
                f"Sensitivity analysis requires at least two {sensitivity_parameter} values"
            )


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
PROXY_METADATA_FIELDS = (
    "proxy_implementation_fidelity",
    "proxy_alias_of",
    "resource_direction",
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


def _correlation_diagnostics(
    target: Sequence[float] | pd.Series,
    score: Sequence[float] | pd.Series,
    *,
    total_count: int | None = None,
    failed_count: int = 0,
) -> dict[str, Any]:
    target_values = pd.to_numeric(pd.Series(target), errors="coerce")
    score_values = pd.to_numeric(pd.Series(score), errors="coerce")
    finite = np.isfinite(target_values.to_numpy(dtype=float)) & np.isfinite(
        score_values.to_numpy(dtype=float)
    )
    paired_target = target_values[finite]
    paired_score = score_values[finite]
    successful_count = int(len(target_values))
    total_count = successful_count if total_count is None else int(total_count)
    sample_count = int(finite.sum())
    target_unique_count = int(paired_target.nunique(dropna=True))
    score_unique_count = int(paired_score.nunique(dropna=True))
    if sample_count < 2:
        status = "insufficient_samples"
    elif target_unique_count < 2 and score_unique_count < 2:
        status = "constant_target_and_score"
    elif target_unique_count < 2:
        status = "constant_target"
    elif score_unique_count < 2:
        status = "constant_score"
    else:
        status = "ok"
    return {
        "total_count": total_count,
        "successful_count": successful_count,
        "failed_count": int(failed_count),
        "sample_count": sample_count,
        "invalid_count": total_count - sample_count,
        "coverage": sample_count / total_count if total_count else 0.0,
        "target_unique_count": target_unique_count,
        "score_unique_count": score_unique_count,
        "target_tied_observations": int(paired_target.duplicated(keep=False).sum()),
        "score_tied_observations": int(paired_score.duplicated(keep=False).sum()),
        "correlation_status": status,
    }


def _invocation_counts(
    all_records: pd.DataFrame,
    group: pd.DataFrame,
    group_by: Sequence[str],
) -> tuple[int, int]:
    matched = all_records
    for field in group_by:
        if field == "component" or field not in matched or field not in group:
            continue
        values = group[field].drop_duplicates()
        if len(values) != 1:
            continue
        value = values.iloc[0]
        matched = matched[matched[field].isna()] if pd.isna(value) else matched[matched[field] == value]
    if "architecture_id" in matched and matched["architecture_id"].notna().any():
        total_count = int(matched["architecture_id"].nunique(dropna=True))
        failed_count = int(
            matched.loc[
                ~matched["status"].fillna("ok").isin(("ok", "success", "completed")),
                "architecture_id",
            ].nunique(dropna=True)
        )
    else:
        total_count = int(len(matched))
        failed_count = int(
            (~matched["status"].fillna("ok").isin(("ok", "success", "completed"))).sum()
        )
    return total_count, failed_count


def _direction_diagnostic(values: pd.Series) -> tuple[str | None, str]:
    directions = sorted(
        {
            str(value).casefold()
            for value in values.dropna()
            if str(value).casefold() in {"maximize", "minimize"}
        }
    )
    if directions == ["maximize"]:
        return "maximize", "identity"
    if directions == ["minimize"]:
        return "minimize", "negated"
    return None, "mixed_or_missing"


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
    all_records = read_scores(source, include_failed=True)
    frame = all_records[
        all_records["status"].fillna("ok").isin(("ok", "success", "completed"))
    ].reset_index(drop=True)
    frame = _direction_adjusted(frame)
    if group_by is None:
        proxy_fields = ["proxy_id", "component"]
        if "proxy_version" in frame and frame["proxy_version"].notna().any():
            proxy_fields.append("proxy_version")
        proxy_fields.extend(
            field
            for field in PROXY_METADATA_FIELDS
            if field in frame and frame[field].notna().any()
        )
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
        duplicate_ids = group["architecture_id"].dropna().duplicated(keep=False)
        if duplicate_ids.any():
            examples = sorted(group.loc[duplicate_ids, "architecture_id"].astype(str).unique())[:3]
            raise ValueError(
                "Correlation input contains duplicate architecture IDs within one protocol: "
                + ", ".join(examples)
            )
        total_count, failed_count = _invocation_counts(all_records, group, group_by)
        record.update(
            _correlation_diagnostics(
                group["target_value"],
                group["score"],
                total_count=total_count,
                failed_count=failed_count,
            )
        )
        score_direction, score_transform = _direction_diagnostic(group["direction"])
        target_direction, target_transform = _direction_diagnostic(group["target_direction"])
        record["score_direction"] = score_direction
        record["score_direction_transform"] = score_transform
        record["target_direction"] = target_direction
        record["target_direction_transform"] = target_transform
        migrations = (
            sorted(group["direction_migration"].dropna().astype(str).unique())
            if "direction_migration" in group
            else []
        )
        reported_versions = (
            sorted(group["reported_proxy_version"].dropna().astype(str).unique())
            if "reported_proxy_version" in group
            else []
        )
        record["legacy_direction_migrated_count"] = int(
            group["direction_migration"].notna().sum()
        ) if "direction_migration" in group else 0
        record["direction_migrations"] = ",".join(migrations) or None
        record["legacy_reported_proxy_versions"] = ",".join(reported_versions) or None
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
    if "proxy_version" in table and table["proxy_version"].notna().any():
        table["proxy"] += "@" + table["proxy_version"].fillna("unknown").astype(str)
    if (
        "proxy_implementation_fidelity" in table
        and table["proxy_implementation_fidelity"].notna().any()
    ):
        table["proxy"] += " [" + table["proxy_implementation_fidelity"].fillna("unknown").astype(str) + "]"
    protocol_fields = [
        field
        for field in PROTOCOL_FIELDS
        if field in table and table[field].notna().any()
    ]
    if not protocol_fields:
        matrix = table.set_index("proxy")[[method]].T
    else:
        def protocol_label(row: pd.Series) -> str:
            benchmark = str(row.get("benchmark_id") or "benchmark")
            dataset = str(row.get("dataset") or "dataset")
            split = str(row.get("target_split") or "split")
            metric = str(row.get("target_metric") or "metric")
            budget = row.get("target_epoch_budget")
            seed = row.get("seed")
            full = json.dumps(
                [(field, None if pd.isna(row[field]) else str(row[field])) for field in protocol_fields],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
            budget_label = "" if pd.isna(budget) else f"@{budget}"
            seed_label = "" if pd.isna(seed) else f" | seed={seed}"
            return (
                f"{benchmark}/{dataset}/{split}/{metric}{budget_label}"
                f"{seed_label} | protocol={digest}"
            )

        table["protocol"] = table.apply(protocol_label, axis=1)
        if table.duplicated(["proxy", "protocol"], keep=False).any():
            raise ValueError("Correlation heatmap has duplicate proxy/protocol rows")
        matrix = table.pivot(index="proxy", columns="protocol", values=method)
    figure, axis = plt.subplots(figsize=(max(6, 1.1 * matrix.shape[1]), max(3, 0.55 * matrix.shape[0])))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    rotation = 20 if matrix.shape[1] > 1 else 0
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=rotation, ha="right")
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
    if "proxy_alias_of" in frame:
        frame = frame[frame["proxy_alias_of"].isna()]
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
        values = pd.to_numeric(group[field], errors="coerce").dropna()
        return float(values.median()) if not values.empty else float("nan")

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


def sensitivity_rank_table(
    source: ScoreSource,
    *,
    parameter: str = "seed",
) -> pd.DataFrame:
    """Measure cross-condition proxy rank stability after architecture-ID alignment."""
    frame = _direction_adjusted(read_scores(source)).dropna(
        subset=["architecture_id", "proxy_id", "component", "score"]
    )
    if parameter not in frame or frame[parameter].dropna().nunique() < 2:
        raise ValueError(f"Sensitivity rank analysis requires at least two {parameter} values")
    proxy_fields = ["proxy_id", "component"]
    if "proxy_version" in frame and frame["proxy_version"].notna().any():
        proxy_fields.append("proxy_version")
    proxy_fields.extend(
        field
        for field in PROXY_METADATA_FIELDS
        if field in frame and frame[field].notna().any()
    )
    excluded = {parameter}
    if parameter == "seed":
        excluded.add("input_fingerprint")
    protocol_fields = tuple(
        field
        for field in protocol_group_fields(frame, ())
        if field not in excluded
    )
    group_fields = tuple(dict.fromkeys((*proxy_fields, *protocol_fields)))
    records: list[dict[str, Any]] = []
    groups = frame.groupby(list(group_fields), dropna=False, sort=True)
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        metadata = dict(zip(group_fields, keys, strict=True))
        values = sorted(group[parameter].dropna().unique(), key=lambda value: str(value))
        for left_value, right_value in combinations(values, 2):
            left = group[group[parameter] == left_value][["architecture_id", "score"]]
            right = group[group[parameter] == right_value][["architecture_id", "score"]]
            for value, side in ((left_value, left), (right_value, right)):
                duplicates = side["architecture_id"].duplicated(keep=False)
                if duplicates.any():
                    example = side.loc[duplicates, "architecture_id"].astype(str).iloc[0]
                    raise ValueError(
                        f"Duplicate architecture ID for {parameter}={value}: {example}"
                    )
            paired = left.merge(
                right,
                on="architecture_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_left", "_right"),
            )
            left_score, right_score = _paired_values(
                paired["score_left"], paired["score_right"]
            )
            union_count = len(set(left["architecture_id"]) | set(right["architecture_id"]))
            records.append(
                {
                    **metadata,
                    "sensitivity_parameter": parameter,
                    "value_left": left_value,
                    "value_right": right_value,
                    "left_count": int(len(left)),
                    "right_count": int(len(right)),
                    "common_count": int(len(paired)),
                    "union_count": int(union_count),
                    "common_coverage": len(paired) / union_count if union_count else 0.0,
                    "spearman": _correlation(left_score, right_score, "spearman"),
                    "kendall_tau_b": _correlation(
                        left_score, right_score, "kendall_tau_b"
                    ),
                    "pearson": _correlation(left_score, right_score, "pearson"),
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


def plot_sensitivity_rank(
    source: ScoreSource | pd.DataFrame,
    destination: PathLike | None = None,
    *,
    parameter: str = "seed",
) -> Any:
    import matplotlib.pyplot as plt

    table = (
        source.copy()
        if isinstance(source, pd.DataFrame)
        and {"value_left", "value_right", "spearman"}.issubset(source.columns)
        else sensitivity_rank_table(source, parameter=parameter)
    )
    if table.empty:
        raise ValueError("Cannot plot empty sensitivity rank data")
    table["proxy"] = table["proxy_id"].astype(str) + " / " + table["component"].astype(str)
    pairs = list(table.groupby(["value_left", "value_right"], dropna=False, sort=True))
    proxies = sorted(table["proxy"].unique())
    x = np.arange(len(proxies), dtype=float)
    width = min(0.8 / max(len(pairs), 1), 0.35)
    figure, axis = plt.subplots(figsize=(max(7, 0.7 * len(proxies)), 5))
    for pair_index, ((left, right), group) in enumerate(pairs):
        values = group.set_index("proxy").reindex(proxies)["spearman"]
        offset = (pair_index - (len(pairs) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=f"{left} ↔ {right}")
    axis.set_xticks(x, proxies, rotation=35, ha="right")
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Cross-condition score Spearman")
    axis.set_title(f"Rank stability across {parameter}")
    axis.grid(axis="y", alpha=0.25)
    if len(pairs) > 1:
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
    metrics = [
        field
        for field in (
            "train_top1",
            "valid_top1",
            "train_top5",
            "valid_top5",
            "train_loss",
            "valid_loss",
            "learning_rate",
            "drop_path_prob",
            "duration_seconds",
            "peak_memory_mb",
        )
        if field in frame
    ]
    if not metrics:
        raise ValueError("Training data contains no supported metrics")
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    for field in metrics:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["epoch"]).sort_values("epoch")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    accuracy_axis, loss_axis, schedule_axis, resource_axis = axes.ravel()
    memory_axis = resource_axis.twinx()
    plotted = False
    for field in metrics:
        values = frame[["epoch", field]].dropna()
        if values.empty:
            continue
        if "top" in field:
            axis = accuracy_axis
        elif "loss" in field:
            axis = loss_axis
        elif field in {"learning_rate", "drop_path_prob"}:
            axis = schedule_axis
        elif field == "peak_memory_mb":
            axis = memory_axis
        else:
            axis = resource_axis
        axis.plot(values["epoch"], values[field], marker="o", label=field)
        plotted = True
    accuracy_axis.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    loss_axis.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    schedule_axis.set(xlabel="Epoch", ylabel="Value", title="Schedule")
    resource_axis.set(xlabel="Epoch", ylabel="Duration (s)", title="Resources")
    memory_axis.set(ylabel="Peak allocated memory (MiB)")
    for axis in (*axes.ravel(), memory_axis):
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
    sample_sizes: Sequence[int] | None = None,
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
        if sample_sizes is not None:
            convergence_csv = output / "sample_size_convergence.csv"
            _write_csv(
                sample_size_convergence(paired, sizes=sample_sizes),
                convergence_csv,
            )
            artifacts.append(convergence_csv)
        correlations = correlation_table(frame, bootstrap_samples=bootstrap_samples)
        correlation_csv = output / "correlations.csv"
        _write_csv(correlations, correlation_csv)
        artifacts.append(correlation_csv)
        top_k_table = top_k_comparison(paired, k=top_k)
        top_k_csv = output / "top_k.csv"
        _write_csv(top_k_table, top_k_csv)
        artifacts.append(top_k_csv)
        for name, table in (
            ("rank_aggregation", rank_aggregation(paired)),
            ("proxy_cost_pareto", proxy_cost_pareto(paired)),
            ("transfer", transfer_correlation_table(paired)),
        ):
            path = output / f"{name}.csv"
            _write_csv(table, path)
            artifacts.append(path)
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
            if paired[sensitivity_parameter].dropna().nunique() > 1:
                rank_stability = sensitivity_rank_table(
                    paired, parameter=sensitivity_parameter
                )
                rank_stability_csv = output / "sensitivity_rank.csv"
                _write_csv(rank_stability, rank_stability_csv)
                artifacts.append(rank_stability_csv)
                plots["sensitivity_rank"] = lambda destination: plot_sensitivity_rank(
                    rank_stability,
                    destination,
                    parameter=sensitivity_parameter,
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
    "plot_sensitivity_rank",
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
    "sensitivity_rank_table",
    "transfer_correlation_table",
    "training_plot",
]
