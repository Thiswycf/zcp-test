from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


_TOPOLOGY_EDGE = re.compile(r"([^|+~]+)~(\d+)")
_VALID_DIRECTIONS = {"maximize", "minimize"}
_VIT_SPACES = {"autoformer", "pit"}
_CORRELATION_METHODS = {"spearman", "kendall", "kendall_tau_b", "pearson"}


def _require_frame(frame: pd.DataFrame, required: Sequence[str], study: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{study} requires a pandas DataFrame")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{study} requires fields: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{study} requires a non-empty DataFrame")


def _require_complete(frame: pd.DataFrame, columns: Sequence[str], study: str) -> None:
    incomplete = [column for column in columns if frame[column].isna().any()]
    if incomplete:
        raise ValueError(f"{study} does not allow null fields: {', '.join(incomplete)}")


def _architecture_value(value: Any, column: str) -> Any:
    if isinstance(value, Mapping):
        if column in value:
            return value[column]
        if "spec" in value and isinstance(value["spec"], Mapping):
            return value["spec"]
    return value


def _architecture_string(value: Any, column: str, row_label: Any, study: str) -> str:
    value = _architecture_value(value, column)
    if isinstance(value, Mapping) and "architecture" in value:
        value = value["architecture"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{study} row {row_label!r} has no valid architecture string")
    return value.strip()


def _unique_architectures(
    frame: pd.DataFrame, architecture_column: str, study: str
) -> pd.DataFrame:
    identity = "architecture_id" if "architecture_id" in frame else architecture_column
    columns = [identity] if identity == architecture_column else [identity, architecture_column]
    architectures = frame.loc[:, columns].copy()
    if identity != architecture_column:
        grouped = architectures.groupby(identity, dropna=False)[architecture_column]
        inconsistent = [key for key, values in grouped if values.map(repr).nunique() != 1]
        if inconsistent:
            raise ValueError(
                f"{study} architecture_id values map to multiple specifications: {inconsistent[:3]}"
            )
    return architectures.drop_duplicates(identity, keep="first").reset_index(drop=True)


def _safe_feature_name(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return result or "unnamed"


def _validate_methods(methods: Sequence[str], study: str) -> None:
    if not methods:
        raise ValueError(f"{study} requires at least one correlation method")
    invalid = sorted(set(methods) - _CORRELATION_METHODS)
    if invalid:
        raise ValueError(f"{study} has unknown correlation methods: {invalid}")


def _append_features(
    frame: pd.DataFrame, records: list[dict[str, Any]], study: str
) -> pd.DataFrame:
    features = pd.DataFrame.from_records(records, index=frame.index)
    collisions = sorted(set(frame.columns) & set(features.columns))
    if collisions:
        raise ValueError(f"{study} feature columns already exist: {', '.join(collisions)}")
    return pd.concat([frame.copy(), features], axis=1)


def _parse_topology(value: str, row_label: Any) -> list[dict[str, Any]]:
    nodes = value.split("+")
    if not nodes or any(not node for node in nodes):
        raise ValueError(f"NB201/NATS-TSS row {row_label!r} has malformed topology {value!r}")
    edges: list[dict[str, Any]] = []
    for destination, node in enumerate(nodes, start=1):
        tokens = [token for token in node.split("|") if token]
        matches = [_TOPOLOGY_EDGE.fullmatch(token) for token in tokens]
        if not tokens or any(match is None for match in matches):
            raise ValueError(f"NB201/NATS-TSS row {row_label!r} has malformed node {node!r}")
        sources = [int(match.group(2)) for match in matches if match is not None]
        if sources != list(range(destination)):
            raise ValueError(
                f"NB201/NATS-TSS row {row_label!r} node {destination} must contain "
                f"sources 0..{destination - 1} in order"
            )
        for match in matches:
            assert match is not None
            edges.append(
                {
                    "source_node": int(match.group(2)),
                    "destination_node": destination,
                    "edge": f"{match.group(2)}->{destination}",
                    "operation": match.group(1),
                }
            )
    return edges


def _topology_features(
    frame: pd.DataFrame, *, architecture_column: str = "architecture"
) -> pd.DataFrame:
    """Add numeric architecture, edge, and operation features for NB201/NATS-TSS."""
    study = "NB201/NATS-TSS topology study"
    _require_frame(frame, [architecture_column], study)
    _require_complete(frame, [architecture_column], study)
    parsed: list[list[dict[str, Any]]] = []
    operations: set[str] = set()
    for row_label, value in frame[architecture_column].items():
        topology = _architecture_string(value, architecture_column, row_label, study)
        edges = _parse_topology(topology, row_label)
        parsed.append(edges)
        operations.update(edge["operation"] for edge in edges)

    records: list[dict[str, Any]] = []
    for edges in parsed:
        operation_counts = pd.Series([edge["operation"] for edge in edges]).value_counts()
        record: dict[str, Any] = {
            "topology_node_count": max(edge["destination_node"] for edge in edges) + 1,
            "topology_intermediate_node_count": max(
                edge["destination_node"] for edge in edges
            ),
            "topology_edge_count": len(edges),
            "topology_unique_operation_count": int(operation_counts.size),
        }
        for operation in sorted(operations):
            name = _safe_feature_name(operation)
            record[f"op_count__{name}"] = int(operation_counts.get(operation, 0))
            record[f"op_fraction__{name}"] = float(operation_counts.get(operation, 0) / len(edges))
        for edge in edges:
            edge_name = edge["edge"].replace("->", "_")
            for operation in sorted(operations):
                record[f"edge_{edge_name}__{_safe_feature_name(operation)}"] = int(
                    edge["operation"] == operation
                )
        records.append(record)
    return _append_features(frame, records, study)


def topology_study(
    frame: pd.DataFrame, *, architecture_column: str = "architecture"
) -> dict[str, pd.DataFrame]:
    """Return unique-architecture features plus long edge and operation summaries."""
    study = "NB201/NATS-TSS topology study"
    _require_frame(frame, [architecture_column], study)
    architectures = _unique_architectures(frame, architecture_column, study)
    features = _topology_features(architectures, architecture_column=architecture_column)
    identity = "architecture_id" if "architecture_id" in architectures else architecture_column
    edge_records: list[dict[str, Any]] = []
    for row_label, row in architectures.iterrows():
        topology = _architecture_string(
            row[architecture_column], architecture_column, row_label, study
        )
        for edge in _parse_topology(topology, row_label):
            edge_records.append({identity: row[identity], **edge})
    edges = pd.DataFrame.from_records(edge_records)
    architecture_count = architectures[identity].nunique(dropna=False)
    operations = (
        edges.groupby("operation", as_index=False, sort=True)
        .agg(edge_count=("edge", "size"), architecture_count=(identity, "nunique"))
        .assign(
            edge_fraction=lambda table: table["edge_count"] / len(edges),
            architecture_fraction=lambda table: table["architecture_count"] / architecture_count,
        )
    )
    result = {"architectures": features, "edges": edges, "operations": operations}
    research_fields = {"proxy_id", "component", "score", "target_value", "direction"}
    if research_fields.issubset(frame.columns):
        detailed = _topology_features(frame, architecture_column=architecture_column)
        feature_columns = [
            column
            for column in detailed
            if column.startswith(("op_count__", "op_fraction__", "edge_"))
        ]
        result["correlations"] = _numeric_feature_correlations(
            detailed,
            feature_columns=feature_columns,
            study=study,
        )
        result["operation_effects"] = _topology_operation_effects(
            frame, architecture_column=architecture_column
        )
    return result


def _parse_nats_size(value: str, row_label: Any) -> list[int]:
    parts = value.split(":")
    if not parts or any(not part or not part.isdecimal() for part in parts):
        raise ValueError(f"NATS-SSS row {row_label!r} has malformed size string {value!r}")
    channels = [int(part) for part in parts]
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"NATS-SSS row {row_label!r} channels must be positive")
    return channels


def _nats_size_features(
    frame: pd.DataFrame, *, architecture_column: str = "architecture"
) -> pd.DataFrame:
    """Add per-stage channels and aggregate size features for NATS-SSS."""
    study = "NATS-SSS size study"
    _require_frame(frame, [architecture_column], study)
    _require_complete(frame, [architecture_column], study)
    parsed = [
        _parse_nats_size(
            _architecture_string(value, architecture_column, row_label, study), row_label
        )
        for row_label, value in frame[architecture_column].items()
    ]
    stage_count = len(parsed[0])
    if any(len(channels) != stage_count for channels in parsed):
        raise ValueError("NATS-SSS size study requires the same stage count in every row")
    records: list[dict[str, Any]] = []
    for channels in parsed:
        array = np.asarray(channels, dtype=float)
        record: dict[str, Any] = {
            "size_stage_count": len(channels),
            "size_channel_sum": int(array.sum()),
            "size_channel_mean": float(array.mean()),
            "size_channel_min": int(array.min()),
            "size_channel_max": int(array.max()),
            "size_channel_range": int(array.max() - array.min()),
            "size_channel_std": float(array.std(ddof=0)),
            "size_first_channel": channels[0],
            "size_last_channel": channels[-1],
            "size_expansion_ratio": float(channels[-1] / channels[0]),
            "size_increase_count": int(np.count_nonzero(np.diff(array) > 0)),
            "size_decrease_count": int(np.count_nonzero(np.diff(array) < 0)),
        }
        record.update({f"stage_{stage}_channel": channel for stage, channel in enumerate(channels)})
        record.update(
            {
                f"stage_{stage}_channel_delta": channels[stage] - channels[stage - 1]
                for stage in range(1, len(channels))
            }
        )
        records.append(record)
    return _append_features(frame, records, study)


def nats_size_study(
    frame: pd.DataFrame, *, architecture_column: str = "architecture"
) -> dict[str, pd.DataFrame]:
    """Return unique-architecture, stage/channel, and numeric size summaries."""
    study = "NATS-SSS size study"
    _require_frame(frame, [architecture_column], study)
    architectures = _unique_architectures(frame, architecture_column, study)
    features = _nats_size_features(architectures, architecture_column=architecture_column)
    identity = "architecture_id" if "architecture_id" in architectures else architecture_column
    stage_columns = sorted(
        (column for column in features if re.fullmatch(r"stage_\d+_channel", column)),
        key=lambda column: int(column.split("_")[1]),
    )
    stages = features.melt(
        id_vars=[identity],
        value_vars=stage_columns,
        var_name="stage",
        value_name="channel",
    )
    stages["stage"] = stages["stage"].str.extract(r"(\d+)", expand=False).astype(int)
    size_columns = [column for column in features if column.startswith("size_")]
    summary = features[size_columns].describe().T.reset_index(names="feature")
    result = {"architectures": features, "stages": stages, "summary": summary}
    research_fields = {"proxy_id", "component", "score", "target_value", "direction"}
    if research_fields.issubset(frame.columns):
        detailed = _nats_size_features(frame, architecture_column=architecture_column)
        feature_columns = [
            column
            for column in detailed
            if column.startswith("size_") or re.fullmatch(r"stage_\d+_channel(?:_delta)?", column)
        ]
        correlations = _numeric_feature_correlations(
            detailed,
            feature_columns=feature_columns,
            study=study,
        )
        result["correlations"] = correlations
        result["stage_sensitivity"] = correlations[
            correlations["feature"].str.match(r"stage_\d+_channel$")
        ].reset_index(drop=True)
    return result


def _vit_spec(value: Any, row_label: Any, architecture_column: str) -> Mapping[str, Any]:
    value = _architecture_value(value, architecture_column)
    if not isinstance(value, Mapping):
        raise ValueError(f"ViT structure study row {row_label!r} architecture must be a mapping")
    return value


def _numeric_sequence(value: Any, field: str, row_label: Any) -> list[float]:
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    invalid = any(isinstance(item, bool) or not isinstance(item, Real) for item in values)
    if not values or invalid:
        raise ValueError(f"ViT structure study row {row_label!r} has invalid {field}")
    result = [float(item) for item in values]
    if any(not math.isfinite(item) or item <= 0 for item in result):
        raise ValueError(f"ViT structure study row {row_label!r} has non-positive {field}")
    return result


def _vit_architecture_features(
    frame: pd.DataFrame,
    *,
    architecture_column: str = "architecture",
    space_column: str = "search_space_id",
) -> pd.DataFrame:
    """Extract numeric AutoFormer/PiT structure features from architecture mappings."""
    study = "ViT structure study"
    _require_frame(frame, [architecture_column, space_column], study)
    _require_complete(frame, [architecture_column, space_column], study)
    records: list[dict[str, Any]] = []
    for row_label, row in frame.iterrows():
        space = str(row[space_column]).casefold()
        if space not in _VIT_SPACES:
            raise ValueError(
                f"ViT structure study row {row_label!r} has unsupported space {space!r}"
            )
        specification = _vit_spec(row[architecture_column], row_label, architecture_column)
        dimension_field = "hidden_dim" if space == "autoformer" else "base_dim"
        required = {"depth", dimension_field, "num_heads", "mlp_ratio"}
        missing = sorted(required - set(specification))
        if missing:
            raise ValueError(
                f"ViT structure study row {row_label!r} is missing: {', '.join(missing)}"
            )
        depths = _numeric_sequence(specification["depth"], "depth", row_label)
        heads = _numeric_sequence(specification["num_heads"], "num_heads", row_label)
        ratios = _numeric_sequence(specification["mlp_ratio"], "mlp_ratio", row_label)
        dimension = _numeric_sequence(specification[dimension_field], dimension_field, row_label)
        if any(not value.is_integer() for value in depths):
            raise ValueError(f"ViT structure study row {row_label!r} depth must be integral")
        if any(not value.is_integer() for value in heads):
            raise ValueError(f"ViT structure study row {row_label!r} num_heads must be integral")
        if len(dimension) != 1:
            raise ValueError(
                f"ViT structure study row {row_label!r} requires scalar {dimension_field}"
            )
        total_depth = int(sum(depths))
        valid_lengths = {1, total_depth, len(depths)}
        if len(heads) not in valid_lengths or len(ratios) not in valid_lengths:
            raise ValueError(
                f"ViT structure study row {row_label!r} heads/ratios must be scalar, per-stage, "
                "or per-layer"
            )
        record: dict[str, Any] = {
            "vit_depth": total_depth,
            "vit_stage_count": len(depths),
            "vit_dimension": dimension[0],
            "vit_heads_mean": float(np.mean(heads)),
            "vit_heads_min": float(np.min(heads)),
            "vit_heads_max": float(np.max(heads)),
            "vit_heads_std": float(np.std(heads, ddof=0)),
            "vit_mlp_ratio_mean": float(np.mean(ratios)),
            "vit_mlp_ratio_min": float(np.min(ratios)),
            "vit_mlp_ratio_max": float(np.max(ratios)),
            "vit_mlp_ratio_std": float(np.std(ratios, ddof=0)),
            "vit_attention_width_mean": float(dimension[0] * np.mean(heads)),
            "vit_mlp_width_mean": float(dimension[0] * np.mean(ratios)),
        }
        record.update({f"vit_stage_{index}_depth": value for index, value in enumerate(depths)})
        record.update({f"vit_head_{index}": value for index, value in enumerate(heads)})
        record.update({f"vit_mlp_ratio_{index}": value for index, value in enumerate(ratios)})
        records.append(record)
    return _append_features(frame, records, study)


def _correlation(left: pd.Series, right: pd.Series, method: str) -> tuple[int, float | None]:
    paired = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return len(paired), None
    if method == "spearman":
        value = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic
    elif method in {"kendall", "kendall_tau_b"}:
        value = stats.kendalltau(paired.iloc[:, 0], paired.iloc[:, 1], variant="b").statistic
    elif method == "pearson":
        value = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return len(paired), float(value) if np.isfinite(value) else None


def _numeric_feature_correlations(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    study: str,
    target_column: str = "target_value",
    score_column: str = "score",
    direction_column: str = "direction",
    group_by: Sequence[str] = ("proxy_id", "component"),
    methods: Sequence[str] = ("spearman", "kendall_tau_b", "pearson"),
) -> pd.DataFrame:
    required = [target_column, score_column, direction_column, *group_by]
    _require_frame(frame, required, study)
    _require_complete(frame, [direction_column, *group_by], study)
    _validate_methods(methods, study)
    directions = frame[direction_column].astype(str).str.casefold()
    invalid = sorted(set(directions) - _VALID_DIRECTIONS)
    if invalid:
        raise ValueError(f"{study} has invalid score directions: {invalid}")
    working = frame.assign(
        _direction_adjusted_score=np.where(
            directions.eq("minimize"),
            -pd.to_numeric(frame[score_column], errors="coerce"),
            pd.to_numeric(frame[score_column], errors="coerce"),
        )
    )
    records: list[dict[str, Any]] = []
    for key, group in working.groupby(list(group_by), dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        identifiers = dict(zip(group_by, keys, strict=True))
        for feature in feature_columns:
            for outcome, outcome_column in (
                ("target", target_column),
                ("score", "_direction_adjusted_score"),
            ):
                for method in methods:
                    sample_count, value = _correlation(
                        group[feature], group[outcome_column], method
                    )
                    records.append(
                        {
                            **identifiers,
                            "feature": feature,
                            "outcome": outcome,
                            "method": method,
                            "sample_count": sample_count,
                            "correlation": value,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _topology_operation_effects(
    frame: pd.DataFrame,
    *,
    architecture_column: str = "architecture",
) -> pd.DataFrame:
    study = "NB201/NATS-TSS topology operation effect study"
    required = [
        architecture_column,
        "architecture_id",
        "proxy_id",
        "component",
        "score",
        "target_value",
        "direction",
    ]
    _require_frame(frame, required, study)
    _require_complete(
        frame,
        [architecture_column, "architecture_id", "proxy_id", "component", "direction"],
        study,
    )
    directions = frame["direction"].astype(str).str.casefold()
    invalid = sorted(set(directions) - _VALID_DIRECTIONS)
    if invalid:
        raise ValueError(f"{study} has invalid score directions: {invalid}")
    records: list[dict[str, Any]] = []
    for (row_label, row), direction in zip(frame.iterrows(), directions, strict=True):
        topology = _architecture_string(
            row[architecture_column], architecture_column, row_label, study
        )
        adjusted_score = float(row["score"])
        if direction == "minimize":
            adjusted_score = -adjusted_score
        for edge in _parse_topology(topology, row_label):
            records.append(
                {
                    "architecture_id": row["architecture_id"],
                    "proxy_id": row["proxy_id"],
                    "component": row["component"],
                    **edge,
                    "target_value": pd.to_numeric(row["target_value"], errors="coerce"),
                    "adjusted_score": adjusted_score,
                }
            )
    detailed = pd.DataFrame.from_records(records)
    groups = ["proxy_id", "component", "edge", "operation"]
    effects = (
        detailed.groupby(groups, as_index=False, dropna=False, sort=True)
        .agg(
            sample_count=("architecture_id", "nunique"),
            target_mean=("target_value", "mean"),
            target_median=("target_value", "median"),
            score_mean=("adjusted_score", "mean"),
            score_median=("adjusted_score", "median"),
        )
    )
    baselines = (
        detailed.groupby(["proxy_id", "component", "edge"], as_index=False, dropna=False)
        .agg(edge_target_mean=("target_value", "mean"), edge_score_mean=("adjusted_score", "mean"))
    )
    effects = effects.merge(
        baselines, on=["proxy_id", "component", "edge"], validate="many_to_one"
    )
    effects["target_delta_from_edge_mean"] = (
        effects["target_mean"] - effects["edge_target_mean"]
    )
    effects["score_delta_from_edge_mean"] = effects["score_mean"] - effects["edge_score_mean"]
    return effects


def _vit_feature_correlations(
    frame: pd.DataFrame,
    *,
    architecture_column: str = "architecture",
    space_column: str = "search_space_id",
    target_column: str = "target_value",
    score_column: str = "score",
    direction_column: str = "direction",
    group_by: Sequence[str] = ("search_space_id", "proxy_id", "component"),
    methods: Sequence[str] = ("spearman", "kendall_tau_b", "pearson"),
) -> pd.DataFrame:
    """Correlate numeric ViT features with targets and direction-adjusted scores."""
    study = "ViT feature correlation study"
    required = [
        architecture_column,
        space_column,
        target_column,
        score_column,
        direction_column,
        *group_by,
    ]
    _require_frame(frame, required, study)
    _require_complete(frame, [architecture_column, space_column, direction_column], study)
    _validate_methods(methods, study)
    directions = frame[direction_column].astype(str).str.casefold()
    invalid = sorted(set(directions) - _VALID_DIRECTIONS)
    if invalid:
        raise ValueError(f"ViT feature correlation study has invalid directions: {invalid}")
    featured = _vit_architecture_features(
        frame, architecture_column=architecture_column, space_column=space_column
    )
    featured = featured.assign(
        _direction_adjusted_score=np.where(
            directions.eq("minimize"),
            -pd.to_numeric(featured[score_column], errors="coerce"),
            pd.to_numeric(featured[score_column], errors="coerce"),
        )
    )
    feature_columns = [column for column in featured if column.startswith("vit_")]
    records: list[dict[str, Any]] = []
    groups = (
        featured.groupby(list(group_by), dropna=False, sort=True)
        if group_by
        else [((), featured)]
    )
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        identifiers = dict(zip(group_by, keys, strict=True))
        for feature in feature_columns:
            for outcome, outcome_column in (
                ("target", target_column),
                ("score", "_direction_adjusted_score"),
            ):
                for method in methods:
                    sample_count, value = _correlation(
                        group[feature], group[outcome_column], method
                    )
                    records.append(
                        {
                            **identifiers,
                            "feature": feature,
                            "outcome": outcome,
                            "method": method,
                            "sample_count": sample_count,
                            "correlation": value,
                        }
                    )
    return pd.DataFrame.from_records(records)


def vit_architecture_study(
    frame: pd.DataFrame,
    *,
    architecture_column: str = "architecture",
    space_column: str = "search_space_id",
    target_column: str = "target_value",
    score_column: str = "score",
    direction_column: str = "direction",
    group_by: Sequence[str] = ("search_space_id", "proxy_id", "component"),
    methods: Sequence[str] = ("spearman", "kendall_tau_b", "pearson"),
) -> dict[str, pd.DataFrame]:
    """Return AutoFormer/PiT numeric features and feature/outcome correlations."""
    features = _vit_architecture_features(
        frame, architecture_column=architecture_column, space_column=space_column
    )
    correlations = _vit_feature_correlations(
        frame,
        architecture_column=architecture_column,
        space_column=space_column,
        target_column=target_column,
        score_column=score_column,
        direction_column=direction_column,
        group_by=group_by,
        methods=methods,
    )
    return {"features": features, "correlations": correlations}


def transnas_transfer_study(
    frame: pd.DataFrame,
    *,
    methods: Sequence[str] = ("spearman", "kendall_tau_b"),
) -> dict[str, pd.DataFrame]:
    """Summarize per-space/task proxy quality and cross-task transfer for TransNAS rows."""
    study = "TransNAS transfer study"
    keys = ["search_space_id", "dataset", "architecture_id", "proxy_id", "component"]
    required = [*keys, "target_value", "score", "direction"]
    _require_frame(frame, required, study)
    _require_complete(frame, [*keys, "direction"], study)
    _validate_methods(methods, study)
    directions = frame["direction"].astype(str).str.casefold()
    invalid = sorted(set(directions) - _VALID_DIRECTIONS)
    if invalid:
        raise ValueError(f"TransNAS transfer study has invalid directions: {invalid}")
    if frame.duplicated(keys, keep=False).any():
        raise ValueError(
            "TransNAS transfer study requires one row per space/task/architecture/proxy/component; "
            "filter the desired metric and split explicitly before calling it"
        )
    working = frame.assign(
        adjusted_score=np.where(
            directions.eq("minimize"),
            -pd.to_numeric(frame["score"], errors="coerce"),
            pd.to_numeric(frame["score"], errors="coerce"),
        ),
        numeric_target=pd.to_numeric(frame["target_value"], errors="coerce"),
    )
    group_columns = ["search_space_id", "dataset", "proxy_id", "component"]
    quality_records: list[dict[str, Any]] = []
    for key, group in working.groupby(group_columns, dropna=False, sort=True):
        identifiers = dict(zip(group_columns, key, strict=True))
        for method in methods:
            sample_count, value = _correlation(
                group["numeric_target"], group["adjusted_score"], method
            )
            quality_records.append(
                {
                    **identifiers,
                    "method": method,
                    "sample_count": sample_count,
                    "correlation": value,
                }
            )
    task_quality = pd.DataFrame.from_records(quality_records)

    transfer_records: list[dict[str, Any]] = []
    transfer_groups = ["search_space_id", "proxy_id", "component"]
    for key, group in working.groupby(transfer_groups, dropna=False, sort=True):
        identifiers = dict(zip(transfer_groups, key, strict=True))
        tasks = sorted(group["dataset"].unique(), key=str)
        for source_task in tasks:
            source = group[group["dataset"].eq(source_task)][
                ["architecture_id", "adjusted_score"]
            ]
            for target_task in tasks:
                target = group[group["dataset"].eq(target_task)][
                    ["architecture_id", "numeric_target"]
                ]
                paired = source.merge(target, on="architecture_id", validate="one_to_one")
                for method in methods:
                    sample_count, value = _correlation(
                        paired["adjusted_score"], paired["numeric_target"], method
                    )
                    transfer_records.append(
                        {
                            **identifiers,
                            "source_task": source_task,
                            "target_task": target_task,
                            "method": method,
                            "sample_count": sample_count,
                            "correlation": value,
                        }
                    )
    task_transfer = pd.DataFrame.from_records(transfer_records)
    space_summary = (
        task_quality.groupby(["search_space_id", "proxy_id", "component", "method"], as_index=False)
        .agg(
            task_count=("dataset", "nunique"),
            valid_task_count=("correlation", "count"),
            mean_correlation=("correlation", "mean"),
            median_correlation=("correlation", "median"),
            min_correlation=("correlation", "min"),
            max_correlation=("correlation", "max"),
        )
    )
    return {
        "task_quality": task_quality,
        "task_transfer": task_transfer,
        "space_summary": space_summary,
    }


__all__ = [
    "nats_size_study",
    "topology_study",
    "transnas_transfer_study",
    "vit_architecture_study",
]
