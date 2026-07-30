from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


_TOPOLOGY_EDGE = re.compile(r"([^|+~]+)~(\d+)")
_TRANSNAS_ARCHITECTURE = re.compile(r"^(\d+)-([1-4]+)-(basic|[0-3]_[0-3]{2}_[0-3]{3})$")
_VALID_DIRECTIONS = {"maximize", "minimize"}
_VIT_SPACES = {"autoformer", "pit"}
_CORRELATION_METHODS = {"spearman", "kendall", "kendall_tau_b", "pearson"}
_STUDY_PROTOCOL_FIELDS = (
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
    "target_seed",
    "target_seed_reduction",
    "proxy_version",
    "model_fidelity",
    "input_source",
    "input_fingerprint",
    "run_id",
    "source_run",
)


def _study_group_fields(frame: pd.DataFrame, base: Sequence[str]) -> tuple[str, ...]:
    fields = list(base)
    for field in _STUDY_PROTOCOL_FIELDS:
        if field in frame and frame[field].notna().any() and field not in fields:
            fields.append(field)
    return tuple(fields)


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
        def stable_spec(value: Any) -> str:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

        inconsistent = [
            key for key, values in grouped if values.map(stable_spec).nunique() != 1
        ]
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
        matched_pairs, matched_summary = _topology_matched_pairs(
            frame, architecture_column=architecture_column
        )
        result["matched_pairs"] = matched_pairs
        result["matched_pair_summary"] = matched_summary
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
        result["size_controlled_correlations"] = _size_controlled_correlations(detailed)
        result["size_strata"] = _size_stratified_proxy_quality(detailed)
    return result


def _rank_residual(values: pd.Series, control: pd.Series) -> np.ndarray:
    ranked = pd.to_numeric(values, errors="coerce").rank(method="average").to_numpy(float)
    ranked_control = (
        pd.to_numeric(control, errors="coerce").rank(method="average").to_numpy(float)
    )
    design = np.column_stack([np.ones(len(ranked_control)), ranked_control])
    coefficients, *_ = np.linalg.lstsq(design, ranked, rcond=None)
    return ranked - design @ coefficients


def _size_controlled_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Partial Spearman correlations controlling the total channel sum."""
    study = "NATS-SSS size-controlled study"
    group_fields = _study_group_fields(frame, ("proxy_id", "component"))
    feature_columns = sorted(
        column
        for column in frame
        if re.fullmatch(r"stage_\d+_channel(?:_delta)?", column)
        or column in {"size_expansion_ratio", "size_channel_range", "size_channel_std"}
    )
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_fields), dropna=False, sort=True):
        identifiers = dict(
            zip(group_fields, key if isinstance(key, tuple) else (key,), strict=True)
        )
        direction = group["direction"].astype(str).str.casefold()
        if not set(direction).issubset(_VALID_DIRECTIONS):
            raise ValueError(f"{study} has invalid score directions")
        score = np.where(
            direction.eq("minimize"),
            -pd.to_numeric(group["score"], errors="coerce"),
            pd.to_numeric(group["score"], errors="coerce"),
        )
        outcomes = {
            "target": pd.to_numeric(group["target_value"], errors="coerce"),
            "score": pd.Series(score, index=group.index),
        }
        control = pd.to_numeric(group["size_channel_sum"], errors="coerce")
        for feature in feature_columns:
            for outcome_name, outcome in outcomes.items():
                finite = np.isfinite(control) & np.isfinite(outcome) & np.isfinite(group[feature])
                sample = group.loc[finite]
                if len(sample) < 3:
                    value = None
                else:
                    left = _rank_residual(sample[feature], control.loc[finite])
                    right = _rank_residual(outcome.loc[finite], control.loc[finite])
                    value = (
                        float(stats.pearsonr(left, right).statistic)
                        if np.std(left) > 0 and np.std(right) > 0
                        else None
                    )
                records.append(
                    {
                        **identifiers,
                        "feature": feature,
                        "outcome": outcome_name,
                        "control": "size_channel_sum",
                        "method": "partial_spearman",
                        "sample_count": int(finite.sum()),
                        "correlation": value,
                    }
                )
    return pd.DataFrame.from_records(records)


def _size_stratified_proxy_quality(frame: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """Measure proxy quality within total-size quantiles to expose scale confounding."""
    group_fields = _study_group_fields(frame, ("proxy_id", "component"))
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_fields), dropna=False, sort=True):
        identifiers = dict(
            zip(group_fields, key if isinstance(key, tuple) else (key,), strict=True)
        )
        working = group.copy()
        unique_sizes = working["size_channel_sum"].nunique()
        if unique_sizes < 2:
            working["size_stratum"] = "all"
        else:
            working["size_stratum"] = pd.qcut(
                working["size_channel_sum"], q=min(bins, unique_sizes), duplicates="drop"
            ).astype(str)
        direction = working["direction"].astype(str).str.casefold()
        working["adjusted_score"] = np.where(
            direction.eq("minimize"), -working["score"], working["score"]
        )
        for stratum, sample in working.groupby("size_stratum", sort=True):
            sample_count, value = _correlation(
                sample["target_value"], sample["adjusted_score"], "spearman"
            )
            records.append(
                {
                    **identifiers,
                    "size_stratum": stratum,
                    "size_min": sample["size_channel_sum"].min(),
                    "size_max": sample["size_channel_sum"].max(),
                    "sample_count": sample_count,
                    "spearman": value,
                }
            )
    return pd.DataFrame.from_records(records)


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


def _expand_vit_values(values: Sequence[float], depths: Sequence[int], field: str) -> list[float]:
    total_depth = sum(depths)
    if len(values) == 1:
        return [float(values[0])] * total_depth
    if len(values) == len(depths):
        return [float(value) for value, depth in zip(values, depths, strict=True) for _ in range(depth)]
    if len(values) == total_depth:
        return [float(value) for value in values]
    raise ValueError(f"ViT {field} must be scalar, per-stage, or per-layer")


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
        integer_depths = [int(value) for value in depths]
        total_depth = int(sum(integer_depths))
        valid_lengths = {1, total_depth, len(depths)}
        if len(heads) not in valid_lengths or len(ratios) not in valid_lengths:
            raise ValueError(
                f"ViT structure study row {row_label!r} heads/ratios must be scalar, per-stage, "
                "or per-layer"
            )
        expanded_heads = _expand_vit_values(heads, integer_depths, "num_heads")
        expanded_ratios = _expand_vit_values(ratios, integer_depths, "mlp_ratio")
        base_dimension = dimension[0]
        if space == "pit":
            expanded_embeddings = base_dimension * np.asarray(expanded_heads)
        else:
            expanded_embeddings = np.full(total_depth, base_dimension)
        record: dict[str, Any] = {
            "vit_depth": total_depth,
            "vit_stage_count": len(depths),
            "vit_dimension": float(np.mean(expanded_embeddings)),
            "vit_base_dimension": base_dimension,
            "vit_heads_mean": float(np.mean(expanded_heads)),
            "vit_heads_min": float(np.min(expanded_heads)),
            "vit_heads_max": float(np.max(expanded_heads)),
            "vit_heads_std": float(np.std(expanded_heads, ddof=0)),
            "vit_head_dimension_mean": float(
                np.mean(expanded_embeddings / np.asarray(expanded_heads))
            ),
            "vit_mlp_ratio_mean": float(np.mean(expanded_ratios)),
            "vit_mlp_ratio_min": float(np.min(expanded_ratios)),
            "vit_mlp_ratio_max": float(np.max(expanded_ratios)),
            "vit_mlp_ratio_std": float(np.std(expanded_ratios, ddof=0)),
            "vit_mlp_width_mean": float(np.mean(expanded_embeddings * expanded_ratios)),
            "vit_attention_parameter_proxy": float(4 * np.sum(expanded_embeddings**2)),
            "vit_mlp_parameter_proxy": float(
                2 * np.sum(expanded_embeddings**2 * expanded_ratios)
            ),
            "vit_block_parameter_proxy": float(
                4 * np.sum(expanded_embeddings**2)
                + 2 * np.sum(expanded_embeddings**2 * expanded_ratios)
            ),
        }
        record.update({f"vit_stage_{index}_depth": value for index, value in enumerate(depths)})
        record.update({f"vit_head_{index}": value for index, value in enumerate(heads)})
        record.update({f"vit_mlp_ratio_{index}": value for index, value in enumerate(ratios)})
        records.append(record)
    return _append_features(frame, records, study)


def _vit_layer_table(
    frame: pd.DataFrame,
    *,
    architecture_column: str,
    space_column: str,
) -> pd.DataFrame:
    study = "ViT layer study"
    architectures = _unique_architectures(frame, architecture_column, study)
    if space_column not in architectures:
        spaces = frame[["architecture_id", space_column]].drop_duplicates()
        architectures = architectures.merge(spaces, on="architecture_id", validate="one_to_one")
    records: list[dict[str, Any]] = []
    for row_label, row in architectures.iterrows():
        specification = _vit_spec(row[architecture_column], row_label, architecture_column)
        space = str(row[space_column]).casefold()
        dimension_field = "hidden_dim" if space == "autoformer" else "base_dim"
        depths = [int(value) for value in _numeric_sequence(specification["depth"], "depth", row_label)]
        heads = _expand_vit_values(
            _numeric_sequence(specification["num_heads"], "num_heads", row_label),
            depths,
            "num_heads",
        )
        ratios = _expand_vit_values(
            _numeric_sequence(specification["mlp_ratio"], "mlp_ratio", row_label),
            depths,
            "mlp_ratio",
        )
        base_dimension = float(specification[dimension_field])
        layer = 0
        for stage, depth in enumerate(depths):
            for stage_layer in range(depth):
                dimension = (
                    base_dimension * heads[layer] if space == "pit" else base_dimension
                )
                records.append(
                    {
                        "architecture_id": row["architecture_id"],
                        "search_space_id": space,
                        "stage": stage,
                        "stage_layer": stage_layer,
                        "layer": layer,
                        "dimension": dimension,
                        "base_dimension": base_dimension,
                        "num_heads": heads[layer],
                        "head_dimension": dimension / heads[layer],
                        "mlp_ratio": ratios[layer],
                        "mlp_width": dimension * ratios[layer],
                    }
                )
                layer += 1
    return pd.DataFrame.from_records(records)


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
    group_by = _study_group_fields(frame, group_by)
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
    identity_fields = [*group_by, "architecture_id"]
    if "architecture_id" in working and working.duplicated(identity_fields, keep=False).any():
        raise ValueError(
            f"{study} requires one row per protocol/proxy/component/architecture; "
            "filter or aggregate repeated runs explicitly"
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
    protocol_fields = _study_group_fields(frame, ("proxy_id", "component"))
    duplicate_fields = [*protocol_fields, "architecture_id"]
    if frame.duplicated(duplicate_fields, keep=False).any():
        raise ValueError(
            f"{study} requires one row per protocol/proxy/component/architecture"
        )
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
                    **{field: row[field] for field in protocol_fields},
                    **edge,
                    "target_value": pd.to_numeric(row["target_value"], errors="coerce"),
                    "adjusted_score": adjusted_score,
                }
            )
    detailed = pd.DataFrame.from_records(records)
    groups = [*protocol_fields, "edge", "operation"]
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
        detailed.groupby([*protocol_fields, "edge"], as_index=False, dropna=False)
        .agg(edge_target_mean=("target_value", "mean"), edge_score_mean=("adjusted_score", "mean"))
    )
    effects = effects.merge(
        baselines, on=[*protocol_fields, "edge"], validate="many_to_one"
    )
    effects["target_delta_from_edge_mean"] = (
        effects["target_mean"] - effects["edge_target_mean"]
    )
    effects["score_delta_from_edge_mean"] = effects["score_mean"] - effects["edge_score_mean"]
    return effects


def _topology_matched_pairs(
    frame: pd.DataFrame,
    *,
    architecture_column: str = "architecture",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare architectures that differ on exactly one topology edge.

    These are matched observational contrasts over the enumerated candidate set. They reduce,
    but do not eliminate, confounding and must not be interpreted as causal operation effects.
    """
    study = "NB201/NATS-TSS matched-pair study"
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
    protocol_fields = _study_group_fields(frame, ("proxy_id", "component"))
    duplicate_fields = [*protocol_fields, "architecture_id"]
    if frame.duplicated(duplicate_fields, keep=False).any():
        raise ValueError(f"{study} requires unique architecture rows within each protocol")

    parsed_records: list[dict[str, Any]] = []
    for row_label, row in frame.iterrows():
        edges = _parse_topology(
            _architecture_string(row[architecture_column], architecture_column, row_label, study),
            row_label,
        )
        direction = str(row["direction"]).casefold()
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"{study} has invalid score direction {direction!r}")
        target = float(row["target_value"])
        if str(row.get("target_direction", "maximize")).casefold() == "minimize":
            target = -target
        parsed_records.append(
            {
                **{field: row[field] for field in protocol_fields},
                "architecture_id": row["architecture_id"],
                "operations": tuple(edge["operation"] for edge in edges),
                "edges": tuple(edge["edge"] for edge in edges),
                "target_value": target,
                "adjusted_score": -float(row["score"])
                if direction == "minimize"
                else float(row["score"]),
            }
        )
    parsed = pd.DataFrame.from_records(parsed_records)
    pair_records: list[dict[str, Any]] = []
    for key, protocol in parsed.groupby(list(protocol_fields), dropna=False, sort=True):
        identifiers = dict(
            zip(protocol_fields, key if isinstance(key, tuple) else (key,), strict=True)
        )
        edge_count = len(protocol.iloc[0]["operations"])
        if any(len(value) != edge_count for value in protocol["operations"]):
            raise ValueError(f"{study} requires a fixed topology edge count")
        for edge_index in range(edge_count):
            contexts: dict[tuple[str, ...], list[pd.Series]] = {}
            for _, row in protocol.iterrows():
                operations = row["operations"]
                context = operations[:edge_index] + operations[edge_index + 1 :]
                contexts.setdefault(context, []).append(row)
            for rows in contexts.values():
                for left, right in combinations(sorted(rows, key=lambda item: str(item["architecture_id"])), 2):
                    left_operation = left["operations"][edge_index]
                    right_operation = right["operations"][edge_index]
                    if left_operation == right_operation:
                        continue
                    target_delta = float(right["target_value"] - left["target_value"])
                    score_delta = float(right["adjusted_score"] - left["adjusted_score"])
                    pair_records.append(
                        {
                            **identifiers,
                            "edge": left["edges"][edge_index],
                            "left_architecture_id": left["architecture_id"],
                            "right_architecture_id": right["architecture_id"],
                            "left_operation": left_operation,
                            "right_operation": right_operation,
                            "target_delta": target_delta,
                            "score_delta": score_delta,
                            "rank_concordant": bool(
                                target_delta != 0
                                and score_delta != 0
                                and np.sign(target_delta) == np.sign(score_delta)
                            ),
                            "target_tie": target_delta == 0,
                            "score_tie": score_delta == 0,
                        }
                    )
    pairs = pd.DataFrame.from_records(pair_records)
    summary_columns = [
        *protocol_fields,
        "edge",
        "left_operation",
        "right_operation",
        "pair_count",
        "strict_concordant_count",
        "strict_concordance_rate",
        "non_tied_pair_count",
        "target_tie_rate",
        "score_tie_rate",
        "mean_absolute_target_delta",
        "mean_absolute_score_delta",
    ]
    if pairs.empty:
        return pairs, pd.DataFrame(columns=summary_columns)
    summary = (
        pairs.assign(
            absolute_target_delta=pairs["target_delta"].abs(),
            absolute_score_delta=pairs["score_delta"].abs(),
            non_tied=~pairs["target_tie"] & ~pairs["score_tie"],
        )
        .groupby(
            [*protocol_fields, "edge", "left_operation", "right_operation"],
            as_index=False,
            dropna=False,
            sort=True,
        )
        .agg(
            pair_count=("left_architecture_id", "size"),
            strict_concordant_count=("rank_concordant", "sum"),
            non_tied_pair_count=("non_tied", "sum"),
            target_tie_rate=("target_tie", "mean"),
            score_tie_rate=("score_tie", "mean"),
            mean_absolute_target_delta=("absolute_target_delta", "mean"),
            mean_absolute_score_delta=("absolute_score_delta", "mean"),
        )
    )
    summary["strict_concordance_rate"] = (
        summary["strict_concordant_count"] / summary["non_tied_pair_count"].replace(0, np.nan)
    )
    return pairs, summary


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
    group_by = _study_group_fields(frame, group_by)
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
    layers = _vit_layer_table(
        frame, architecture_column=architecture_column, space_column=space_column
    )
    return {"features": features, "layers": layers, "correlations": correlations}


def _transnas_architecture_features(
    frame: pd.DataFrame, *, architecture_column: str = "architecture"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    study = "TransNAS architecture study"
    _require_frame(frame, [architecture_column, "architecture_id", "search_space_id"], study)
    module_names = {
        "1": "normal",
        "2": "channel_x2",
        "3": "resolution_div2",
        "4": "channel_x2_resolution_div2",
    }
    operation_names = {"0": "none", "1": "skip_connect", "2": "conv_1x1", "3": "conv_3x3"}
    architecture_rows = frame[
        ["architecture_id", "search_space_id", architecture_column]
    ].copy()
    architecture_rows["_canonical_architecture"] = architecture_rows[architecture_column].map(
        lambda value: json.dumps(value, sort_keys=True) if isinstance(value, Mapping) else str(value)
    )
    if architecture_rows.groupby(
        ["architecture_id", "search_space_id"], dropna=False
    )["_canonical_architecture"].nunique().gt(1).any():
        raise ValueError("TransNAS architecture IDs map to conflicting specifications")
    architecture_rows = architecture_rows.drop_duplicates(
        ["architecture_id", "search_space_id"]
    )
    feature_records: list[dict[str, Any]] = []
    factor_records: list[dict[str, Any]] = []
    for row_label, row in architecture_rows.iterrows():
        encoded = _architecture_string(
            row[architecture_column], architecture_column, row_label, study
        )
        match = _TRANSNAS_ARCHITECTURE.fullmatch(encoded)
        if match is None:
            raise ValueError(f"{study} row {row_label!r} has malformed encoding {encoded!r}")
        base_channel, macro_code, cell_code = match.groups()
        space = str(row["search_space_id"])
        expected_micro = cell_code != "basic"
        if expected_micro != space.endswith("micro"):
            raise ValueError(
                f"{study} row {row_label!r} encoding disagrees with search_space_id={space!r}"
            )
        feature: dict[str, Any] = {
            "architecture_id": row["architecture_id"],
            "search_space_id": space,
            "transnas_base_channel": int(base_channel),
            "transnas_module_count": len(macro_code),
            "transnas_width_double_count": sum(code in "24" for code in macro_code),
            "transnas_downsample_count": sum(code in "34" for code in macro_code),
            "transnas_joint_scale_count": macro_code.count("4"),
        }
        for code, name in module_names.items():
            feature[f"transnas_module_count__{name}"] = macro_code.count(code)
        for position, code in enumerate(macro_code):
            feature[f"transnas_module_{position}__{module_names[code]}"] = 1
            factor_records.append(
                {
                    "architecture_id": row["architecture_id"],
                    "search_space_id": space,
                    "factor_type": "macro_module",
                    "position": position,
                    "code": code,
                    "label": module_names[code],
                }
            )
        if expected_micro:
            operation_code = cell_code.replace("_", "")
            for code, name in operation_names.items():
                feature[f"transnas_cell_op_count__{name}"] = operation_code.count(code)
            edge_index = 0
            for node, node_code in enumerate(cell_code.split("_"), start=1):
                for source, code in enumerate(node_code):
                    factor_records.append(
                        {
                            "architecture_id": row["architecture_id"],
                            "search_space_id": space,
                            "factor_type": "micro_operation",
                            "position": edge_index,
                            "node": node,
                            "source": source,
                            "code": code,
                            "label": operation_names[code],
                        }
                    )
                    edge_index += 1
        feature_records.append(feature)
    features = pd.DataFrame.from_records(feature_records)
    return (
        frame.merge(
            features,
            on=["architecture_id", "search_space_id"],
            validate="many_to_one",
        ),
        pd.DataFrame.from_records(factor_records).drop_duplicates(),
    )


def _transnas_factor_effects(frame: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    protocol_fields = _study_group_fields(frame, ("search_space_id", "dataset", "proxy_id", "component"))
    columns = [*protocol_fields, "architecture_id", "score", "target_value", "direction"]
    working = frame[columns].copy()
    directions = working["direction"].astype(str).str.casefold()
    working["adjusted_score"] = np.where(
        directions.eq("minimize"), -working["score"], working["score"]
    )
    detailed = working.merge(factors, on=["architecture_id", "search_space_id"], validate="many_to_many")
    groups = [*protocol_fields, "factor_type", "position", "label"]
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
    baseline_groups = [*protocol_fields, "factor_type", "position"]
    baselines = (
        detailed.groupby(baseline_groups, as_index=False, dropna=False)
        .agg(factor_target_mean=("target_value", "mean"), factor_score_mean=("adjusted_score", "mean"))
    )
    effects = effects.merge(baselines, on=baseline_groups, validate="many_to_one")
    effects["target_delta_from_position_mean"] = effects["target_mean"] - effects["factor_target_mean"]
    effects["score_delta_from_position_mean"] = effects["score_mean"] - effects["factor_score_mean"]
    return effects


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
    protocol_fields = _study_group_fields(
        frame, ("search_space_id", "dataset", "proxy_id", "component")
    )
    duplicate_keys = [*protocol_fields, "architecture_id"]
    if frame.duplicated(duplicate_keys, keep=False).any():
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
    group_columns = list(protocol_fields)
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
    task_specific_fields = {
        "dataset",
        "target_metric",
        "target_split",
        "target_direction",
        "target_epoch_budget",
        "target_seed",
        "target_seed_reduction",
        "input_fingerprint",
    }
    transfer_groups = [field for field in protocol_fields if field not in task_specific_fields]
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
        task_quality.groupby([*transfer_groups, "method"], as_index=False, dropna=False)
        .agg(
            task_count=("dataset", "nunique"),
            valid_task_count=("correlation", "count"),
            mean_correlation=("correlation", "mean"),
            median_correlation=("correlation", "median"),
            min_correlation=("correlation", "min"),
            max_correlation=("correlation", "max"),
        )
    )
    result = {
        "task_quality": task_quality,
        "task_transfer": task_transfer,
        "space_summary": space_summary,
    }
    if "architecture" in frame:
        featured, factors = _transnas_architecture_features(frame)
        feature_columns = [column for column in featured if column.startswith("transnas_")]
        result.update(
            {
                "architecture_features": featured.drop_duplicates(
                    ["search_space_id", "architecture_id"]
                ),
                "architecture_factors": factors,
                "feature_correlations": _numeric_feature_correlations(
                    featured,
                    feature_columns=feature_columns,
                    study=study,
                    group_by=("search_space_id", "dataset", "proxy_id", "component"),
                    methods=methods,
                ),
                "factor_effects": _transnas_factor_effects(frame, factors),
            }
        )
    return result


__all__ = [
    "nats_size_study",
    "topology_study",
    "transnas_transfer_study",
    "vit_architecture_study",
]
