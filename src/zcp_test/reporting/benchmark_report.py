from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


_BUDGET_PLOT_PROTOCOL_FIELDS = (
    "proxy_id",
    "component",
    "proxy_version",
    "seed",
    "target_seed_reduction",
    "input_source",
    "input_fingerprint",
    "model_fidelity",
)


def _budget_plot_groups(
    table: pd.DataFrame, extra_fields: tuple[str, ...] = ()
) -> tuple[list[str], Any]:
    fields = [
        field
        for field in (*_BUDGET_PLOT_PROTOCOL_FIELDS, *extra_fields)
        if field in table and table[field].notna().any()
    ]
    return fields, table.groupby(fields, dropna=False, sort=True)


def _budget_plot_label(fields: list[str], key: Any) -> str:
    values = key if isinstance(key, tuple) else (key,)
    visible = {"proxy_id", "component", "proxy_version", "seed", "requested_k"}
    labels = []
    for field, value in zip(fields, values, strict=True):
        if field in visible:
            if field == "component" and str(value) == "score":
                continue
            labels.append(f"{field}={value}")
    return " / ".join(labels)


def _budget_series_style(index: int, count: int, plt: Any) -> dict[str, Any]:
    fraction = index / max(count - 1, 1)
    markers = ("o", "s", "^", "D", "v", "P", "X")
    return {
        "color": plt.get_cmap("turbo")(fraction),
        "marker": markers[index % len(markers)],
        "linestyle": ("-", "--", "-.", ":")[(index // len(markers)) % 4],
    }


def _save_figure(figure: Any, output: Path, name: str) -> list[str]:
    artifacts = []
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight", metadata={"Creator": "zcp-test"})
        artifacts.append(path.name)
    return artifacts


def _plot_budget(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    table = tables.get("correlations", pd.DataFrame())
    if table.empty or not {"epoch_budget", "spearman"}.issubset(table):
        return []
    figure, axis = plt.subplots(figsize=(12, 6))
    fields, groups = _budget_plot_groups(table)
    series = list(groups)
    for index, (key, group) in enumerate(series):
        ordered = group.sort_values("epoch_budget")
        axis.plot(
            ordered["epoch_budget"],
            ordered["spearman"],
            label=_budget_plot_label(fields, key),
            **_budget_series_style(index, len(series), plt),
        )
    axis.set(xlabel="Epoch budget", ylabel="Spearman", title="ZCP correlation across budgets")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.72, 0.5), fontsize="x-small")
        figure.subplots_adjust(right=0.7)
    else:
        figure.tight_layout()
    artifacts = _save_figure(figure, output, "budget_correlation")
    plt.close(figure)
    retrieval = tables.get("top_k_retrieval", pd.DataFrame())
    if not retrieval.empty and {"epoch_budget", "requested_k", "precision_at_k"}.issubset(
        retrieval
    ):
        requested = sorted(int(value) for value in retrieval["requested_k"].dropna().unique())
        figure, axes = plt.subplots(
            len(requested),
            1,
            figsize=(12, max(5, 3.7 * len(requested))),
            squeeze=False,
            sharex=True,
        )
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for row, requested_k in enumerate(requested):
            axis = axes[row, 0]
            selected = retrieval[retrieval["requested_k"].eq(requested_k)]
            fields, groups = _budget_plot_groups(selected)
            series = list(groups)
            for index, (key, group) in enumerate(series):
                ordered = group.sort_values("epoch_budget")
                axis.plot(
                    ordered["epoch_budget"],
                    ordered["precision_at_k"],
                    label=_budget_plot_label(fields, key),
                    **_budget_series_style(index, len(series), plt),
                )
            axis.set(ylabel=f"Precision@{requested_k}")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.25)
            if row == 0:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
        axes[-1, 0].set_xlabel("Epoch budget")
        figure.suptitle("NAS-Bench-101 top-k retrieval across budgets")
        if legend_handles:
            figure.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.72, 0.5),
                fontsize="x-small",
            )
            figure.subplots_adjust(right=0.7, hspace=0.28, top=0.94)
        else:
            figure.tight_layout()
        artifacts.extend(_save_figure(figure, output, "budget_top_k_retrieval"))
        plt.close(figure)
    controlled = tables.get("structure_controlled_correlations", pd.DataFrame())
    if not controlled.empty and {"epoch_budget", "spearman"}.issubset(controlled):
        figure, axis = plt.subplots(figsize=(10, 6))
        fields, groups = _budget_plot_groups(controlled)
        series = list(groups)
        for index, (key, group) in enumerate(series):
            ordered = group.sort_values("epoch_budget")
            axis.plot(
                ordered["epoch_budget"],
                ordered["spearman"],
                label=_budget_plot_label(fields, key),
                **_budget_series_style(index, len(series), plt),
            )
        axis.set(
            xlabel="Epoch budget",
            ylabel="Controlled Spearman",
            title="NB101 ZCP correlation after structure-count controls",
        )
        axis.set_ylim(-1.05, 1.05)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.72, 0.5), fontsize="x-small")
            figure.subplots_adjust(right=0.7)
        else:
            figure.tight_layout()
        artifacts.extend(_save_figure(figure, output, "budget_structure_controlled"))
        plt.close(figure)
    neighborhood = tables.get("neighborhood_correlations", pd.DataFrame())
    if not neighborhood.empty and {
        "epoch_budget",
        "direction_agreement_rate",
    }.issubset(neighborhood):
        figure, axis = plt.subplots(figsize=(10, 6))
        fields, groups = _budget_plot_groups(neighborhood)
        series = list(groups)
        for index, (key, group) in enumerate(series):
            ordered = group.sort_values("epoch_budget")
            axis.plot(
                ordered["epoch_budget"],
                ordered["direction_agreement_rate"],
                label=_budget_plot_label(fields, key),
                **_budget_series_style(index, len(series), plt),
            )
        axis.set(
            xlabel="Epoch budget",
            ylabel="One-edit direction agreement",
            title="NB101 local one-edit ranking agreement",
        )
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.72, 0.5), fontsize="x-small")
            figure.subplots_adjust(right=0.7)
        else:
            figure.tight_layout()
        artifacts.extend(_save_figure(figure, output, "budget_neighborhood_agreement"))
        plt.close(figure)
    return artifacts


def _plot_topology(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    table = tables.get("operations", pd.DataFrame())
    if table.empty or not {"operation", "edge_fraction"}.issubset(table):
        return []
    figure, axis = plt.subplots(figsize=(8, 5))
    ordered = table.sort_values("edge_fraction", ascending=False)
    axis.bar(ordered["operation"].astype(str), ordered["edge_fraction"])
    axis.set(xlabel="Operation", ylabel="Edge fraction", title="Topology operation distribution")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    artifacts = _save_figure(figure, output, "topology_operations")
    plt.close(figure)
    return artifacts


def _plot_size(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    table = tables.get("stages", pd.DataFrame())
    if table.empty or not {"stage", "channel"}.issubset(table):
        return []
    stages = sorted(table["stage"].unique())
    values = [table.loc[table["stage"].eq(stage), "channel"].to_numpy() for stage in stages]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=[str(stage) for stage in stages])
    axis.set(xlabel="Stage", ylabel="Channels", title="NATS-SSS channel distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    artifacts = _save_figure(figure, output, "size_stages")
    plt.close(figure)
    return artifacts


def _plot_correlation_bars(
    tables: dict[str, pd.DataFrame], output: Path, name: str, title: str
) -> list[str]:
    import matplotlib.pyplot as plt

    table = tables.get("correlations", tables.get("task_quality", pd.DataFrame()))
    if table.empty:
        return []
    value_column = "correlation" if "correlation" in table else None
    label_column = "feature" if "feature" in table else "dataset" if "dataset" in table else None
    if value_column is None or label_column is None:
        return []
    selected = table.copy()
    if "method" in selected:
        preferred = selected[selected["method"].eq("spearman")]
        if not preferred.empty:
            selected = preferred
    selected = selected.dropna(subset=[value_column]).head(30)
    if selected.empty:
        return []
    labels = selected[label_column].astype(str)
    if "proxy_id" in selected:
        labels = selected["proxy_id"].astype(str) + " / " + labels
    figure, axis = plt.subplots(figsize=(max(8, len(selected) * 0.35), 5))
    axis.bar(labels, selected[value_column])
    axis.set(ylabel="Correlation", title=title)
    axis.set_ylim(-1.05, 1.05)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    artifacts = _save_figure(figure, output, name)
    plt.close(figure)
    return artifacts


def _plot_transfer_matrix(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    table = tables.get("task_transfer", pd.DataFrame())
    if table.empty or not {"source_task", "target_task", "correlation"}.issubset(table):
        return []
    selected = table[table["method"].eq("spearman")] if "method" in table else table
    if selected.empty:
        return []
    grouping = [field for field in ("search_space_id", "proxy_id", "component") if field in selected]
    artifacts: list[str] = []
    for index, (key, group) in enumerate(selected.groupby(grouping, dropna=False, sort=True)):
        matrix = group.pivot(index="source_task", columns="target_task", values="correlation")
        figure, axis = plt.subplots(figsize=(max(6, len(matrix.columns)), max(5, len(matrix))))
        image = axis.imshow(matrix.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set(xlabel="Target task", ylabel="Source task", title=f"Task transfer: {key}")
        for row in range(len(matrix.index)):
            for column in range(len(matrix.columns)):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.85)
        figure.tight_layout()
        artifacts.extend(_save_figure(figure, output, f"task_transfer_{index}"))
        plt.close(figure)
    return artifacts


def _plot_darts_interactions(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    table = tables.get("operation_topology_interactions", pd.DataFrame())
    required = {"cell", "node", "operation", "target_delta_from_interaction_mean"}
    if table.empty or not required.issubset(table):
        return []
    selected = table.copy()
    if "proxy_id" in selected:
        first = selected[["proxy_id", "component"]].drop_duplicates().iloc[0]
        selected = selected[
            selected["proxy_id"].eq(first["proxy_id"])
            & selected["component"].eq(first["component"])
        ]
    selected["location"] = selected["cell"].astype(str) + "/node" + selected["node"].astype(str)
    matrix = selected.pivot_table(
        index="operation",
        columns="location",
        values="target_delta_from_interaction_mean",
        aggfunc="mean",
    )
    figure, axis = plt.subplots(figsize=(max(7, len(matrix.columns)), max(4, 0.6 * len(matrix))))
    maximum = np.nanmax(np.abs(matrix.to_numpy(float))) if matrix.size else 1.0
    maximum = maximum if np.isfinite(maximum) and maximum > 0 else 1.0
    image = axis.imshow(matrix.to_numpy(float), cmap="coolwarm", vmin=-maximum, vmax=maximum)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("DARTS operation × topology target effects")
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    artifacts = _save_figure(figure, output, "darts_operation_topology_interactions")
    plt.close(figure)
    return artifacts


def write_benchmark_study(
    tables: dict[str, pd.DataFrame],
    destination: str | Path,
    *,
    view: str,
    benchmark_id: str,
) -> dict[str, Any]:
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        artifacts.append(path.name)
    if view == "budget":
        artifacts.extend(_plot_budget(tables, output))
    elif view == "topology":
        artifacts.extend(_plot_topology(tables, output))
        artifacts.extend(
            _plot_correlation_bars(
                tables,
                output,
                "topology_feature_correlations",
                "Topology feature correlations",
            )
        )
    elif view == "size":
        artifacts.extend(_plot_size(tables, output))
        artifacts.extend(
            _plot_correlation_bars(
                tables, output, "size_feature_correlations", "NATS-SSS feature correlations"
            )
        )
    elif view == "darts":
        artifacts.extend(
            _plot_correlation_bars(
                tables, output, "darts_feature_correlations", "DARTS feature correlations"
            )
        )
        artifacts.extend(_plot_darts_interactions(tables, output))
    elif view == "architecture":
        artifacts.extend(
            _plot_correlation_bars(tables, output, "architecture_features", "Architecture feature correlations")
        )
    elif view == "transfer":
        artifacts.extend(
            _plot_correlation_bars(tables, output, "task_quality", "Cross-task ZCP quality")
        )
        artifacts.extend(_plot_transfer_matrix(tables, output))
    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "view": view,
        "tables": {name: len(table) for name, table in tables.items()},
        "artifacts": artifacts,
    }
    sections = []
    for name, table in tables.items():
        sections.append(f"<h2>{html.escape(name)}</h2>{table.head(100).to_html(index=False)}")
    images = "".join(
        f'<img src="{html.escape(path)}" alt="{html.escape(path)}">'
        for path in artifacts
        if path.endswith(".png")
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>zcp-test benchmark study</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:.3rem}img{max-width:100%;display:block}</style>"
        f"<h1>{html.escape(benchmark_id)} / {html.escape(view)}</h1>{images}{''.join(sections)}",
        encoding="utf-8",
    )
    manifest["artifacts"].extend(["study.json", "index.html"])
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output_directory": str(output), **manifest}


__all__ = ["write_benchmark_study"]
