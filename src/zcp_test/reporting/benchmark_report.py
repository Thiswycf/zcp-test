from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, group in table.groupby(["proxy_id", "component"], dropna=False):
        ordered = group.sort_values("epoch_budget")
        axis.plot(ordered["epoch_budget"], ordered["spearman"], marker="o", label=" / ".join(map(str, label)))
    axis.set(xlabel="Epoch budget", ylabel="Spearman", title="ZCP correlation across budgets")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    artifacts = _save_figure(figure, output, "budget_correlation")
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
    elif view == "size":
        artifacts.extend(_plot_size(tables, output))
    elif view == "architecture":
        artifacts.extend(
            _plot_correlation_bars(tables, output, "architecture_features", "Architecture feature correlations")
        )
    elif view == "transfer":
        artifacts.extend(
            _plot_correlation_bars(tables, output, "task_quality", "Cross-task ZCP quality")
        )
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
