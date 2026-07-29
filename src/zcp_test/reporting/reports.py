from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from zcp_test.artifacts import read_jsonl


def jsonl_to_csv(source: str | Path, destination: str | Path) -> int:
    rows = list(read_jsonl(source))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return len(rows)


def static_html(source: str | Path, destination: str | Path, title: str = "zcp-test report") -> int:
    rows = list(read_jsonl(source))
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    body = [f"<h1>{html.escape(title)}</h1>", "<table><thead><tr>"]
    body.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    body.append("</tr></thead><tbody>")
    for row in rows:
        body.append("<tr>")
        body.extend(f"<td><pre>{html.escape(json.dumps(row.get(field), ensure_ascii=False))}</pre></td>" for field in fields)
        body.append("</tr>")
    body.append("</tbody></table>")
    document = "<!doctype html><meta charset='utf-8'><style>table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px}pre{white-space:pre-wrap}</style>" + "".join(body)
    Path(destination).write_text(document, encoding="utf-8")
    return len(rows)


def curve_plot(source: str | Path, destination: str | Path, kind: str) -> int:
    import matplotlib.pyplot as plt

    from zcp_test.reporting.analysis import plot_search, plot_training

    rows = list(read_jsonl(source))
    if not rows:
        raise ValueError("Cannot plot an empty JSONL file")
    if kind == "training":
        figure = plot_training(rows, destination)
    elif kind == "search":
        figure = plot_search(rows, destination)
    else:
        raise ValueError(f"Unknown plot kind: {kind}")
    plt.close(figure)
    return len(rows)
