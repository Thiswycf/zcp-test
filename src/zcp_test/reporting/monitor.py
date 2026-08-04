from __future__ import annotations

import html
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PathLike = str | Path


def _source_path(source: PathLike) -> Path:
    path = Path(source)
    if not path.is_dir():
        return path
    for name in ("scores.jsonl", "search.jsonl", "events.jsonl", "training.jsonl"):
        candidate = path / name
        if candidate.exists():
            return candidate
    candidates = [
        child
        for child in sorted(path.iterdir())
        if child.is_dir()
        and any(
            (child / name).exists()
            for name in ("scores.jsonl", "search.jsonl", "events.jsonl", "training.jsonl")
        )
    ]
    if len(candidates) == 1:
        return _source_path(candidates[0])
    if candidates:
        raise ValueError(
            f"Monitor root contains {len(candidates)} runs; select one timestamped run directory"
        )
    raise FileNotFoundError(f"No monitorable JSONL file in {path}")


def monitor_source_path(source: PathLike) -> Path:
    """Resolve a JSONL file, run directory, or unambiguous run parent."""
    return _source_path(source)


def read_jsonl_tolerant(
    source: PathLike,
    *,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Read complete JSONL records, retaining an incomplete trailing line for retry."""
    path = _source_path(source)
    size = path.stat().st_size
    if offset < 0 or offset > size:
        raise ValueError(f"offset must be between 0 and {size}")
    rows: list[dict[str, Any]] = []
    ignored_partial = False
    next_offset = offset
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(size - offset)
        position = offset
        for line in payload.splitlines(keepends=True):
            line_start = position
            position += len(line)
            terminated = line.endswith(b"\n")
            if not line.strip():
                next_offset = position
                continue
            try:
                text = line.decode("utf-8")
                value = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if not terminated and position == size:
                    ignored_partial = True
                    next_offset = line_start
                    break
                raise ValueError(f"Invalid JSONL at byte {line_start} in {path}") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected a JSON object at byte {line_start} in {path}")
            rows.append(dict(value))
            next_offset = position
    return rows, next_offset, ignored_partial


def _monitor_html(
    source: Path,
    rows: list[dict[str, Any]],
    partial: bool,
    title: str,
    browser_refresh_seconds: float,
) -> str:
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    body = [f"<h1>{html.escape(title)}</h1>"]
    body.append(
        f"<p>Source: <code>{html.escape(str(source))}</code> · Rows: {len(rows)}"
        f" · Partial tail: {'yes' if partial else 'no'}</p>"
    )
    body.append("<table><thead><tr>")
    body.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    body.append("</tr></thead><tbody>")
    for row in rows[-200:]:
        body.append("<tr>")
        for field in fields:
            value = json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
            body.append(f"<td><pre>{html.escape(value)}</pre></td>")
        body.append("</tr>")
    body.append("</tbody></table>")
    style = "body{font:14px system-ui;margin:20px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:4px;vertical-align:top}pre{margin:0;white-space:pre-wrap}"
    refresh_value = format(browser_refresh_seconds, "g")
    return f"<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='{refresh_value}'><title>{html.escape(title)}</title><style>{style}</style>{''.join(body)}"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def refresh_once(
    source: PathLike,
    destination: PathLike | None = None,
    *,
    offset: int = 0,
    title: str = "zcp-test monitor",
    history: list[dict[str, Any]] | None = None,
    browser_refresh_seconds: float = 5.0,
) -> dict[str, Any]:
    """Read one stable snapshot and optionally atomically refresh a static HTML view."""
    path = _source_path(source)
    rows, next_offset, ignored_partial = read_jsonl_tolerant(path, offset=offset)
    snapshot = [*(history or []), *rows]
    if browser_refresh_seconds <= 0:
        raise ValueError("browser_refresh_seconds must be positive")
    output: Path | None = None
    if destination is not None:
        output = Path(destination)
        if output.suffix.lower() != ".html":
            output = output / "monitor.html"
        _atomic_write(
            output,
            _monitor_html(
                path, snapshot, ignored_partial, title, browser_refresh_seconds
            ),
        )
    return {
        "source": str(path),
        "output": None if output is None else str(output),
        "rows": rows,
        "row_count": len(snapshot),
        "new_row_count": len(rows),
        "next_offset": next_offset,
        "ignored_partial_line": ignored_partial,
    }


monitor_once = refresh_once
refresh_monitor_once = refresh_once
refresh_monitor = refresh_once


__all__ = [
    "monitor_source_path",
    "monitor_once",
    "read_jsonl_tolerant",
    "refresh_monitor",
    "refresh_monitor_once",
    "refresh_once",
]
