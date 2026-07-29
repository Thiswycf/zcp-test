from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator

from zcp_test.artifacts.jsonl import read_jsonl


def _primary_component(proxy_id: str, components: dict[str, float]) -> str:
    try:
        from zcp_test.proxies import PROXIES, load_builtin_proxies

        load_builtin_proxies()
        capability = PROXIES.create(proxy_id).capability
        if capability.primary_component in components:
            return capability.primary_component
    except (ImportError, KeyError):
        pass
    if "score" in components:
        return "score"
    return sorted(components)[0]


def normalize_score_records(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield schema-2 score rows, folding schema-1 component-long records."""

    groups: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    for row in rows:
        if "components" in row:
            normalized = dict(row)
            normalized.setdefault("schema_version", "2.0")
            normalized.setdefault("primary_component", "score")
            yield normalized
            continue
        key = (
            row.get("run_id"),
            row.get("architecture_id"),
            row.get("proxy_id"),
            row.get("dataset"),
            row.get("seed"),
            row.get("input_fingerprint"),
        )
        group = groups.setdefault(key, dict(row))
        component = str(row.get("component") or "score")
        if row.get("score") is not None:
            group.setdefault("components", {})[component] = float(row["score"])
        if row.get("status", "ok") != "ok":
            group.update(
                status=row.get("status"),
                error_type=row.get("error_type"),
                error_message=row.get("error_message"),
            )
    for group in groups.values():
        components = group.pop("components", {})
        group.pop("component", None)
        primary = _primary_component(str(group.get("proxy_id")), components) if components else "score"
        group.update(
            schema_version="2.0",
            primary_component=primary,
            components=components,
            score=components.get(primary),
        )
        yield group


def read_score_records(path: str | Path) -> Iterator[dict[str, Any]]:
    return normalize_score_records(read_jsonl(path))


def score_component(row: dict[str, Any], component: str | None = None) -> float | None:
    selected = component or str(row.get("primary_component", "score"))
    value = row.get("components", {}).get(selected)
    if value is None and selected == row.get("primary_component"):
        value = row.get("score")
    return None if value is None else float(value)
