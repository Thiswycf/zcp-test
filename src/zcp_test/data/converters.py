from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from zcp_test.data.assets import sha256_file
from zcp_test.data.jsonl import convert_trusted_torch_records

RecordParser = Callable[[Any], Iterable[tuple[Mapping[str, Any], list[Mapping[str, Any]]]]]


def convert_trusted_benchmark(
    source: str | Path,
    destination: str | Path,
    *,
    benchmark_id: str,
    search_space_id: str,
    benchmark_version: str,
    protocol: str,
    parser: RecordParser,
    trusted: bool,
) -> Path:
    """Normalize a release-specific trusted torch file into runtime-safe JSONL."""
    source_path = Path(source).expanduser().resolve()
    source_hash = sha256_file(source_path)

    def convert(payload: Any):
        for index, (specification, metrics) in enumerate(parser(payload)):
            if not isinstance(specification, Mapping) or not isinstance(metrics, list):
                raise ValueError("Converter parser must yield (specification mapping, metrics list)")
            yield {
                "record_kind": "benchmark_architecture",
                "benchmark_id": benchmark_id,
                "search_space_id": search_space_id,
                "benchmark_version": benchmark_version,
                "benchmark_index": index,
                "protocol": protocol,
                "source_sha256": source_hash,
                "specification": dict(specification),
                "metrics": [dict(metric) for metric in metrics],
            }

    return convert_trusted_torch_records(source_path, destination, convert, trusted=trusted)


def convert_vitbench101(
    source: str | Path,
    destination: str | Path,
    *,
    slice_id: str,
    parser: RecordParser,
    trusted: bool,
) -> Path:
    from zcp_test.benchmarks.vitbench101 import VIT_SLICES

    try:
        search_space_id, protocol = VIT_SLICES[slice_id]
    except KeyError as error:
        raise ValueError(f"Unknown ViT-Bench-101 slice: {slice_id}") from error
    return convert_trusted_benchmark(
        source,
        destination,
        benchmark_id="vitbench101",
        search_space_id=search_space_id,
        benchmark_version="auto-prox-90ed458",
        protocol=protocol,
        parser=parser,
        trusted=trusted,
    )


def vitbench101_release_parser(payload: Any):
    """Parse the list-of-dicts format published with Auto-Prox commit 90ed458."""
    if not isinstance(payload, list):
        raise ValueError("ViT-Bench-101 release must contain a list")
    metric_map = {
        "c100_base_acc": ("cifar100", "accuracy_vanilla"),
        "c100_kd_acc": ("cifar100", "accuracy_kd"),
        "flower_base_acc": ("flowers", "accuracy_vanilla"),
        "flower_kd_acc": ("flowers", "accuracy_kd"),
        "chaoyang_base_acc": ("chaoyang", "accuracy_vanilla"),
        "chaoyang_kd_acc": ("chaoyang", "accuracy_kd"),
        "imagenet_super_acc": ("imagenet1k", "accuracy_inherited"),
    }
    for entry in payload:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("arch"), Mapping):
            raise ValueError("Invalid ViT-Bench-101 release record")
        metrics = []
        for source_key, (dataset, metric_name) in metric_map.items():
            value = entry.get(source_key)
            if value is not None:
                metrics.append(
                    {
                        "dataset": dataset,
                        "split": "test",
                        "metric_name": metric_name,
                        "value": float(value),
                    }
                )
        yield dict(entry["arch"]), metrics


def convert_transnasbench101(
    source: str | Path,
    destination: str | Path,
    *,
    space: str,
    trusted: bool,
) -> Path:
    if space not in {"micro", "macro"}:
        raise ValueError("TransNAS space must be 'micro' or 'macro'")

    def parser(payload: Any):
        try:
            architectures = payload["data"][space]
        except (KeyError, TypeError) as error:
            raise ValueError(f"TransNAS release does not contain space {space!r}") from error
        for architecture, task_results in architectures.items():
            metrics = []
            for task, result in task_results.items():
                budget = int(result["total_epochs"])
                for metric_name, values in result["metrics"].items():
                    if not isinstance(values, (list, tuple)) or not values:
                        continue
                    lowered = metric_name.lower()
                    split = next(
                        (candidate for candidate in ("train", "valid", "test") if candidate in lowered),
                        "benchmark",
                    )
                    metrics.append(
                        {
                            "dataset": task,
                            "split": split,
                            "metric_name": metric_name,
                            "epoch_budget": budget,
                            "value": float(values[-1]),
                        }
                    )
            yield {"architecture": str(architecture)}, metrics

    return convert_trusted_benchmark(
        source,
        destination,
        benchmark_id="transnasbench101",
        search_space_id=f"transnas_{space}",
        benchmark_version="v10141024",
        protocol=f"transnasbench101-{space}-final",
        parser=parser,
        trusted=trusted,
    )
