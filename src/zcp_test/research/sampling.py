from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from zcp_test.types import Architecture


def _stable_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def architecture_stratum(benchmark_id: str, specification: Mapping[str, Any]) -> str:
    if benchmark_id == "nasbench101":
        matrix = specification["matrix"]
        operations = specification["operations"][1:-1]
        counts = Counter(operations)
        return _stable_key(
            {
                "vertices": len(matrix),
                "edges": sum(sum(int(value) for value in row) for row in matrix),
                "conv3": counts["conv3x3-bn-relu"],
                "conv1": counts["conv1x1-bn-relu"],
                "pool": counts["maxpool3x3"],
            }
        )
    if benchmark_id in {"nasbench201", "nats_tss"}:
        operations = re.findall(r"([a-z0-9_]+)~", str(specification["architecture"]))
        return _stable_key(Counter(operations))
    if benchmark_id == "nats_sss":
        widths = [int(value) for value in str(specification["architecture"]).split(":")]
        return _stable_key(
            {
                "total_width_bin": sum(widths) // 32,
                "unique_widths": len(set(widths)),
                "nondecreasing": widths == sorted(widths),
            }
        )
    if benchmark_id == "nasbench301_surrogate":
        result: dict[str, Any] = {}
        for cell_name in ("normal", "reduce"):
            edges = specification[cell_name]
            operations = Counter(str(edge[0]) for edge in edges)
            parents = Counter(int(edge[1]) for edge in edges)
            result[cell_name] = {
                "skip": operations["skip_connect"],
                "pool": operations["avg_pool_3x3"] + operations["max_pool_3x3"],
                "sep": operations["sep_conv_3x3"] + operations["sep_conv_5x5"],
                "dil": operations["dil_conv_3x3"] + operations["dil_conv_5x5"],
                "reused_parents": sum(value > 1 for value in parents.values()),
            }
        return _stable_key(result)
    if benchmark_id == "transnasbench101":
        encoding = str(specification["architecture"])
        parts = encoding.split("-")
        if len(parts) != 3:
            raise ValueError(f"Invalid TransNAS architecture: {encoding!r}")
        base_channel, macro_code, cell_code = parts
        if not base_channel.isdigit() or not macro_code or any(
            digit not in "1234" for digit in macro_code
        ):
            raise ValueError(f"Invalid TransNAS architecture: {encoding!r}")
        macro_digits = Counter(macro_code)
        stratum: dict[str, Any] = {
            "variant": "macro" if cell_code == "basic" else "micro",
            "base_channel": int(base_channel),
            "macro_module_count": len(macro_code),
            "macro_code_1_count": macro_digits["1"],
            "macro_code_2_count": macro_digits["2"],
            "macro_code_3_count": macro_digits["3"],
            "macro_code_4_count": macro_digits["4"],
        }
        if cell_code != "basic":
            cell_groups = cell_code.split("_")
            if tuple(map(len, cell_groups)) != (1, 2, 3) or any(
                operation not in "0123"
                for group in cell_groups
                for operation in group
            ):
                raise ValueError(f"Invalid TransNAS micro architecture: {encoding!r}")
            cell_operations = Counter("".join(cell_groups))
            stratum.update(
                {
                    "cell_operation_count": 6,
                    "cell_op_0_count": cell_operations["0"],
                    "cell_op_1_count": cell_operations["1"],
                    "cell_op_2_count": cell_operations["2"],
                    "cell_op_3_count": cell_operations["3"],
                }
            )
        return _stable_key(stratum)
    if benchmark_id == "vitbench101":
        depth = specification.get("depth")
        if isinstance(depth, list):
            depth_value = sum(int(value) for value in depth)
        else:
            depth_value = int(depth)
        hidden = specification.get("hidden_dim", specification.get("base_dim"))
        return _stable_key({"depth": depth_value, "hidden": int(hidden)})
    return _stable_key({"benchmark_id": benchmark_id})


def _proportional_allocations(
    groups: Mapping[str, list[dict[str, Any]]], count: int, population_size: int, seed: int
) -> dict[str, int]:
    exact = {
        key: count * len(records) / population_size for key, records in groups.items()
    }
    allocations = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(allocations.values())
    rng = random.Random(seed)
    tie_breakers = {key: rng.random() for key in groups}
    order = sorted(
        groups,
        key=lambda key: (exact[key] - allocations[key], tie_breakers[key]),
        reverse=True,
    )
    for key in order[:remaining]:
        allocations[key] += 1
    return allocations


def create_sample_manifest(
    benchmark_id: str,
    benchmark_version: str | None,
    architectures: Iterable[Architecture],
    *,
    count: int | None = None,
    fraction: float | None = None,
    seed: int = 0,
    shards: int = 1,
) -> dict[str, Any]:
    if (count is None) == (fraction is None):
        raise ValueError("Specify exactly one of count or fraction")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if shards <= 0:
        raise ValueError("shards must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    architecture_ids: set[str] = set()
    search_space_ids: set[str] = set()
    sampling_provenance = dict(
        getattr(architectures, "sampling_provenance", {}) or {}
    )
    for position, architecture in enumerate(architectures):
        if architecture.architecture_id in architecture_ids:
            raise ValueError(f"Duplicate architecture ID: {architecture.architecture_id}")
        architecture_ids.add(architecture.architecture_id)
        search_space_ids.add(architecture.search_space_id)
        index = position if architecture.benchmark_index is None else architecture.benchmark_index
        stratum = architecture_stratum(benchmark_id, architecture.spec)
        groups[stratum].append(
            {
                "benchmark_index": int(index),
                "architecture_id": architecture.architecture_id,
                "stratum": stratum,
            }
        )
    population_size = sum(len(records) for records in groups.values())
    if population_size == 0:
        raise ValueError("Cannot sample an empty benchmark")
    sample_count = int(count) if count is not None else math.ceil(population_size * float(fraction))
    if not 0 < sample_count <= population_size:
        raise ValueError("sample count must be within the benchmark population")
    allocations = _proportional_allocations(groups, sample_count, population_size, seed)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        records = list(groups[key])
        rng.shuffle(records)
        selected.extend(records[: allocations[key]])
    selected.sort(key=lambda record: record["benchmark_index"])
    shard_records = [selected[index::shards] for index in range(shards)]
    if len(search_space_ids) != 1:
        raise ValueError("Sample population must contain exactly one search space")
    search_space_id = next(iter(search_space_ids))
    recorded_search_space = sampling_provenance.get("search_space_id")
    if recorded_search_space and recorded_search_space != search_space_id:
        raise ValueError("Sampling provenance search_space_id does not match architectures")
    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "search_space_id": search_space_id,
        "strategy": "proportional_feature_stratified",
        "seed": seed,
        "population_size": population_size,
        "sample_count": sample_count,
        "sample_fraction": sample_count / population_size,
        "stratum_count": len(groups),
        "selected": selected,
        "shards": [
            {
                "shard_index": index,
                "sample_count": len(records),
                "benchmark_indices": [record["benchmark_index"] for record in records],
            }
            for index, records in enumerate(shard_records)
        ],
    }
    if benchmark_id == "transnasbench101":
        variant = sampling_provenance.get("benchmark_variant")
        expected_variant = search_space_id.removeprefix("transnas_")
        if variant and variant != expected_variant:
            raise ValueError("Sampling provenance variant does not match search_space_id")
        manifest.update(
            {
                "benchmark_variant": variant or expected_variant,
                "source_sha256": sampling_provenance.get("source_sha256"),
                "converted_file_sha256": sampling_provenance.get(
                    "converted_file_sha256"
                ),
            }
        )
    return manifest


def load_sample_indices(
    path: str | Path,
    *,
    benchmark_id: str,
    benchmark_version: str | None,
    search_space_id: str | None = None,
    shard_index: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    source = Path(path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported sample manifest schema")
    if manifest.get("benchmark_id") != benchmark_id:
        raise ValueError("Sample manifest benchmark_id does not match evaluate benchmark")
    recorded_version = manifest.get("benchmark_version")
    if recorded_version and benchmark_version and str(recorded_version) != str(benchmark_version):
        raise ValueError("Sample manifest benchmark_version does not match evaluate benchmark")
    recorded_space = manifest.get("search_space_id")
    if recorded_space and search_space_id and str(recorded_space) != str(search_space_id):
        raise ValueError("Sample manifest search_space_id does not match evaluate search space")
    if shard_index is None:
        indices = [int(record["benchmark_index"]) for record in manifest["selected"]]
    else:
        shards = manifest.get("shards", [])
        if not 0 <= shard_index < len(shards):
            raise ValueError("sample shard index is outside the manifest shard range")
        indices = [int(value) for value in shards[shard_index]["benchmark_indices"]]
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("Sample manifest contains invalid or duplicate benchmark indices")
    return indices, manifest
