#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from zcp_test.proxies.az_nas import PLAINNET_COMPONENTS, log_rank_aggregate


SHANGHAI = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="1000,5000,10000,25000,50000,100000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value]
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("sizes must contain positive integers")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(args.seed)
    rows = [
        dict(zip(PLAINNET_COMPONENTS, values, strict=True))
        for values in rng.normal(size=(max(sizes), len(PLAINNET_COMPONENTS)))
    ]
    measurements = []
    coefficients = []
    for size in sizes:
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            scores = log_rank_aggregate(rows[:size], PLAINNET_COMPONENTS)
            durations.append(time.perf_counter() - started)
            if len(scores) != size or not all(math.isfinite(value) for value in scores):
                raise RuntimeError("log-rank benchmark returned invalid scores")
        median = statistics.median(durations)
        coefficient = median / (size * math.log2(max(size, 2)))
        coefficients.append(coefficient)
        measurements.append(
            {
                "candidates": size,
                "durations_seconds": durations,
                "median_seconds": median,
                "seconds_per_n_log2_n": coefficient,
            }
        )
    upper_coefficient = max(coefficients)
    cumulative_units = sum(
        size * math.log2(max(size, 2)) for size in range(1, 100_001)
    )
    payload = {
        "schema_version": "1.0",
        "recorded_at": datetime.now(SHANGHAI).isoformat(),
        "scope": "cpu_only_planning_benchmark_not_formal_search",
        "controller_semantics": "full-history four-component log-rank after every accepted candidate",
        "formal_candidate_count": 100_000,
        "components": list(PLAINNET_COMPONENTS),
        "seed": args.seed,
        "repeats": args.repeats,
        "measurements": measurements,
        "conservative_cumulative_rerank_seconds": upper_coefficient * cumulative_units,
        "limitations": [
            "This estimates CPU aggregation only and excludes model construction, proxy evaluation, JSONL fsync, mutation, and rejected candidates.",
            "The estimate is hardware- and software-version-specific and is not a completed 100000-candidate search.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
