#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from zcp_test.artifacts import JsonlWriter
from zcp_test.gpu import configure_cuda, gpu_lock, select_gpu
from zcp_test.search.plainnet_source_aligned import (
    PlainNetSourceAlignedSearch,
    resolve_target_profile,
)
from zcp_test.spaces import SPACES, load_builtin_spaces


SHANGHAI = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flops-target", choices=("450m", "600m", "1g"), default="450m")
    parser.add_argument("--accepted", type=int, default=3)
    parser.add_argument("--lock-timeout", type=float, default=21_600)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now(SHANGHAI).isoformat()


def main() -> None:
    args = parse_args()
    if not 1 <= args.accepted < 100_000:
        raise ValueError("accepted must be between 1 and 99999 for a non-terminal preflight")
    selection = select_gpu(args.gpu)
    configure_cuda(selection)
    run = args.output / f"{datetime.now(SHANGHAI).strftime('%Y%m%dT%H%M%S%z')}_plainnet-preflight"
    run.mkdir(parents=True, exist_ok=False)
    manifest_path = run / "preflight-manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "waiting_for_gpu_lock",
        "scope": "gpu_memory_throughput_preflight_not_formal_search",
        "started_at": timestamp(),
        "formal_valid_candidates": 100_000,
        "stop_after_accepted": args.accepted,
        "flops_target": args.flops_target,
        "batch_size": 64,
        "input_size": 224,
        "gpu_selection": selection,
    }
    atomic_json(manifest_path, manifest)
    try:
        with gpu_lock(selection, timeout=args.lock_timeout):
            manifest.update({"status": "running", "lock_acquired_at": timestamp()})
            atomic_json(manifest_path, manifest)
            import torch

            from zcp_test.cli import _seed_search_model
            from zcp_test.proxies.evaluator import evaluate_proxy

            load_builtin_spaces()
            space = SPACES.create("zennas_plainnet_mbv2")
            device = torch.device("cuda:0")
            torch.manual_seed(args.seed)
            inputs = torch.randn(64, 3, 224, 224, device=device)
            labels = torch.zeros(64, dtype=torch.long, device=device)
            loss_fn = torch.nn.CrossEntropyLoss()
            measurements: list[dict[str, object]] = []
            measurement_writer = JsonlWriter(run / "measurements.jsonl", 1)

            def evaluator(architecture):
                _seed_search_model(args.seed, architecture.architecture_id)
                model = space.build_model(architecture, 1_000).to(device)
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    result = evaluate_proxy(
                        "az_nas_plainnet",
                        model,
                        inputs,
                        labels,
                        loss_fn,
                        space.model_family,
                    )
                    torch.cuda.synchronize(device)
                    row = {
                        "timestamp": timestamp(),
                        "architecture_id": architecture.architecture_id,
                        "status": result.status.value,
                        "duration_seconds": time.perf_counter() - started,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                        "error_type": result.error_type,
                        "error_message": result.error_message,
                    }
                    measurement_writer.append(row)
                    measurements.append(row)
                    if result.status.value != "ok" or result.score is None:
                        raise RuntimeError(
                            f"PlainNet preflight failed: {result.error_type}: {result.error_message}"
                        )
                    if not all(math.isfinite(float(value)) for value in result.components.values()):
                        raise RuntimeError("PlainNet preflight returned non-finite components")
                    return dict(result.components)
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

            search = PlainNetSourceAlignedSearch(
                space=space,
                evaluator=evaluator,
                writer=JsonlWriter(run / "search.jsonl", 1),
                state_path=run / "search-state.json",
                seed=args.seed,
                target=resolve_target_profile(args.flops_target),
                valid_candidates=100_000,
                parent_pool=1_024,
                classes=1_000,
                record_metadata={
                    "purpose": "gpu_memory_throughput_preflight",
                    "formal_search_completed": False,
                    "formal_valid_candidates": 100_000,
                },
            )
            result = search.run(stop_after_accepted=args.accepted)
            if result is not None:
                raise RuntimeError("PlainNet preflight unexpectedly finalized the formal search")
            state = json.loads((run / "search-state.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (run / "search.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            if state.get("status") != "running":
                raise RuntimeError("PlainNet preflight state must remain running")
            if state.get("identity", {}).get("valid_candidates") != 100_000:
                raise RuntimeError("PlainNet preflight lost the formal 100k identity")
            if state.get("accepted_count") != args.accepted or len(rows) != args.accepted:
                raise RuntimeError("PlainNet preflight accepted-count mismatch")
            if any(row.get("record_kind") != "candidate" for row in rows):
                raise RuntimeError("PlainNet preflight must not write a search summary")
            durations = [float(row["duration_seconds"]) for row in measurements]
            summary = {
                **manifest,
                "status": "preflight_completed",
                "completed_at": timestamp(),
                "formal_search_completed": False,
                "state_status": state["status"],
                "accepted_count": state["accepted_count"],
                "measurement_count": len(measurements),
                "median_candidate_seconds": statistics.median(durations),
                "max_candidate_seconds": max(durations),
                "max_peak_allocated_gib": max(float(row["peak_allocated_gib"]) for row in measurements),
                "max_peak_reserved_gib": max(float(row["peak_reserved_gib"]) for row in measurements),
            }
            atomic_json(run / "preflight-summary.json", summary)
            atomic_json(manifest_path, summary)
            print(json.dumps(summary, ensure_ascii=False))
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at": timestamp(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "formal_search_completed": False,
            }
        )
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
