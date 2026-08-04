#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


TIME_ZONE = ZoneInfo("Asia/Shanghai")
PANEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = PANEL_DIR.parent
DARTS_STATUS = REPO_ROOT / "runs/acceptance/darts-imagenet-parallel/status.json"
AUTOFORMER_ROOT = REPO_ROOT / "runs/acceptance/autoformer-aznas-random-8000"
AUTOFORMER_STATUS = AUTOFORMER_ROOT / "status.json"
LIVE_PATH = PANEL_DIR / "live.json"
DEFAULT_TOTAL = 8_000


def now_shanghai():
    return datetime.now(TIME_ZONE)


def iso_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, TIME_ZONE).isoformat(timespec="seconds")


def read_json(path, warnings, label):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        warnings.append(f"{label} 暂缺")
    except (OSError, json.JSONDecodeError):
        warnings.append(f"{label} 暂不可读，保留下一轮重试")
    return None


def public_status(source, path):
    if not isinstance(source, dict):
        return {"status": "unavailable", "updated_at": None, "detail": "状态源暂缺"}
    updated_at = source.get("updated_at")
    if not updated_at:
        try:
            updated_at = iso_from_timestamp(path.stat().st_mtime)
        except OSError:
            updated_at = None
    return {
        "status": source.get("status") or "unknown",
        "current": source.get("current"),
        "detail": source.get("detail"),
        "updated_at": updated_at,
        "execution_strategy": source.get("execution_strategy"),
        "gpu_uuids": source.get("gpu_uuids") if isinstance(source.get("gpu_uuids"), list) else [],
    }


def latest_search_states():
    states = []
    if not AUTOFORMER_ROOT.exists():
        return states
    for seed_dir in sorted(AUTOFORMER_ROOT.glob("seed-*")):
        candidates = list(seed_dir.glob("*/search-state.json"))
        if not candidates:
            continue
        ranked = []
        for candidate in candidates:
            try:
                manifest = json.loads((candidate.parent / "manifest.json").read_text(encoding="utf-8"))
                completed = manifest.get("status") == "completed"
                ranked.append((completed, candidate.stat().st_mtime, candidate))
            except (OSError, json.JSONDecodeError):
                continue
        if not ranked:
            continue
        latest = max(ranked, key=lambda item: (item[0], item[1]))[2]
        states.append((seed_dir.name.removeprefix("seed-"), latest))
    return states


def search_progress(seed, path, warnings, generated_at):
    state = read_json(path, warnings, f"AutoFormer seed {seed} search-state.json")
    if not isinstance(state, dict):
        return {
            "seed": int(seed) if seed.isdigit() else seed,
            "status": "unavailable",
            "candidates": 0,
            "evaluations": 0,
            "unique_evaluations": 0,
            "cache_hits": 0,
            "total": DEFAULT_TOTAL,
            "rate_per_second": None,
            "eta_seconds": None,
            "updated_at": None,
            "stale": True,
        }
    identity = state.get("identity") if isinstance(state.get("identity"), dict) else {}
    unique_evaluations = max(0, int(state.get("evaluations") or 0))
    population = state.get("population") if isinstance(state.get("population"), list) else []
    candidates = len(population)
    cache_hits = max(0, int(state.get("cache_hits") or 0))
    total = max(1, int(identity.get("population_size") or state.get("population_size") or DEFAULT_TOTAL))
    elapsed = float(state.get("elapsed_seconds") or 0.0)
    rate = candidates / elapsed if candidates > 0 and elapsed > 0 else None
    eta = (total - candidates) / rate if rate and candidates < total else 0 if candidates >= total else None
    manifest = read_json(path.parent / "manifest.json", warnings, f"AutoFormer seed {seed} manifest.json")
    manifest_status = manifest.get("status") if isinstance(manifest, dict) else None
    completed = (
        manifest_status == "completed"
        and int(state.get("completed_generation", -1)) >= 0
        and candidates >= total
    )
    status = "completed" if completed else "failed" if manifest_status == "failed" else "running"
    try:
        modified_at = path.stat().st_mtime
        updated_at = iso_from_timestamp(modified_at)
        stale = status == "running" and generated_at.timestamp() - modified_at > 90
    except OSError:
        updated_at = None
        stale = True
    return {
        "seed": int(seed) if seed.isdigit() else seed,
        "status": status,
        "manifest_status": manifest_status,
        "candidates": candidates,
        "evaluations": candidates,
        "unique_evaluations": unique_evaluations,
        "cache_hits": cache_hits,
        "total": total,
        "rate_per_second": round(rate, 3) if rate is not None else None,
        "eta_seconds": round(eta) if eta is not None else None,
        "updated_at": updated_at,
        "stale": stale,
    }


def gpu_snapshot(warnings):
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        warnings.append("nvidia-smi 暂不可用")
        return []
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            gpus.append({
                "index": int(fields[0]),
                "uuid": fields[1],
                "utilization_percent": float(fields[2]),
                "memory_used_mib": float(fields[3]),
                "memory_total_mib": float(fields[4]),
            })
        except ValueError:
            continue
    return gpus


def build_live_status():
    generated_at = now_shanghai()
    warnings = []
    darts_source = read_json(DARTS_STATUS, warnings, "DARTS status.json")
    autoformer_source = read_json(AUTOFORMER_STATUS, warnings, "AutoFormer status.json")
    discovered_states = dict(latest_search_states())
    expected_seeds = (
        [str(seed) for seed in autoformer_source.get("seeds", [])]
        if isinstance(autoformer_source, dict) and autoformer_source.get("seeds")
        else sorted(discovered_states)
    )
    population_per_seed = (
        int(autoformer_source.get("population_per_seed") or DEFAULT_TOTAL)
        if isinstance(autoformer_source, dict)
        else DEFAULT_TOTAL
    )
    seeds = []
    for seed in expected_seeds:
        path = discovered_states.get(seed)
        if path is not None:
            seeds.append(search_progress(seed, path, warnings, generated_at))
            continue
        warnings.append(f"AutoFormer seed {seed} search-state.json 暂缺")
        seeds.append({
            "seed": int(seed) if seed.isdigit() else seed,
            "status": "missing",
            "manifest_status": None,
            "candidates": 0,
            "evaluations": 0,
            "unique_evaluations": 0,
            "cache_hits": 0,
            "total": population_per_seed,
            "rate_per_second": None,
            "eta_seconds": None,
            "updated_at": None,
            "stale": False,
        })
    if not seeds:
        warnings.append("AutoFormer search-state.json 暂缺")
    autoformer_status = public_status(autoformer_source, AUTOFORMER_STATUS)
    supervisor_status = (
        autoformer_source.get("supervisor_terminal_status")
        if isinstance(autoformer_source, dict)
        else None
    ) or autoformer_status["status"]
    supervisor_detail = (
        autoformer_source.get("supervisor_terminal_detail")
        if isinstance(autoformer_source, dict)
        else None
    )
    seed_complete = bool(seeds) and all(seed["status"] == "completed" for seed in seeds)
    reconciled = bool(
        isinstance(autoformer_source, dict)
        and autoformer_source.get("cohort_validation_sha256")
    )
    if seed_complete:
        autoformer_status["status"] = "completed"
        if supervisor_status != "completed" and not reconciled:
            warnings.append(
                "AutoFormer 三个 seed 产物均 completed，但 supervisor status 非 completed；需执行产物归并审计"
            )
    autoformer_status.update({
        "supervisor_status": supervisor_status,
        "supervisor_detail": supervisor_detail,
        "reconciled": reconciled,
        "cohort_status_source": "seed_manifests" if seed_complete else "supervisor",
        "candidates_total": sum(seed["candidates"] for seed in seeds),
        "evaluations_total": sum(seed["candidates"] for seed in seeds),
        "unique_evaluations_total": sum(seed["unique_evaluations"] for seed in seeds),
        "cache_hits_total": sum(seed["cache_hits"] for seed in seeds),
        "target_total": sum(seed["total"] for seed in seeds),
        "seeds": seeds,
    })
    return {
        "schema_version": 1,
        "time_zone": "Asia/Shanghai",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "stale_after_seconds": 45,
        "darts": public_status(darts_source, DARTS_STATUS),
        "autoformer": autoformer_status,
        "gpus": gpu_snapshot(warnings),
        "warnings": warnings,
    }


def write_live_status(payload):
    temporary = LIVE_PATH.with_name(f".{LIVE_PATH.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, LIVE_PATH)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_once():
    write_live_status(build_live_status())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate panel/live.json from acceptance run state.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Generate live.json once and exit.")
    mode.add_argument("--watch", action="store_true", help="Regenerate live.json continuously.")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch interval in seconds (default: 15).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    if args.once:
        run_once()
        return
    while True:
        started = time.monotonic()
        run_once()
        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
