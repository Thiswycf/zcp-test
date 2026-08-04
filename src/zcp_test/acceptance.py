from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zcp_test.artifacts.run import file_sha256, project_now_iso
from zcp_test.config import load_config
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.types import Architecture


@dataclass(frozen=True)
class ResourceMeasurement:
    parameters: int
    compute_value: int
    compute_metric: str
    generic_flops: bool
    input_size: int


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_search_selection(
    search_run: str | Path, expected_space: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = Path(search_run).expanduser().resolve()
    required = {
        "manifest": run / "manifest.json",
        "config": run / "config.yaml",
        "search": run / "search.jsonl",
        "state": run / "search-state.json",
        "best": run / "best_architecture.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Search run is missing required files: {', '.join(missing)}")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Candidate freezing requires a completed search run")
    config = load_config(required["config"])
    identity = config.get("search_identity")
    if not isinstance(identity, dict):
        raise ValueError("Search run does not contain a versioned search_identity")
    if identity.get("search_space_id") != expected_space:
        raise ValueError(
            "Search-space mismatch: "
            f"{identity.get('search_space_id')!r} != {expected_space!r}"
        )
    for field in ("proxy_id", "proxy_version", "input_fingerprint", "dataset", "seed"):
        if identity.get(field) in (None, ""):
            raise ValueError(f"Search identity is missing {field}")
    best = json.loads(required["best"].read_text(encoding="utf-8"))
    if best.get("search_space_id") != expected_space or not isinstance(best.get("spec"), dict):
        raise ValueError("best_architecture.json does not match the requested search space")
    best_id = str(best.get("architecture_id", ""))
    candidate_ids: set[str] = set()
    with required["search"].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_kind") == "candidate":
                candidate_ids.add(str(row.get("architecture_id", "")))
    if best_id not in candidate_ids:
        raise ValueError("Best architecture is absent from search.jsonl candidate records")
    state = json.loads(required["state"].read_text(encoding="utf-8"))
    if int(state.get("completed_generation", -1)) < 0:
        raise ValueError("Candidate freezing requires a completed search state")
    if state.get("identity") != identity:
        raise ValueError("Search state identity does not match resolved search config")
    population = state.get("population")
    if not isinstance(population, list) or not population:
        raise ValueError("Search state does not contain a final population")
    scored: list[tuple[float, str, dict[str, Any]]] = []
    architecture_by_id: dict[str, dict[str, Any]] = {}
    for entry in population:
        if not isinstance(entry, dict) or not isinstance(entry.get("architecture"), dict):
            raise ValueError("Search state contains an invalid population entry")
        architecture = entry["architecture"]
        architecture_id = str(architecture.get("architecture_id", ""))
        try:
            score = float(entry["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Search state contains an invalid candidate score") from error
        if not architecture_id or not math.isfinite(score):
            raise ValueError("Search state contains a non-finite or unidentified candidate")
        previous = architecture_by_id.setdefault(architecture_id, architecture)
        if previous != architecture:
            raise ValueError("Search state maps one architecture ID to conflicting specifications")
        scored.append((score, architecture_id, architecture))
    maximum = max(score for score, _architecture_id, _architecture in scored)
    tied = sorted(
        {
            architecture_id: architecture
            for score, architecture_id, architecture in scored
            if score == maximum
        }.items()
    )
    tied_ids = [architecture_id for architecture_id, _architecture in tied]
    if best_id not in tied_ids:
        raise ValueError("best_architecture.json is not a maximum-score final candidate")
    stable_id, stable_architecture = tied[0]
    if stable_id not in candidate_ids:
        raise ValueError("Stable best architecture is absent from search.jsonl candidate records")
    best = {
        "search_space_id": stable_architecture.get("search_space_id"),
        "architecture_id": stable_id,
        "benchmark_index": stable_architecture.get("benchmark_index"),
        "spec": stable_architecture.get("spec"),
    }
    provenance = {
        "search_run_id": manifest.get("run_id", run.name),
        "search_manifest_sha256": file_sha256(required["manifest"]),
        "search_config_sha256": file_sha256(required["config"]),
        "search_jsonl_sha256": file_sha256(required["search"]),
        "search_state_sha256": file_sha256(required["state"]),
        "search_identity": identity,
        "best_selection": {
            "strategy": "maximum_score_then_architecture_id_ascending_v1",
            "maximum_score": maximum,
            "maximum_score_tie_count": len(tied),
            "best_file_architecture_id": best_id,
            "selected_architecture_id": stable_id,
        },
    }
    return best, provenance


_COHORT_IDENTITY_FIELDS = (
    "search_space_id",
    "model_fidelity",
    "model_profile",
    "implementation_commit",
    "proxy_id",
    "proxy_version",
    "proxy_direction",
    "aggregator",
    "model_initialization_protocol",
    "dataset",
    "input_source",
    "population_size",
    "elite_ratio",
    "batch_size",
    "input_size",
    "classes",
    "weight_mode",
)


def _validate_supporting_searches(
    primary: Mapping[str, Any],
    supporting: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    primary_identity = primary["search_identity"]
    primary_missing = [
        field for field in _COHORT_IDENTITY_FIELDS if field not in primary_identity
    ]
    if supporting and primary_missing:
        raise ValueError(
            "Primary search identity is incomplete for a cohort: "
            + ", ".join(primary_missing)
        )
    primary_seed = int(primary_identity["seed"])
    seen_seeds = {primary_seed}
    records = []
    top_candidates = [
        {
            "role": "primary_selection",
            "seed": primary_seed,
            "architecture_id": primary["best_selection"]["selected_architecture_id"],
        }
    ]
    for best, provenance in supporting:
        identity = provenance["search_identity"]
        missing = [field for field in _COHORT_IDENTITY_FIELDS if field not in identity]
        if missing:
            raise ValueError(
                "Supporting search identity is incomplete: " + ", ".join(missing)
            )
        mismatched = [
            field
            for field in _COHORT_IDENTITY_FIELDS
            if primary_identity.get(field) != identity.get(field)
        ]
        if mismatched:
            raise ValueError(
                "Supporting search protocol mismatch: " + ", ".join(mismatched)
            )
        seed = int(identity["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Search cohort contains duplicate seed {seed}")
        seen_seeds.add(seed)
        records.append(dict(provenance))
        top_candidates.append(
            {
                "role": "supporting_robustness_only",
                "seed": seed,
                "architecture_id": best["architecture_id"],
            }
        )
    return {
        "protocol": "predeclared_primary_run_supporting_seed_robustness_v1",
        "selection_rule": "Only the primary run selects zcp_selected; supporting runs are not averaged or cherry-picked.",
        "primary_seed": primary_seed,
        "supporting_seeds": sorted(seen_seeds - {primary_seed}),
        "supporting_searches": records,
        "top_candidates": top_candidates,
    }


def _build_training_model(
    space: Any,
    architecture: Architecture,
    training_config: Mapping[str, Any],
    classes: int,
) -> Any:
    if hasattr(space, "build_training_model"):
        return space.build_training_model(architecture, classes, training_config)
    return space.build_model(architecture, classes)


def measure_architecture_resources(
    space: Any,
    architecture: Architecture,
    training_config: Mapping[str, Any],
    classes: int,
) -> ResourceMeasurement:
    import torch

    model = _build_training_model(space, architecture, training_config, classes).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    input_size = int(architecture.spec.get("resolution", training_config["input_size"]))
    if hasattr(model, "official_complexity_ops"):
        compute_value = int(model.official_complexity_ops())
        compute_metric = "cream_autoformer_official_complexity_ops"
        generic_flops = False
    else:
        from thop import profile

        with torch.no_grad():
            macs, _ = profile(
                model,
                inputs=(torch.zeros(1, 3, input_size, input_size),),
                verbose=False,
            )
        compute_value = int(macs)
        compute_metric = "thop_macs"
        generic_flops = True
    return ResourceMeasurement(
        parameters=parameters,
        compute_value=compute_value,
        compute_metric=compute_metric,
        generic_flops=generic_flops,
        input_size=input_size,
    )


def _distance(left: ResourceMeasurement, right: ResourceMeasurement) -> float:
    if left.compute_metric != right.compute_metric:
        raise ValueError("Resource measurements use different compute protocols")
    return abs(math.log(left.parameters / right.parameters)) + abs(
        math.log(left.compute_value / right.compute_value)
    )


def _candidate_payload(
    architecture: Architecture,
    role: str,
    resources: ResourceMeasurement,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **architecture.to_dict(),
        "candidate_role": role,
        "resources": asdict(resources),
        "provenance": dict(provenance),
    }


def freeze_training_candidates(
    *,
    search_run: str | Path,
    training_config_path: str | Path,
    output: str | Path,
    seed: int,
    pool_size: int = 32,
    classes: int = 1000,
    supporting_search_runs: Sequence[str | Path] = (),
    measure: Callable[[Any, Architecture, Mapping[str, Any], int], ResourceMeasurement]
    | None = None,
) -> dict[str, Any]:
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")
    training_config_path = Path(training_config_path).expanduser().resolve()
    training_config = load_config(training_config_path)
    space_id = str(training_config["space"])
    load_builtin_spaces()
    space = SPACES.create(space_id)
    best, search_provenance = _load_search_selection(search_run, space_id)
    supporting = [
        _load_search_selection(supporting_run, space_id)
        for supporting_run in supporting_search_runs
    ]
    cohort = _validate_supporting_searches(search_provenance, supporting)
    selected = space.canonicalize(best["spec"])
    if selected.architecture_id != best["architecture_id"]:
        raise ValueError("Best architecture ID does not match current canonicalization")
    measure_fn = measure or measure_architecture_resources
    selected_resources = measure_fn(space, selected, training_config, classes)

    seen = {selected.architecture_id}
    fixed_random: Architecture | None = None
    pool: list[Architecture] = []
    attempt = 0
    while fixed_random is None or len(pool) < pool_size:
        candidate = space.sample(seed + attempt)
        attempt += 1
        if candidate.architecture_id in seen:
            if attempt > pool_size * 100:
                raise RuntimeError("Unable to sample enough unique candidate architectures")
            continue
        seen.add(candidate.architecture_id)
        if fixed_random is None:
            fixed_random = candidate
        else:
            pool.append(candidate)
    assert fixed_random is not None
    fixed_resources = measure_fn(space, fixed_random, training_config, classes)
    measured_pool = [
        (candidate, measure_fn(space, candidate, training_config, classes))
        for candidate in pool
    ]
    matched, matched_resources = min(
        measured_pool,
        key=lambda item: (
            _distance(selected_resources, item[1]),
            item[0].architecture_id,
        ),
    )
    match_distance = _distance(selected_resources, matched_resources)

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    common = {
        "frozen_at": project_now_iso(),
        "training_config_sha256": file_sha256(training_config_path),
        "seed": seed,
    }
    payloads = {
        "zcp_selected.json": _candidate_payload(
            selected,
            "zcp_selected",
            selected_resources,
            {**common, **search_provenance, "search_cohort": cohort},
        ),
        "fixed_random.json": _candidate_payload(
            fixed_random,
            "fixed_random",
            fixed_resources,
            {**common, "sampling_seed": seed},
        ),
        "params_flops_matched.json": _candidate_payload(
            matched,
            "params_flops_matched",
            matched_resources,
            {
                **common,
                "pool_size": pool_size,
                "pool_seed_start": seed + 1,
                "match_distance_log_l1": match_distance,
                "target_architecture_id": selected.architecture_id,
            },
        ),
    }
    for name, payload in payloads.items():
        _atomic_json(destination / name, payload)
    manifest = {
        "schema_version": "1.0",
        "search_space_id": space_id,
        "created_at": project_now_iso(),
        "training_config_sha256": common["training_config_sha256"],
        "search_provenance": search_provenance,
        "search_cohort": cohort,
        "resource_protocol": {
            "compute_metric": selected_resources.compute_metric,
            "generic_flops": selected_resources.generic_flops,
            "distance": "abs(log(params/target_params)) + abs(log(compute/target_compute))",
        },
        "candidates": {
            name: {
                "sha256": hashlib.sha256(
                    (destination / name).read_bytes()
                ).hexdigest(),
                "architecture_id": payload["architecture_id"],
                "role": payload["candidate_role"],
                "resources": payload["resources"],
            }
            for name, payload in payloads.items()
        },
    }
    _atomic_json(destination / "candidates-manifest.json", manifest)
    return {"output": str(destination), **manifest}


def _validate_zero_generation_search_run(
    search_run: str | Path,
    *,
    expected_space: str,
    expected_population: int,
    expected_components: Sequence[str],
) -> dict[str, Any]:
    run = Path(search_run).expanduser().resolve()
    best, provenance = _load_search_selection(run, expected_space)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not manifest.get("ended_at"):
        raise ValueError("Search run manifest is not terminal-complete")
    state = json.loads((run / "search-state.json").read_text(encoding="utf-8"))
    if int(state.get("completed_generation", -1)) != 0:
        raise ValueError("Search cohort reconciliation requires a completed generation-0 run")
    population = state.get("population")
    if not isinstance(population, list) or len(population) != expected_population:
        raise ValueError("Search state population size does not match the expected cohort size")
    rows = []
    with (run / "search.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Search JSONL contains invalid JSON at line {line_number}"
                ) from error
    candidates = [row for row in rows if row.get("record_kind") == "candidate"]
    summaries = [row for row in rows if row.get("record_kind") == "generation_summary"]
    if len(candidates) != expected_population or len(summaries) != 1:
        raise ValueError("Search JSONL must contain the expected candidates and one summary")
    component_names = set(expected_components)
    architecture_rows: dict[str, tuple[Any, Any]] = {}
    for row in candidates:
        architecture_id = str(row.get("architecture_id", ""))
        components = row.get("components")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Search candidate contains an invalid score") from error
        if not architecture_id or not math.isfinite(score):
            raise ValueError("Search candidate contains a non-finite or unidentified score")
        if not isinstance(components, dict) or set(components) != component_names:
            raise ValueError("Search candidate components do not match the expected protocol")
        if not all(math.isfinite(float(value)) for value in components.values()):
            raise ValueError("Search candidate contains a non-finite component")
        identity = (
            json.dumps(row.get("architecture"), sort_keys=True),
            json.dumps(components, sort_keys=True),
        )
        previous = architecture_rows.setdefault(architecture_id, identity)
        if previous != identity:
            raise ValueError("Duplicate architecture ID has conflicting search records")
    row_cache_hits = sum(bool(row.get("cache_hit")) for row in candidates)
    unique_evaluations = int(state.get("evaluations", -1))
    state_cache_hits = int(state.get("cache_hits", -1))
    expected_cache_hits = expected_population - unique_evaluations
    if expected_cache_hits < 0 or row_cache_hits != expected_cache_hits:
        raise ValueError("Search candidate cache-hit count does not match unique evaluations")
    if state_cache_hits != expected_cache_hits:
        raise ValueError("Search state cache-hit count does not match unique evaluations")
    summary = summaries[0]
    if (
        int(summary.get("cumulative_evaluations", -1)) != unique_evaluations
        or int(summary.get("cumulative_cache_hits", -1)) != state_cache_hits
    ):
        raise ValueError("Search summary counters do not match the final state")
    identity = provenance["search_identity"]
    return {
        "seed": int(identity["seed"]),
        "run": str(run),
        "run_id": manifest.get("run_id", run.name),
        "ended_at": manifest["ended_at"],
        "candidate_rows": len(candidates),
        "unique_architectures": len(architecture_rows),
        "unique_evaluations": unique_evaluations,
        "cache_hits": state_cache_hits,
        "summary_rows": len(summaries),
        "best_architecture_id": best["architecture_id"],
        "best_selection": provenance["best_selection"],
        "search_manifest_sha256": provenance["search_manifest_sha256"],
        "search_config_sha256": provenance["search_config_sha256"],
        "search_jsonl_sha256": provenance["search_jsonl_sha256"],
        "search_state_sha256": provenance["search_state_sha256"],
        "search_identity": identity,
    }


def reconcile_search_cohort(
    *,
    cohort_root: str | Path,
    search_runs: Sequence[str | Path],
    expected_space: str,
    expected_population: int,
    expected_seeds: Sequence[int],
    expected_components: Sequence[str],
) -> dict[str, Any]:
    if expected_population <= 0:
        raise ValueError("expected_population must be positive")
    if not search_runs:
        raise ValueError("At least one search run is required")
    expected_seed_set = {int(seed) for seed in expected_seeds}
    if len(expected_seed_set) != len(expected_seeds):
        raise ValueError("Expected cohort seeds must be unique")
    records = [
        _validate_zero_generation_search_run(
            run,
            expected_space=expected_space,
            expected_population=expected_population,
            expected_components=expected_components,
        )
        for run in search_runs
    ]
    actual_seeds = {record["seed"] for record in records}
    if actual_seeds != expected_seed_set:
        raise ValueError(
            f"Search cohort seeds do not match: {sorted(actual_seeds)} != {sorted(expected_seed_set)}"
        )
    root = Path(cohort_root).expanduser().resolve()
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    primary_seed = int(status.get("primary_selection_seed", min(expected_seed_set)))
    if primary_seed not in expected_seed_set:
        raise ValueError("Predeclared primary seed is not part of the expected cohort")
    primary = next(record for record in records if record["seed"] == primary_seed)
    supporting = [
        (
            {"architecture_id": record["best_architecture_id"]},
            {
                key: value
                for key, value in record.items()
                if key.startswith("search_") or key == "best_selection"
            },
        )
        for record in records
        if record["seed"] != primary_seed
    ]
    _validate_supporting_searches(
        {
            "search_identity": primary["search_identity"],
            "best_selection": primary["best_selection"],
        },
        supporting,
    )
    declared_primary = status.get("primary_selection_seed")
    declared_supporting = status.get("supporting_robustness_seeds")
    if declared_primary is not None and int(declared_primary) != primary_seed:
        raise ValueError("Cohort primary seed contradicts the predeclared status")
    if declared_supporting is not None and {
        int(seed) for seed in declared_supporting
    } != expected_seed_set - {primary_seed}:
        raise ValueError("Cohort supporting seeds contradict the predeclared status")
    validation = {
        "schema_version": "1.0",
        "validated_at": project_now_iso(),
        "status": "completed",
        "search_space_id": expected_space,
        "expected_population_per_seed": expected_population,
        "expected_components": list(expected_components),
        "primary_selection_seed": primary_seed,
        "supporting_robustness_seeds": sorted(expected_seed_set - {primary_seed}),
        "candidate_rows_total": sum(record["candidate_rows"] for record in records),
        "unique_evaluations_total": sum(record["unique_evaluations"] for record in records),
        "cache_hits_total": sum(record["cache_hits"] for record in records),
        "runs": sorted(records, key=lambda record: record["seed"]),
        "supervisor_status_before_reconciliation": status.get(
            "supervisor_terminal_status", status.get("status")
        ),
        "supervisor_detail_before_reconciliation": status.get(
            "supervisor_terminal_detail", status.get("detail")
        ),
    }
    validation_path = root / "cohort-validation.json"
    _atomic_json(validation_path, validation)
    validation_sha256 = file_sha256(validation_path)
    reconciled_status = {
        **status,
        "status": "completed",
        "detail": "all seed artifacts passed cohort reconciliation",
        "ended_at": max(record["ended_at"] for record in records),
        "updated_at": validation["validated_at"],
        "supervisor_terminal_status": status.get(
            "supervisor_terminal_status", status.get("status")
        ),
        "supervisor_terminal_detail": status.get(
            "supervisor_terminal_detail", status.get("detail")
        ),
        "reconciled_at": validation["validated_at"],
        "cohort_validation": str(validation_path),
        "cohort_validation_sha256": validation_sha256,
        "candidate_rows_total": validation["candidate_rows_total"],
        "unique_evaluations_total": validation["unique_evaluations_total"],
        "cache_hits_total": validation["cache_hits_total"],
    }
    _atomic_json(status_path, reconciled_status)
    return {**validation, "output": str(validation_path), "sha256": validation_sha256}


__all__ = [
    "ResourceMeasurement",
    "freeze_training_candidates",
    "measure_architecture_resources",
    "reconcile_search_cohort",
]
