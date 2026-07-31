from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
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
    found = False
    with required["search"].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_kind") == "candidate" and row.get("architecture_id") == best_id:
                found = True
                break
    if not found:
        raise ValueError("Best architecture is absent from search.jsonl candidate records")
    provenance = {
        "search_run_id": manifest.get("run_id", run.name),
        "search_manifest_sha256": file_sha256(required["manifest"]),
        "search_config_sha256": file_sha256(required["config"]),
        "search_jsonl_sha256": file_sha256(required["search"]),
        "search_identity": identity,
    }
    return best, provenance


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
            selected, "zcp_selected", selected_resources, {**common, **search_provenance}
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


__all__ = [
    "ResourceMeasurement",
    "freeze_training_candidates",
    "measure_architecture_resources",
]
