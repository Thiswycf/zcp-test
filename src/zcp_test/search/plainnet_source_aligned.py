from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from zcp_test.artifacts import JsonlWriter
from zcp_test.models.plainnet import (
    AZNAS_COMMIT,
    INITIAL_STRUCTURE,
    PlainNetBlockSpec,
    _official_conv_bn_ops,
    _official_depthwise_bn_ops,
    _round_channels,
    canonical_plainnet_structure,
    parse_plainnet_structure,
    reconnect_plainnet_blocks,
)
from zcp_test.proxies.az_nas import PLAINNET_COMPONENTS, log_rank_aggregate
from zcp_test.spaces.base import SearchSpace
from zcp_test.types import Architecture


CONTROLLER_ID = "plainnet_source_aligned"
CONTROLLER_VERSION = "aznas-5e6683-plainnet-controller-v1"
CONTROLLER_FIDELITY = "source_aligned_control_flow_port"
ONE_PERCENT_CONTROLLER_FIDELITY = (
    "source_aligned_control_flow_port_truncated_one_percent_budget"
)
SOURCE_URL = (
    "https://github.com/cvlab-yonsei/AZ-NAS/blob/"
    f"{AZNAS_COMMIT}/ImageNet_MBV2/evolution_search_az.py"
)
STATE_SCHEMA_VERSION = "plainnet-source-aligned-1.0"
SOURCE_VALID_CANDIDATES = 100_000
SOURCE_PARENT_POOL = 1_024
SOURCE_INITIAL_CANDIDATES = 11
SOURCE_SPLIT_THRESHOLD = 6


@dataclass(frozen=True)
class PlainNetTargetProfile:
    target_id: str
    flops_budget: int
    max_layers: int


TARGET_PROFILES = {
    "450m": PlainNetTargetProfile("450m", 450_000_000, 14),
    "600m": PlainNetTargetProfile("600m", 600_000_000, 14),
    "1g": PlainNetTargetProfile("1g", 1_000_000_000, 16),
}


@dataclass
class PlainNetSearchCandidate:
    architecture: Architecture
    components: dict[str, float]
    score: float
    score_at_acceptance: float
    parent_architecture_id: str | None
    parent_index: int | None
    parent_selection: str
    replacements_requested: int
    replacement_block_ids: tuple[int, ...]
    official_flops: int
    num_layers: int
    cache_hit: bool
    record: dict[str, Any]


def resolve_target_profile(target: str) -> PlainNetTargetProfile:
    normalized = str(target).strip().lower()
    try:
        return TARGET_PROFILES[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown PlainNet FLOPs target {target!r}; choices={tuple(TARGET_PROFILES)}"
        ) from error


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _serialized_row(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                payload = _serialized_row(row)
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _numpy_state_payload(state: tuple[Any, ...]) -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = state
    return {
        "name": str(name),
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_state(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(payload["name"]),
        np.asarray(payload["keys"], dtype=np.uint32),
        int(payload["position"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def load_plainnet_search_state(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported or invalid PlainNet source-aligned search state")
    return payload


def _source_round_channels(value: float) -> int:
    return max(8, round(value / 8.0) * 8)


def _channel_choices(channels: int) -> tuple[int, ...]:
    values = (
        channels * 2.5,
        channels * 2,
        channels * 1.5,
        channels * 1.25,
        channels,
        channels / 1.25,
        channels / 1.5,
        channels / 2,
        channels / 2.5,
    )
    return tuple(sorted({_source_round_channels(max(8, value)) for value in values}, reverse=True))


def _sublayer_choices(sub_layers: int) -> tuple[int, ...]:
    values = {
        max(0, round(value))
        for value in (
            sub_layers,
            sub_layers + 1,
            sub_layers + 2,
            sub_layers - 1,
            sub_layers - 2,
        )
    }
    return tuple(sorted((value for value in values if value > 0), reverse=True))


def _replacement_options(block: PlainNetBlockSpec) -> tuple[PlainNetBlockSpec, ...]:
    if block.kind == "conv":
        output_channels = (
            (block.out_channels,)
            if block.kernel_size == 1
            else _channel_choices(block.out_channels)
        )
        options = {
            replace(block, out_channels=channels, sub_layers=1)
            for channels in output_channels
        }
    else:
        options = {
            PlainNetBlockSpec(
                kind="residual",
                in_channels=block.in_channels,
                out_channels=out_channels,
                stride=block.stride,
                sub_layers=sub_layers,
                kernel_size=kernel_size,
                expansion=expansion,
                bottleneck_channels=bottleneck_channels,
            )
            for expansion in (1, 2, 4, 6)
            for kernel_size in (3, 5, 7)
            for out_channels in _channel_choices(block.out_channels)
            for sub_layers in _sublayer_choices(block.sub_layers)
            for bottleneck_channels in _channel_choices(
                int(block.bottleneck_channels or block.out_channels)
            )
        }
    return tuple(sorted(options, key=lambda option: option.encode()))


def _split_blocks(
    blocks: Sequence[PlainNetBlockSpec],
    threshold: int = SOURCE_SPLIT_THRESHOLD,
) -> tuple[PlainNetBlockSpec, ...]:
    split: list[PlainNetBlockSpec] = []
    for block in blocks:
        if block.kind != "residual" or block.sub_layers < threshold:
            split.append(block)
            continue
        first_layers = threshold // 2
        second_layers = block.sub_layers - first_layers
        split.append(replace(block, sub_layers=first_layers))
        split.append(
            replace(
                block,
                in_channels=block.out_channels,
                stride=1,
                sub_layers=second_layers,
            )
        )
    return reconnect_plainnet_blocks(split)


def source_block_replacement(
    space: SearchSpace,
    architecture: Architecture,
    rng: random.Random,
    replacements: int,
) -> tuple[Architecture, tuple[int, ...]]:
    if replacements <= 0:
        raise ValueError("PlainNet source-aligned replacement count must be positive")
    blocks = list(parse_plainnet_structure(str(architecture.spec["structure"])))
    selected: set[int] = set()
    for _ in range(replacements):
        block_id = rng.randint(0, len(blocks) - 1)
        if block_id in selected:
            continue
        selected.add(block_id)
        blocks[block_id] = rng.choice(_replacement_options(blocks[block_id]))
    connected = _split_blocks(reconnect_plainnet_blocks(blocks))
    candidate = space.canonicalize(
        {
            "structure": canonical_plainnet_structure(connected),
            "resolution": 224,
        }
    )
    return candidate, tuple(sorted(selected))


def plainnet_num_layers(architecture: Architecture) -> int:
    return sum(
        block.sub_layers
        for block in parse_plainnet_structure(str(architecture.spec["structure"]))
    )


def plainnet_official_flops(
    architecture: Architecture,
    *,
    resolution: int = 224,
    classes: int = 1_000,
) -> int:
    blocks = parse_plainnet_structure(str(architecture.spec["structure"]))
    current_resolution = int(resolution)
    operations = 0.0
    for block in blocks:
        if block.kind == "conv":
            channels_in = block.in_channels
            stride = block.stride
            for _ in range(block.sub_layers):
                operations += (
                    channels_in
                    * block.out_channels
                    * block.kernel_size**2
                    * current_resolution**2
                    // stride**2
                )
                current_resolution //= stride
                operations += current_resolution**2 * block.out_channels
                channels_in = block.out_channels
                stride = 1
            continue

        if block.expansion is None or block.bottleneck_channels is None:
            raise ValueError("PlainNet residual block is incomplete")
        channels_in = block.in_channels
        stride = block.stride
        for sub_layer in range(block.sub_layers):
            hidden_channels = _round_channels(
                block.bottleneck_channels * block.expansion
            )
            operations += _official_conv_bn_ops(
                channels_in, hidden_channels, 1, 1, current_resolution
            )
            operations += _official_depthwise_bn_ops(
                hidden_channels, block.kernel_size, stride, current_resolution
            )
            current_resolution //= stride
            operations += _official_conv_bn_ops(
                hidden_channels,
                block.bottleneck_channels,
                1,
                1,
                current_resolution,
            )
            if (
                sub_layer == 0
                or stride > 1
                or channels_in != block.bottleneck_channels
            ):
                projected_resolution = current_resolution / stride
                operations += (
                    channels_in
                    * block.bottleneck_channels
                    * projected_resolution**2
                    + projected_resolution**2 * block.bottleneck_channels
                )

            hidden_channels = _round_channels(block.out_channels * block.expansion)
            operations += _official_conv_bn_ops(
                block.bottleneck_channels,
                hidden_channels,
                1,
                1,
                current_resolution,
            )
            operations += _official_depthwise_bn_ops(
                hidden_channels, block.kernel_size, 1, current_resolution
            )
            operations += _official_conv_bn_ops(
                hidden_channels,
                block.out_channels,
                1,
                1,
                current_resolution,
            )
            if block.bottleneck_channels != block.out_channels:
                operations += (
                    block.bottleneck_channels
                    * block.out_channels
                    * current_resolution**2
                    + current_resolution**2 * block.out_channels
                )
            channels_in = block.out_channels
            stride = 1
    operations += blocks[-1].out_channels * int(classes)
    return int(operations)


class PlainNetSourceAlignedSearch:
    def __init__(
        self,
        *,
        space: SearchSpace,
        evaluator: Callable[[Architecture], Mapping[str, float]],
        writer: JsonlWriter,
        state_path: str | Path,
        seed: int,
        target: PlainNetTargetProfile,
        valid_candidates: int = SOURCE_VALID_CANDIDATES,
        parent_pool: int = SOURCE_PARENT_POOL,
        classes: int = 1_000,
        controller_fidelity: str = CONTROLLER_FIDELITY,
        search_budget_protocol: str = "upstream_full_100k",
        search_budget_fraction: float = 1.0,
        record_metadata: Mapping[str, Any] | None = None,
        state_identity: Mapping[str, Any] | None = None,
        resume_state: Mapping[str, Any] | None = None,
        resume_journal_path: str | Path | None = None,
        resource_evaluator: Callable[[Architecture], int] | None = None,
        mutator: Callable[
            [SearchSpace, Architecture, random.Random, int],
            tuple[Architecture, tuple[int, ...]],
        ] = source_block_replacement,
        max_attempts: int | None = None,
    ) -> None:
        if space.search_space_id != "zennas_plainnet_mbv2":
            raise ValueError(
                "PlainNet source-aligned controller requires zennas_plainnet_mbv2"
            )
        if valid_candidates <= 0:
            raise ValueError("valid_candidates must be positive")
        if parent_pool < 2:
            raise ValueError("parent_pool must be at least 2")
        if classes <= 0:
            raise ValueError("classes must be positive")
        if max_attempts is not None and max_attempts < valid_candidates:
            raise ValueError("max_attempts cannot be smaller than valid_candidates")
        self.space = space
        self.evaluator = evaluator
        self.writer = writer
        self.state_path = Path(state_path)
        self.seed = int(seed)
        self.target = target
        self.valid_candidates = int(valid_candidates)
        self.parent_pool = int(parent_pool)
        self.classes = int(classes)
        self.controller_fidelity = str(controller_fidelity)
        self.search_budget_protocol = str(search_budget_protocol)
        self.search_budget_fraction = float(search_budget_fraction)
        self.record_metadata = dict(record_metadata or {})
        self.resource_evaluator = resource_evaluator or (
            lambda architecture: plainnet_official_flops(
                architecture, resolution=224, classes=self.classes
            )
        )
        self.mutator = mutator
        self.max_attempts = max_attempts
        self.python_rng = random.Random(self.seed)
        self.numpy_rng = np.random.RandomState(self.seed)
        self.initial_architecture = self.space.canonicalize(
            {"structure": INITIAL_STRUCTURE, "resolution": 224}
        )
        fixed_identity = {
            "controller_id": CONTROLLER_ID,
            "controller_version": CONTROLLER_VERSION,
            "controller_fidelity": self.controller_fidelity,
            "search_budget_protocol": self.search_budget_protocol,
            "search_budget_fraction": self.search_budget_fraction,
            "implementation_source": SOURCE_URL,
            "implementation_commit": AZNAS_COMMIT,
            "search_space_id": self.space.search_space_id,
            "seed": self.seed,
            "valid_candidates": self.valid_candidates,
            "parent_pool": self.parent_pool,
            "top_parent_pool": self.parent_pool - 1,
            "initial_candidates": SOURCE_INITIAL_CANDIDATES,
            "initial_replacements": 1,
            "subsequent_replacements": 2,
            "split_layer_threshold": SOURCE_SPLIT_THRESHOLD,
            "crossover": False,
            "component_names": list(PLAINNET_COMPONENTS),
            "aggregator": "az_nas_log_rank",
            "flops_target": target.target_id,
            "flops_budget": target.flops_budget,
            "max_layers": target.max_layers,
            "input_source": "random",
            "batch_size": 64,
            "input_size": 224,
            "classes": self.classes,
        }
        supplied_identity = dict(state_identity or {})
        conflicts = {
            key: (supplied_identity[key], value)
            for key, value in fixed_identity.items()
            if key in supplied_identity and supplied_identity[key] != value
        }
        if conflicts:
            raise ValueError(f"PlainNet source protocol identity conflicts: {conflicts}")
        self.state_identity = {**supplied_identity, **fixed_identity}
        self.history: list[PlainNetSearchCandidate] = []
        self.cache: dict[str, dict[str, float]] = {}
        self.attempts = 0
        self.evaluations = 0
        self.cache_hits = 0
        self.rejected_flops = 0
        self.rejected_layers = 0
        self._journal_digest = hashlib.sha256()
        self._summary_written = False
        self._completed = False
        self.resume_journal_path = (
            Path(resume_journal_path) if resume_journal_path is not None else None
        )
        if resume_state is None:
            if self.writer.path.exists() and self.writer.path.stat().st_size:
                raise ValueError("New PlainNet search requires an empty search journal")
        else:
            self._restore(resume_state)

    def _validate_components(self, value: Mapping[str, float]) -> dict[str, float]:
        components = {str(name): float(component) for name, component in value.items()}
        if set(components) != set(PLAINNET_COMPONENTS):
            raise ValueError(
                "PlainNet AZ-NAS evaluator must return exactly "
                f"{PLAINNET_COMPONENTS}"
            )
        if not all(math.isfinite(component) for component in components.values()):
            raise ValueError("PlainNet AZ-NAS evaluator returned a non-finite component")
        return components

    def _rerank(self) -> None:
        scores = log_rank_aggregate(
            [candidate.components for candidate in self.history],
            PLAINNET_COMPONENTS,
        )
        for candidate, score in zip(self.history, scores, strict=True):
            candidate.score = float(score)

    def _parent(self) -> tuple[Architecture, int | None, str, int]:
        count = len(self.history)
        if count <= 10:
            return self.initial_architecture, None, "initial_structure", 1
        if count < self.parent_pool - 1:
            parent_index = self.python_rng.randint(0, count - 1)
            return (
                self.history[parent_index].architecture,
                parent_index,
                "all_history_random",
                2,
            )
        scores = np.asarray([candidate.score for candidate in self.history])
        top_indices = np.argsort(scores, axis=0)[-self.parent_pool + 1 :]
        parent_index = int(self.numpy_rng.choice(top_indices))
        return (
            self.history[parent_index].architecture,
            parent_index,
            "top_pool_random",
            2,
        )

    def _candidate_from_record(self, row: Mapping[str, Any]) -> PlainNetSearchCandidate:
        architecture = self.space.canonicalize(
            {
                "structure": row["architecture"]["structure"],
                "resolution": row["architecture"].get("resolution", 224),
            }
        )
        if architecture.architecture_id != row.get("architecture_id"):
            raise ValueError("PlainNet journal architecture ID is not canonical")
        components = self._validate_components(row["components"])
        return PlainNetSearchCandidate(
            architecture=architecture,
            components=components,
            score=float(row.get("score", 0.0)),
            score_at_acceptance=float(row.get("score_at_acceptance", row.get("score", 0.0))),
            parent_architecture_id=row.get("parent_architecture_id"),
            parent_index=row.get("parent_index"),
            parent_selection=str(row["parent_selection"]),
            replacements_requested=int(row["replacements_requested"]),
            replacement_block_ids=tuple(int(value) for value in row["replacement_block_ids"]),
            official_flops=int(row["official_flops"]),
            num_layers=int(row["num_layers"]),
            cache_hit=bool(row["cache_hit"]),
            record=dict(row),
        )

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported PlainNet source-aligned state schema")
        if state.get("identity") != self.state_identity:
            raise ValueError("PlainNet source-aligned search state identity mismatch")
        accepted_count = int(state.get("accepted_count", -1))
        summary_written = bool(state.get("summary_written", False))
        committed_rows = accepted_count + int(summary_written)
        journal_source = self.resume_journal_path or self.writer.path
        payloads = (
            journal_source.read_bytes().splitlines(keepends=True)
            if journal_source.exists()
            else []
        )
        if len(payloads) < committed_rows:
            raise ValueError("PlainNet search journal is shorter than committed state")
        committed_payloads = payloads[:committed_rows]
        digest = hashlib.sha256(b"".join(committed_payloads)).hexdigest()
        if digest != state.get("journal_sha256"):
            raise ValueError("PlainNet search journal checksum does not match state")
        committed: list[dict[str, Any]] = []
        for line_number, payload in enumerate(committed_payloads, start=1):
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"PlainNet committed journal is invalid at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError("PlainNet committed journal rows must be objects")
            committed.append(row)
        if journal_source != self.writer.path or len(payloads) > committed_rows:
            digest = _atomic_jsonl(self.writer.path, committed)
        self._journal_digest = hashlib.sha256()
        for payload in committed_payloads:
            self._journal_digest.update(payload)
        candidate_rows = committed[:accepted_count]
        if any(row.get("record_kind") != "candidate" for row in candidate_rows):
            raise ValueError("PlainNet committed journal prefix contains non-candidates")
        self.history = [self._candidate_from_record(row) for row in candidate_rows]
        self._rerank() if self.history else None
        for candidate in self.history:
            previous = self.cache.setdefault(
                candidate.architecture.architecture_id, candidate.components
            )
            if previous != candidate.components:
                raise ValueError("PlainNet cache maps one architecture to conflicting components")
        self.attempts = int(state["attempts"])
        self.evaluations = int(state["evaluations"])
        self.cache_hits = int(state["cache_hits"])
        self.rejected_flops = int(state["rejected_flops"])
        self.rejected_layers = int(state["rejected_layers"])
        if self.evaluations != len(self.cache):
            raise ValueError("PlainNet state evaluation count does not match cache")
        if self.cache_hits != len(self.history) - self.evaluations:
            raise ValueError("PlainNet state cache-hit count does not match history")
        self.python_rng.setstate(_tuple_tree(state["python_rng_state"]))
        self.numpy_rng.set_state(_restore_numpy_state(state["numpy_rng_state"]))
        self._summary_written = summary_written
        self._completed = state.get("status") == "completed"
        if self._completed and (
            not self._summary_written or accepted_count != self.valid_candidates
        ):
            raise ValueError("PlainNet completed state is internally inconsistent")

    def _state_payload(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "identity": self.state_identity,
            "status": status,
            "accepted_count": len(self.history),
            "attempts": self.attempts,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "rejected_flops": self.rejected_flops,
            "rejected_layers": self.rejected_layers,
            "summary_written": self._summary_written,
            "journal_rows": len(self.history) + int(self._summary_written),
            "journal_sha256": self._journal_digest.hexdigest(),
            "python_rng_state": self.python_rng.getstate(),
            "numpy_rng_state": _numpy_state_payload(self.numpy_rng.get_state()),
            "best_architecture_id": (
                self.best().architecture.architecture_id if self.history else None
            ),
            "best_score": self.best().score if self.history else None,
        }

    def _save_state(self, status: str = "running") -> None:
        _atomic_json(self.state_path, self._state_payload(status))

    def _append(self, row: Mapping[str, Any]) -> None:
        self.writer.append(row)
        self._journal_digest.update(_serialized_row(row))

    def _accept(
        self,
        architecture: Architecture,
        parent: Architecture,
        parent_index: int | None,
        parent_selection: str,
        replacements_requested: int,
        replacement_block_ids: tuple[int, ...],
        official_flops: int,
        num_layers: int,
    ) -> None:
        architecture_id = architecture.architecture_id
        cache_hit = architecture_id in self.cache
        if cache_hit:
            self.cache_hits += 1
            components = dict(self.cache[architecture_id])
        else:
            components = self._validate_components(self.evaluator(architecture))
            self.cache[architecture_id] = components
            self.evaluations += 1
        candidate = PlainNetSearchCandidate(
            architecture=architecture,
            components=components,
            score=0.0,
            score_at_acceptance=0.0,
            parent_architecture_id=(
                None if parent_index is None else parent.architecture_id
            ),
            parent_index=parent_index,
            parent_selection=parent_selection,
            replacements_requested=replacements_requested,
            replacement_block_ids=replacement_block_ids,
            official_flops=official_flops,
            num_layers=num_layers,
            cache_hit=cache_hit,
            record={},
        )
        self.history.append(candidate)
        self._rerank()
        candidate.score_at_acceptance = candidate.score
        row = {
            **self.record_metadata,
            "record_kind": "candidate",
            "controller_id": CONTROLLER_ID,
            "controller_version": CONTROLLER_VERSION,
            "controller_fidelity": self.controller_fidelity,
            "search_budget_protocol": self.search_budget_protocol,
            "search_budget_fraction": self.search_budget_fraction,
            "source_url": SOURCE_URL,
            "implementation_commit": AZNAS_COMMIT,
            "accepted_index": len(self.history) - 1,
            "attempt": self.attempts,
            "search_space_id": architecture.search_space_id,
            "architecture_id": architecture.architecture_id,
            "architecture": dict(architecture.spec),
            "parent_architecture_id": candidate.parent_architecture_id,
            "parent_index": parent_index,
            "parent_selection": parent_selection,
            "operation": "source_block_replacement",
            "crossover": False,
            "replacements_requested": replacements_requested,
            "replacement_block_ids": list(replacement_block_ids),
            "official_flops": official_flops,
            "flops_budget": self.target.flops_budget,
            "num_layers": num_layers,
            "max_layers": self.target.max_layers,
            "components": components,
            "score": candidate.score,
            "score_at_acceptance": candidate.score_at_acceptance,
            "score_semantics": "post_insert_full_history_log_rank",
            "cache_hit": cache_hit,
            "cumulative_evaluations": self.evaluations,
            "cumulative_cache_hits": self.cache_hits,
            "cumulative_rejected_flops": self.rejected_flops,
            "cumulative_rejected_layers": self.rejected_layers,
        }
        candidate.record = row
        self._append(row)
        self._save_state()

    def best(self) -> PlainNetSearchCandidate:
        if not self.history:
            raise ValueError("PlainNet source-aligned search has no accepted candidates")
        return min(
            self.history,
            key=lambda candidate: (
                -candidate.score,
                candidate.architecture.architecture_id,
            ),
        )

    def _finalize(self) -> PlainNetSearchCandidate:
        self._rerank()
        for candidate in self.history:
            candidate.record = {
                **candidate.record,
                "score": candidate.score,
                "score_semantics": "final_full_history_log_rank",
            }
        best = self.best()
        scores = sorted(candidate.score for candidate in self.history)
        summary = {
            **self.record_metadata,
            "record_kind": "search_summary",
            "controller_id": CONTROLLER_ID,
            "controller_version": CONTROLLER_VERSION,
            "controller_fidelity": self.controller_fidelity,
            "search_budget_protocol": self.search_budget_protocol,
            "search_budget_fraction": self.search_budget_fraction,
            "source_url": SOURCE_URL,
            "implementation_commit": AZNAS_COMMIT,
            "valid_candidates": len(self.history),
            "attempts": self.attempts,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "rejected_flops": self.rejected_flops,
            "rejected_layers": self.rejected_layers,
            "best_architecture_id": best.architecture.architecture_id,
            "best_score": best.score,
            "mean_score": sum(scores) / len(scores),
            "q25": scores[round((len(scores) - 1) * 0.25)],
            "q50": scores[round((len(scores) - 1) * 0.50)],
            "q75": scores[round((len(scores) - 1) * 0.75)],
            "diversity": len(
                {candidate.architecture.architecture_id for candidate in self.history}
            )
            / len(self.history),
            "flops_target": self.target.target_id,
            "flops_budget": self.target.flops_budget,
            "max_layers": self.target.max_layers,
            "crossover": False,
        }
        rows = [candidate.record for candidate in self.history] + [summary]
        digest = _atomic_jsonl(self.writer.path, rows)
        self._journal_digest = hashlib.sha256()
        for row in rows:
            self._journal_digest.update(_serialized_row(row))
        if self._journal_digest.hexdigest() != digest:
            raise RuntimeError("PlainNet finalized journal checksum mismatch")
        self._summary_written = True
        self._completed = True
        self._save_state("completed")
        return best

    def run(
        self,
        *,
        stop_after_accepted: int | None = None,
    ) -> PlainNetSearchCandidate | None:
        if self._completed:
            return self.best()
        limit = self.valid_candidates
        if stop_after_accepted is not None:
            if stop_after_accepted < len(self.history):
                raise ValueError("stop_after_accepted precedes restored progress")
            limit = min(limit, int(stop_after_accepted))
        while len(self.history) < limit:
            if self.max_attempts is not None and self.attempts >= self.max_attempts:
                raise RuntimeError(
                    "PlainNet source-aligned search exhausted max_attempts before completion"
                )
            parent, parent_index, parent_selection, replacements = self._parent()
            architecture, replacement_ids = self.mutator(
                self.space, parent, self.python_rng, replacements
            )
            self.attempts += 1
            num_layers = plainnet_num_layers(architecture)
            if num_layers > self.target.max_layers:
                self.rejected_layers += 1
                self._save_state()
                continue
            official_flops = int(self.resource_evaluator(architecture))
            if official_flops > self.target.flops_budget:
                self.rejected_flops += 1
                self._save_state()
                continue
            self._accept(
                architecture,
                parent,
                parent_index,
                parent_selection,
                replacements,
                replacement_ids,
                official_flops,
                num_layers,
            )
        if len(self.history) == self.valid_candidates:
            return self._finalize()
        return None


__all__ = [
    "CONTROLLER_FIDELITY",
    "ONE_PERCENT_CONTROLLER_FIDELITY",
    "CONTROLLER_ID",
    "CONTROLLER_VERSION",
    "PlainNetSearchCandidate",
    "PlainNetSourceAlignedSearch",
    "PlainNetTargetProfile",
    "SOURCE_PARENT_POOL",
    "SOURCE_VALID_CANDIDATES",
    "TARGET_PROFILES",
    "load_plainnet_search_state",
    "plainnet_num_layers",
    "plainnet_official_flops",
    "resolve_target_profile",
    "source_block_replacement",
]
