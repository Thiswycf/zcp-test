from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import pytest

from zcp_test.artifacts import JsonlWriter, read_jsonl
from zcp_test.search import EvolutionSearch, load_search_state
from zcp_test.spaces.base import SearchSpace
from zcp_test.types import Architecture


class ScriptedConstraintSpace(SearchSpace):
    search_space_id = "scripted_constraints"
    model_family = "test"

    def __init__(self) -> None:
        self.calls = {"sample": 0, "mutation": 0, "crossover": 0}

    def _next(self, operation: str) -> Architecture:
        self.calls[operation] += 1
        return self.canonicalize(
            {
                "operation": operation,
                "serial": self.calls[operation],
                "allowed": self.calls[operation] % 2 == 0,
            }
        )

    def sample(self, seed: int | None = None) -> Architecture:
        return self._next("sample")

    def mutate(
        self, architecture: Architecture, seed: int | None = None
    ) -> Architecture:
        return self._next("mutation")

    def crossover(
        self,
        left: Architecture,
        right: Architecture,
        seed: int | None = None,
    ) -> Architecture:
        return self._next("crossover")

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        spec = dict(specification)
        identifier = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return Architecture(self.search_space_id, identifier, spec)

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        raise NotImplementedError


class SeededConstraintSpace(SearchSpace):
    search_space_id = "seeded_constraints"
    model_family = "test"

    def _architecture(self, operation: str, value: int) -> Architecture:
        return self.canonicalize({"operation": operation, "value": value})

    def sample(self, seed: int | None = None) -> Architecture:
        return self._architecture("sample", int(seed or 0))

    def mutate(
        self, architecture: Architecture, seed: int | None = None
    ) -> Architecture:
        return self._architecture("mutation", int(seed or 0))

    def crossover(
        self,
        left: Architecture,
        right: Architecture,
        seed: int | None = None,
    ) -> Architecture:
        return self._architecture("crossover", int(seed or 0))

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        spec = dict(specification)
        identifier = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return Architecture(self.search_space_id, identifier, spec)

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        raise NotImplementedError


def _resource_constraint(architecture: Architecture) -> dict[str, Any] | None:
    if "allowed" in architecture.spec:
        if not architecture.spec["allowed"]:
            return None
        value = int(architecture.spec["serial"])
    else:
        value = int(architecture.spec["value"])
        if value % 3 == 0:
            return None
    return {"parameters": value + 1, "resource_protocol": "test-v1"}


@pytest.mark.parametrize(
    ("population_size", "elite_ratio", "branch_random", "operation"),
    [(2, 0.5, 1.0, "mutation"), (4, 0.5, 0.0, "crossover")],
)
def test_constraints_retry_every_generation_path_before_evaluation(
    tmp_path, population_size, elite_ratio, branch_random, operation
):
    space = ScriptedConstraintSpace()
    evaluated: list[Architecture] = []

    def evaluator(architecture: Architecture) -> float:
        assert architecture.spec["allowed"] is True
        evaluated.append(architecture)
        return float(architecture.spec["serial"])

    output = tmp_path / f"{operation}.jsonl"
    search = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(output, 1),
        population_size=population_size,
        elite_ratio=elite_ratio,
        candidate_constraint=_resource_constraint,
        max_constraint_attempts=2,
    )
    search.rng.random = lambda: branch_random
    search.run(1)

    rows = list(read_jsonl(output))
    candidates = [row for row in rows if row["record_kind"] == "candidate"]
    summaries = [row for row in rows if row["record_kind"] == "generation_summary"]
    generated_count = population_size + (population_size - search.elite_count)
    assert len(evaluated) == generated_count
    assert space.calls["sample"] == population_size * 2
    assert space.calls[operation] == (population_size - search.elite_count) * 2
    assert len(candidates) == generated_count
    assert all(row["selected"] is True for row in candidates)
    assert all(row["parameters"] > 0 for row in candidates)
    assert all(row["resource_protocol"] == "test-v1" for row in candidates)
    assert search.constraint_attempts == generated_count * 2
    assert search.constraint_rejections == generated_count
    assert summaries[-1]["cumulative_constraint_attempts"] == generated_count * 2
    assert summaries[-1]["cumulative_constraint_rejections"] == generated_count


def test_constraint_attempt_limit_fails_closed_without_proxy_evaluation(tmp_path):
    evaluator_calls = 0

    def evaluator(architecture: Architecture) -> float:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return 1.0

    search = EvolutionSearch(
        ScriptedConstraintSpace(),
        evaluator,
        JsonlWriter(tmp_path / "rejected.jsonl", 1),
        population_size=2,
        candidate_constraint=lambda architecture: None,
        max_constraint_attempts=3,
    )
    with pytest.raises(RuntimeError, match=r"maximum number.*\(3\)"):
        search.run(0)

    assert evaluator_calls == 0
    assert search.constraint_attempts == 3
    assert search.constraint_rejections == 3
    assert list(read_jsonl(tmp_path / "rejected.jsonl")) == []


def test_constraint_state_resume_matches_uninterrupted_search(tmp_path):
    identity = {"protocol": "constraint-resume-v1"}
    partial_state_path = tmp_path / "partial-state.json"
    evaluator_calls = 0

    def interrupted_evaluator(architecture: Architecture) -> float:
        nonlocal evaluator_calls
        evaluator_calls += 1
        if evaluator_calls == 4:
            raise RuntimeError("simulated interruption")
        return float(architecture.spec["value"])

    with pytest.raises(RuntimeError, match="simulated interruption"):
        EvolutionSearch(
            SeededConstraintSpace(),
            interrupted_evaluator,
            JsonlWriter(tmp_path / "partial.jsonl", 1),
            population_size=6,
            seed=23,
            state_path=partial_state_path,
            state_identity=identity,
            initial_checkpoint_interval=1,
            candidate_constraint=_resource_constraint,
        ).run(2)

    partial_state = load_search_state(partial_state_path)
    assert partial_state["constraint_attempts"] > len(partial_state["population"])
    assert partial_state["constraint_rejections"] > 0
    resumed_output = tmp_path / "resumed.jsonl"
    resumed = EvolutionSearch(
        SeededConstraintSpace(),
        lambda architecture: float(architecture.spec["value"]),
        JsonlWriter(resumed_output, 1),
        population_size=6,
        seed=23,
        state_path=tmp_path / "resumed-state.json",
        resume_state=partial_state,
        state_identity=identity,
        initial_checkpoint_interval=1,
        candidate_constraint=_resource_constraint,
    ).run(2)
    uninterrupted_output = tmp_path / "uninterrupted.jsonl"
    uninterrupted_search = EvolutionSearch(
        SeededConstraintSpace(),
        lambda architecture: float(architecture.spec["value"]),
        JsonlWriter(uninterrupted_output, 1),
        population_size=6,
        seed=23,
        candidate_constraint=_resource_constraint,
    )
    uninterrupted = uninterrupted_search.run(2)

    def scientific_trace(path):
        return [
            {
                key: value
                for key, value in row.items()
                if key != "elapsed_seconds"
            }
            for row in read_jsonl(path)
        ]

    assert resumed.architecture.architecture_id == uninterrupted.architecture.architecture_id
    assert scientific_trace(resumed_output) == scientific_trace(uninterrupted_output)
    resumed_state = load_search_state(tmp_path / "resumed-state.json")
    assert resumed_state["constraint_attempts"] == uninterrupted_search.constraint_attempts
    assert resumed_state["constraint_rejections"] == uninterrupted_search.constraint_rejections
    assert all(
        candidate["constraint_metadata"]["resource_protocol"] == "test-v1"
        for candidate in resumed_state["population"]
    )
