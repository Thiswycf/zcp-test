from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from zcp_test.artifacts import JsonlWriter, read_jsonl
from zcp_test.spaces.base import SearchSpace
from zcp_test.types import Architecture


@dataclass
class Candidate:
    architecture: Architecture
    score: float
    parents: tuple[str, ...] = ()
    operation: str = "sample"
    evaluation_metadata: dict[str, Any] | None = None
    components: dict[str, float] | None = None


def cache_key(
    architecture: Architecture,
    proxy_id: str,
    dataset: str,
    seed: int,
    input_fingerprint: str,
    proxy_version: str = "1",
) -> str:
    payload = [
        architecture.search_space_id,
        architecture.architecture_id,
        proxy_id,
        proxy_version,
        dataset,
        seed,
        input_fingerprint,
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def load_search_state(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported or invalid search state schema")
    return payload


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def validate_search_state_identity(
    state: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    state_identity = state.get("identity")
    if not isinstance(state_identity, Mapping) or dict(state_identity) != dict(identity):
        raise ValueError("Search state identity does not match the requested search protocol")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
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


class EvolutionSearch:
    def __init__(
        self,
        space: SearchSpace,
        evaluator: Callable[[Architecture], float | Mapping[str, float]],
        writer: JsonlWriter,
        population_size: int = 20,
        elite_ratio: float = 0.2,
        seed: int = 42,
        record_metadata: Mapping[str, Any] | None = None,
        evaluation_identity: Callable[[Architecture], tuple[str, Mapping[str, Any]]] | None = None,
        component_aggregator: Callable[[Sequence[Mapping[str, float]]], Sequence[float]]
        | None = None,
        *,
        state_path: str | Path | None = None,
        resume_state: Mapping[str, Any] | None = None,
        state_identity: Mapping[str, Any] | None = None,
        initial_checkpoint_interval: int = 100,
    ) -> None:
        if population_size < 2 or not 0 < elite_ratio <= 1:
            raise ValueError("Invalid population settings")
        self.space = space
        self.evaluator = evaluator
        self.writer = writer
        self.population_size = population_size
        self.elite_count = max(1, round(population_size * elite_ratio))
        self.rng = random.Random(seed)
        self.cache: dict[str, float | dict[str, float]] = {}
        self.started = time.perf_counter()
        self.elapsed_offset = 0.0
        self.cache_hits = 0
        self.evaluations = 0
        self.record_metadata = dict(record_metadata or {})
        self.evaluation_identity = evaluation_identity
        self.component_aggregator = component_aggregator
        self.state_path = None if state_path is None else Path(state_path)
        self.state_identity = dict(state_identity or {})
        if initial_checkpoint_interval <= 0:
            raise ValueError("Initial checkpoint interval must be positive")
        self.initial_checkpoint_interval = initial_checkpoint_interval
        self._restored_population: list[Candidate] | None = None
        self._partial_population: list[Candidate] | None = None
        self._partial_cache_hits: list[bool] = []
        self._completed_generation = -1
        if resume_state is not None:
            self._restore(resume_state)

    def _elapsed(self) -> float:
        return self.elapsed_offset + time.perf_counter() - self.started

    def _candidate_from_state(self, payload: Mapping[str, Any]) -> Candidate:
        architecture_payload = payload.get("architecture")
        if not isinstance(architecture_payload, Mapping):
            raise ValueError("Search state candidate architecture must be an object")
        specification = architecture_payload.get("spec")
        if not isinstance(specification, Mapping):
            raise ValueError("Search state candidate spec must be an object")
        architecture = self.space.canonicalize(specification)
        if architecture.search_space_id != architecture_payload.get("search_space_id"):
            raise ValueError("Search state search-space identity does not match")
        if architecture.architecture_id != architecture_payload.get("architecture_id"):
            raise ValueError("Search state architecture ID does not match its canonical spec")
        return Candidate(
            architecture,
            float(payload["score"]),
            tuple(str(value) for value in payload.get("parents", ())),
            str(payload.get("operation", "sample")),
            dict(payload.get("evaluation_metadata") or {}),
            (
                None
                if payload.get("components") is None
                else {
                    str(key): float(value)
                    for key, value in payload["components"].items()
                }
            ),
        )

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "1.0":
            raise ValueError("Unsupported or invalid search state schema")
        validate_search_state_identity(state, self.state_identity)
        if int(state.get("population_size", -1)) != self.population_size:
            raise ValueError("Search state population size does not match")
        if int(state.get("elite_count", -1)) != self.elite_count:
            raise ValueError("Search state elite count does not match")
        population = state.get("population")
        if not isinstance(population, list):
            raise ValueError("Search state population must be a list")
        if self.writer.path.exists() and self.writer.path.stat().st_size != 0:
            raise ValueError("Search resume writer must start with an empty JSONL file")
        history = state.get("history")
        if not isinstance(history, list) or not all(isinstance(row, dict) for row in history):
            raise ValueError("Search state history must be a list of JSON objects")
        completed_generation = int(state["completed_generation"])
        partial_initial_population = completed_generation == -1
        if completed_generation < -1:
            raise ValueError("Search state completed generation is invalid")
        if partial_initial_population:
            if not population or len(population) > self.population_size:
                raise ValueError("Partial search state population is invalid")
        elif len(population) != self.population_size:
            raise ValueError("Search state population is incomplete")
        summary_generations = [
            row.get("generation")
            for row in history
            if row.get("record_kind") == "generation_summary"
        ]
        if summary_generations != list(range(completed_generation + 1)):
            raise ValueError("Search state history has incomplete or duplicate generation summaries")
        restored_population = [self._candidate_from_state(item) for item in population]
        cache_payload = state.get("cache")
        if not isinstance(cache_payload, Mapping):
            raise ValueError("Search state cache must be an object")
        restored_cache: dict[str, float | dict[str, float]] = {}
        for key, value in cache_payload.items():
            restored_cache[str(key)] = (
                {str(name): float(component) for name, component in value.items()}
                if isinstance(value, Mapping)
                else float(value)
            )
        for candidate in restored_population:
            identity, metadata = self._resolve_evaluation_identity(candidate.architecture)
            if identity not in restored_cache:
                raise ValueError("Search state population does not match its score cache")
            cached = restored_cache[identity]
            if self.component_aggregator is None and cached != candidate.score:
                raise ValueError("Search state population does not match its score cache")
            if self.component_aggregator is not None and cached != candidate.components:
                raise ValueError("Search state population components do not match its cache")
            if candidate.evaluation_metadata and candidate.evaluation_metadata != metadata:
                raise ValueError("Search state candidate input metadata does not match")
            candidate.evaluation_metadata = metadata
        cache_hits = int(state["cache_hits"])
        evaluations = int(state["evaluations"])
        elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        if cache_hits < 0 or evaluations < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Search state counters are invalid")
        if partial_initial_population:
            if history:
                raise ValueError("Partial initial search state must not contain final history")
            initial_cache_hits = state.get("initial_cache_hits")
            if (
                not isinstance(initial_cache_hits, list)
                or len(initial_cache_hits) != len(restored_population)
                or not all(isinstance(value, bool) for value in initial_cache_hits)
            ):
                raise ValueError("Partial search state cache-hit flags are invalid")
        else:
            if not history or history[-1].get("record_kind") != "generation_summary":
                raise ValueError("Search state history must end with a generation summary")
            if (
                history[-1].get("cumulative_cache_hits") != cache_hits
                or history[-1].get("cumulative_evaluations") != evaluations
            ):
                raise ValueError("Search state history counters do not match")
        restored_rng = random.Random()
        try:
            restored_rng.setstate(_tuple_tree(state["rng_state"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Search state RNG state is invalid") from error
        for row in history:
            self.writer.append(row)
        if partial_initial_population:
            self._partial_population = restored_population
            self._partial_cache_hits = list(initial_cache_hits)
        else:
            self._restored_population = restored_population
        self._completed_generation = completed_generation
        self.cache = restored_cache
        self.cache_hits = cache_hits
        self.evaluations = evaluations
        self.elapsed_offset = elapsed_seconds
        self.started = time.perf_counter()
        self.rng.setstate(restored_rng.getstate())

    def _candidate_state(self, candidate: Candidate) -> dict[str, Any]:
        return {
            "architecture": candidate.architecture.to_dict(),
            "score": candidate.score,
            "parents": list(candidate.parents),
            "operation": candidate.operation,
            "evaluation_metadata": candidate.evaluation_metadata or {},
            "components": candidate.components,
        }

    def _save_state(self, generation: int, population: list[Candidate]) -> None:
        if self.state_path is None:
            return
        payload = {
            "schema_version": "1.0",
            "identity": self.state_identity,
            "completed_generation": generation,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "population": [self._candidate_state(candidate) for candidate in population],
            "cache": self.cache,
            "cache_hits": self.cache_hits,
            "evaluations": self.evaluations,
            "elapsed_seconds": self._elapsed(),
            "rng_state": self.rng.getstate(),
            "history": list(read_jsonl(self.writer.path)),
        }
        _atomic_write_json(self.state_path, payload)

    def _save_partial_initial_state(
        self, population: list[Candidate], cache_hits: list[bool]
    ) -> None:
        if self.state_path is None:
            return
        payload = {
            "schema_version": "1.0",
            "identity": self.state_identity,
            "completed_generation": -1,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "population": [self._candidate_state(candidate) for candidate in population],
            "initial_cache_hits": cache_hits,
            "cache": self.cache,
            "cache_hits": self.cache_hits,
            "evaluations": self.evaluations,
            "elapsed_seconds": self._elapsed(),
            "rng_state": self.rng.getstate(),
            "history": [],
        }
        _atomic_write_json(self.state_path, payload)

    def _resolve_evaluation_identity(
        self, architecture: Architecture
    ) -> tuple[str, dict[str, Any]]:
        if self.evaluation_identity is None:
            return architecture.architecture_id, {}
        identity, metadata = self.evaluation_identity(architecture)
        return str(identity), dict(metadata)

    def _score(
        self, architecture: Architecture
    ) -> tuple[float, dict[str, float] | None, bool, dict[str, Any]]:
        identity, metadata = self._resolve_evaluation_identity(architecture)
        if identity in self.cache:
            self.cache_hits += 1
            cached = self.cache[identity]
            if isinstance(cached, dict):
                return 0.0, dict(cached), True, metadata
            return cached, None, True, metadata
        value = self.evaluator(architecture)
        self.evaluations += 1
        if isinstance(value, Mapping):
            if self.component_aggregator is None:
                raise ValueError("Component-valued evaluator requires a component aggregator")
            components = {str(name): float(component) for name, component in value.items()}
            if not components or not all(
                math.isfinite(component) for component in components.values()
            ):
                raise ValueError("Search evaluator returned invalid components")
            self.cache[identity] = components
            return 0.0, components, False, metadata
        if self.component_aggregator is not None:
            raise ValueError("Component aggregator requires a component-valued evaluator")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("Search evaluator returned a non-finite score")
        self.cache[identity] = score
        return score, None, False, metadata

    def _aggregate_components(self, population: Sequence[Candidate]) -> None:
        if self.component_aggregator is None:
            return
        cached_items = [
            (identity, value)
            for identity, value in sorted(self.cache.items())
            if isinstance(value, dict)
        ]
        component_rows = [value for _identity, value in cached_items]
        aggregate_scores = list(self.component_aggregator(component_rows))
        if len(aggregate_scores) != len(component_rows) or not all(
            math.isfinite(float(score)) for score in aggregate_scores
        ):
            raise ValueError("Component aggregator returned invalid scores")
        score_by_identity = {
            identity: float(score)
            for (identity, _components), score in zip(
                cached_items, aggregate_scores, strict=True
            )
        }
        for candidate in population:
            identity, _metadata = self._resolve_evaluation_identity(candidate.architecture)
            if candidate.components is None or identity not in score_by_identity:
                raise ValueError("Aggregated candidate is missing cached components")
            candidate.score = score_by_identity[identity]

    def run(self, generations: int) -> Candidate:
        if generations < 0:
            raise ValueError("generations must be non-negative")
        if self._restored_population is None:
            population = list(self._partial_population or [])
            initial_records = list(zip(population, self._partial_cache_hits, strict=True))
            while len(population) < self.population_size:
                architecture = self.space.sample(self.rng.randrange(2**31))
                score, components, hit, metadata = self._score(architecture)
                population.append(
                    Candidate(
                        architecture,
                        score,
                        evaluation_metadata=metadata,
                        components=components,
                    )
                )
                initial_records.append((population[-1], hit))
                if len(population) % self.initial_checkpoint_interval == 0:
                    self._save_partial_initial_state(
                        population,
                        [cache_hit for _candidate, cache_hit in initial_records],
                    )
            self._aggregate_components(population)
            for candidate, hit in initial_records:
                self._record(0, candidate, hit, selected=True)
            self._summary(0, population)
            self._save_state(0, population)
            first_generation = 1
        else:
            population = self._restored_population
            if generations < self._completed_generation:
                raise ValueError("Requested generations precede the completed search state")
            first_generation = self._completed_generation + 1
        for generation in range(first_generation, generations + 1):
            population.sort(key=lambda candidate: candidate.score, reverse=True)
            elites = population[: self.elite_count]
            next_population = list(elites)
            new_records: list[tuple[Candidate, bool]] = []
            while len(next_population) < self.population_size:
                if len(elites) > 1 and self.rng.random() < 0.35:
                    left, right = self.rng.sample(elites, 2)
                    architecture = self.space.crossover(
                        left.architecture, right.architecture, self.rng.randrange(2**31)
                    )
                    parents, operation = (
                        (left.architecture.architecture_id, right.architecture.architecture_id),
                        "crossover",
                    )
                else:
                    parent = self.rng.choice(elites)
                    architecture = self.space.mutate(parent.architecture, self.rng.randrange(2**31))
                    parents, operation = (parent.architecture.architecture_id,), "mutation"
                score, components, hit, metadata = self._score(architecture)
                candidate = Candidate(
                    architecture,
                    score,
                    parents,
                    operation,
                    metadata,
                    components,
                )
                next_population.append(candidate)
                new_records.append((candidate, hit))
            population = next_population
            self._aggregate_components(population)
            for candidate, hit in new_records:
                self._record(generation, candidate, hit, selected=True)
            self._summary(generation, population)
            self._save_state(generation, population)
        return max(population, key=lambda candidate: candidate.score)

    def _record(
        self, generation: int, candidate: Candidate, cache_hit: bool, selected: bool
    ) -> None:
        self.writer.append(
            {
                **self.record_metadata,
                **(candidate.evaluation_metadata or {}),
                "record_kind": "candidate",
                "generation": generation,
                "search_space_id": candidate.architecture.search_space_id,
                "architecture_id": candidate.architecture.architecture_id,
                "architecture": candidate.architecture.spec,
                "parents": candidate.parents,
                "operation": candidate.operation,
                "score": candidate.score,
                "components": candidate.components,
                "cache_hit": cache_hit,
                "selected": selected,
                "cumulative_evaluations": self.evaluations,
                "cumulative_cache_hits": self.cache_hits,
                "elapsed_seconds": self._elapsed(),
            }
        )

    def _summary(self, generation: int, population: list[Candidate]) -> None:
        scores = sorted(candidate.score for candidate in population)

        def percentile(fraction: float) -> float:
            return scores[round((len(scores) - 1) * fraction)]

        self.writer.append(
            {
                **self.record_metadata,
                "record_kind": "generation_summary",
                "generation": generation,
                "best_so_far": (
                    max(self.cache.values())
                    if self.component_aggregator is None
                    else max(scores)
                ),
                "mean_score": sum(scores) / len(scores),
                "q25": percentile(0.25),
                "q50": percentile(0.5),
                "q75": percentile(0.75),
                "diversity": len(
                    {candidate.architecture.architecture_id for candidate in population}
                )
                / len(population),
                "cumulative_evaluations": self.evaluations,
                "cumulative_cache_hits": self.cache_hits,
                "elapsed_seconds": self._elapsed(),
            }
        )
