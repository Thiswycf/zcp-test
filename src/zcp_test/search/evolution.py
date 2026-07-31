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
from typing import Any, Callable, Mapping

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
        evaluator: Callable[[Architecture], float],
        writer: JsonlWriter,
        population_size: int = 20,
        elite_ratio: float = 0.2,
        seed: int = 42,
        record_metadata: Mapping[str, Any] | None = None,
        evaluation_identity: Callable[[Architecture], tuple[str, Mapping[str, Any]]] | None = None,
        *,
        state_path: str | Path | None = None,
        resume_state: Mapping[str, Any] | None = None,
        state_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if population_size < 2 or not 0 < elite_ratio <= 1:
            raise ValueError("Invalid population settings")
        self.space = space
        self.evaluator = evaluator
        self.writer = writer
        self.population_size = population_size
        self.elite_count = max(1, round(population_size * elite_ratio))
        self.rng = random.Random(seed)
        self.cache: dict[str, float] = {}
        self.started = time.perf_counter()
        self.elapsed_offset = 0.0
        self.cache_hits = 0
        self.evaluations = 0
        self.record_metadata = dict(record_metadata or {})
        self.evaluation_identity = evaluation_identity
        self.state_path = None if state_path is None else Path(state_path)
        self.state_identity = dict(state_identity or {})
        self._restored_population: list[Candidate] | None = None
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
        if not isinstance(population, list) or len(population) != self.population_size:
            raise ValueError("Search state population is incomplete")
        if self.writer.path.exists() and self.writer.path.stat().st_size != 0:
            raise ValueError("Search resume writer must start with an empty JSONL file")
        history = state.get("history")
        if not isinstance(history, list) or not all(isinstance(row, dict) for row in history):
            raise ValueError("Search state history must be a list of JSON objects")
        completed_generation = int(state["completed_generation"])
        if completed_generation < 0:
            raise ValueError("Search state completed generation must be non-negative")
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
        restored_cache = {str(key): float(value) for key, value in cache_payload.items()}
        for candidate in restored_population:
            identity, metadata = self._resolve_evaluation_identity(candidate.architecture)
            if identity not in restored_cache or restored_cache[identity] != candidate.score:
                raise ValueError("Search state population does not match its score cache")
            if candidate.evaluation_metadata and candidate.evaluation_metadata != metadata:
                raise ValueError("Search state candidate input metadata does not match")
            candidate.evaluation_metadata = metadata
        cache_hits = int(state["cache_hits"])
        evaluations = int(state["evaluations"])
        elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        if cache_hits < 0 or evaluations < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Search state counters are invalid")
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

    def _resolve_evaluation_identity(
        self, architecture: Architecture
    ) -> tuple[str, dict[str, Any]]:
        if self.evaluation_identity is None:
            return architecture.architecture_id, {}
        identity, metadata = self.evaluation_identity(architecture)
        return str(identity), dict(metadata)

    def _score(self, architecture: Architecture) -> tuple[float, bool, dict[str, Any]]:
        identity, metadata = self._resolve_evaluation_identity(architecture)
        if identity in self.cache:
            self.cache_hits += 1
            return self.cache[identity], True, metadata
        score = float(self.evaluator(architecture))
        self.evaluations += 1
        self.cache[identity] = score
        return score, False, metadata

    def run(self, generations: int) -> Candidate:
        if generations < 0:
            raise ValueError("generations must be non-negative")
        if self._restored_population is None:
            population: list[Candidate] = []
            for _ in range(self.population_size):
                architecture = self.space.sample(self.rng.randrange(2**31))
                score, hit, metadata = self._score(architecture)
                population.append(
                    Candidate(architecture, score, evaluation_metadata=metadata)
                )
                self._record(0, population[-1], hit, selected=True)
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
                score, hit, metadata = self._score(architecture)
                candidate = Candidate(architecture, score, parents, operation, metadata)
                next_population.append(candidate)
                self._record(generation, candidate, hit, selected=True)
            population = next_population
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
                "best_so_far": max(self.cache.values()),
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
