from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from zcp_test.artifacts import JsonlWriter
from zcp_test.spaces.base import SearchSpace
from zcp_test.types import Architecture


@dataclass
class Candidate:
    architecture: Architecture
    score: float
    parents: tuple[str, ...] = ()
    operation: str = "sample"


def cache_key(architecture: Architecture, proxy_id: str, dataset: str, seed: int, input_fingerprint: str, proxy_version: str = "1") -> str:
    payload = [architecture.search_space_id, architecture.architecture_id, proxy_id, proxy_version, dataset, seed, input_fingerprint]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


class EvolutionSearch:
    def __init__(self, space: SearchSpace, evaluator: Callable[[Architecture], float], writer: JsonlWriter, population_size: int = 20, elite_ratio: float = 0.2, seed: int = 42, record_metadata: Mapping[str, Any] | None = None) -> None:
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
        self.cache_hits = 0
        self.evaluations = 0
        self.record_metadata = dict(record_metadata or {})

    def _score(self, architecture: Architecture) -> tuple[float, bool]:
        if architecture.architecture_id in self.cache:
            self.cache_hits += 1
            return self.cache[architecture.architecture_id], True
        score = float(self.evaluator(architecture))
        self.evaluations += 1
        self.cache[architecture.architecture_id] = score
        return score, False

    def run(self, generations: int) -> Candidate:
        population: list[Candidate] = []
        for _ in range(self.population_size):
            architecture = self.space.sample(self.rng.randrange(2**31))
            score, hit = self._score(architecture)
            population.append(Candidate(architecture, score))
            self._record(0, population[-1], hit, selected=True)
        self._summary(0, population)
        for generation in range(1, generations + 1):
            population.sort(key=lambda candidate: candidate.score, reverse=True)
            elites = population[: self.elite_count]
            next_population = list(elites)
            while len(next_population) < self.population_size:
                if len(elites) > 1 and self.rng.random() < 0.35:
                    left, right = self.rng.sample(elites, 2)
                    architecture = self.space.crossover(left.architecture, right.architecture, self.rng.randrange(2**31))
                    parents, operation = (left.architecture.architecture_id, right.architecture.architecture_id), "crossover"
                else:
                    parent = self.rng.choice(elites)
                    architecture = self.space.mutate(parent.architecture, self.rng.randrange(2**31))
                    parents, operation = (parent.architecture.architecture_id,), "mutation"
                score, hit = self._score(architecture)
                candidate = Candidate(architecture, score, parents, operation)
                next_population.append(candidate)
                self._record(generation, candidate, hit, selected=True)
            population = next_population
            self._summary(generation, population)
        return max(population, key=lambda candidate: candidate.score)

    def _record(self, generation: int, candidate: Candidate, cache_hit: bool, selected: bool) -> None:
        self.writer.append({**self.record_metadata, "record_kind": "candidate", "generation": generation, "search_space_id": candidate.architecture.search_space_id, "architecture_id": candidate.architecture.architecture_id, "architecture": candidate.architecture.spec, "parents": candidate.parents, "operation": candidate.operation, "score": candidate.score, "cache_hit": cache_hit, "selected": selected, "cumulative_evaluations": self.evaluations, "cumulative_cache_hits": self.cache_hits, "elapsed_seconds": time.perf_counter() - self.started})

    def _summary(self, generation: int, population: list[Candidate]) -> None:
        scores = sorted(candidate.score for candidate in population)

        def percentile(fraction: float) -> float:
            return scores[round((len(scores) - 1) * fraction)]

        self.writer.append({**self.record_metadata, "record_kind": "generation_summary", "generation": generation, "best_so_far": max(self.cache.values()), "mean_score": sum(scores) / len(scores), "q25": percentile(0.25), "q50": percentile(0.5), "q75": percentile(0.75), "diversity": len({candidate.architecture.architecture_id for candidate in population}) / len(population), "cumulative_evaluations": self.evaluations, "cumulative_cache_hits": self.cache_hits, "elapsed_seconds": time.perf_counter() - self.started})
