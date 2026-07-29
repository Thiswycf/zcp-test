from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

import numpy as np

from zcp_test.spaces.base import SearchSpace
from zcp_test.types import Architecture


AVAILABLE_OPS = ("conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3")


def prune_spec(matrix: list[list[int]], operations: list[str]) -> tuple[list[list[int]], list[str]]:
    vertices = len(matrix)
    from_input = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for target in range(source + 1, vertices):
            if matrix[source][target] and target not in from_input:
                from_input.add(target)
                frontier.append(target)
    from_output = {vertices - 1}
    frontier = [vertices - 1]
    while frontier:
        target = frontier.pop()
        for source in range(target):
            if matrix[source][target] and source not in from_output:
                from_output.add(source)
                frontier.append(source)
    keep = sorted(from_input & from_output)
    if len(keep) < 2:
        raise ValueError("NAS-Bench-101 input is not connected to output")
    return [[matrix[source][target] for target in keep] for source in keep], [
        operations[index] for index in keep
    ]


def hash_module(matrix: list[list[int]], labels: list[int]) -> str:
    array = np.asarray(matrix, dtype=np.int8)
    in_edges = np.sum(array, axis=0).tolist()
    out_edges = np.sum(array, axis=1).tolist()
    hashes = [
        hashlib.md5(str(item).encode()).hexdigest()
        for item in zip(out_edges, in_edges, labels, strict=True)
    ]
    for _ in range(len(matrix)):
        updated = []
        for vertex in range(len(matrix)):
            incoming = [hashes[source] for source in range(len(matrix)) if array[source, vertex]]
            outgoing = [hashes[target] for target in range(len(matrix)) if array[vertex, target]]
            payload = "".join(sorted(incoming)) + "|" + "".join(sorted(outgoing)) + "|" + hashes[vertex]
            updated.append(hashlib.md5(payload.encode()).hexdigest())
        hashes = updated
    return hashlib.md5(str(sorted(hashes)).encode()).hexdigest()


class Nb101Space(SearchSpace):
    search_space_id = "nb101_dag"
    model_family = "cnn"
    model_fidelity = "reference_topology_pytorch_port"

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        if not isinstance(specification, Mapping):
            raise TypeError("NAS-Bench-101 specification must be a mapping")
        raw_matrix = specification.get("matrix")
        raw_operations = specification.get("operations")
        if not isinstance(raw_matrix, (list, tuple)) or not 2 <= len(raw_matrix) <= 7:
            raise ValueError("NAS-Bench-101 matrix must have 2 to 7 vertices")
        vertices = len(raw_matrix)
        if any(not isinstance(row, (list, tuple)) or len(row) != vertices for row in raw_matrix):
            raise ValueError("NAS-Bench-101 matrix must be square")
        matrix = [[int(value) for value in row] for row in raw_matrix]
        if any(value not in (0, 1) for row in matrix for value in row):
            raise ValueError("NAS-Bench-101 adjacency values must be 0 or 1")
        if any(matrix[source][target] for source in range(vertices) for target in range(source + 1)):
            raise ValueError("NAS-Bench-101 matrix must be strictly upper triangular")
        if sum(map(sum, matrix)) > 9:
            raise ValueError("NAS-Bench-101 supports at most 9 edges")
        if not isinstance(raw_operations, (list, tuple)) or len(raw_operations) != vertices:
            raise ValueError("NAS-Bench-101 operations must match matrix dimensions")
        operations = [str(operation) for operation in raw_operations]
        if operations[0] != "input" or operations[-1] != "output":
            raise ValueError("NAS-Bench-101 operations must start with input and end with output")
        if any(operation not in AVAILABLE_OPS for operation in operations[1:-1]):
            raise ValueError("NAS-Bench-101 contains an unsupported operation")
        matrix, operations = prune_spec(matrix, operations)
        labels = [-1] + [AVAILABLE_OPS.index(operation) for operation in operations[1:-1]] + [-2]
        specification_id = hash_module(matrix, labels)
        canonical = {"matrix": matrix, "operations": operations}
        return Architecture(self.search_space_id, specification_id, canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        for _ in range(10_000):
            vertices = rng.randint(2, 7)
            possible = [(source, target) for target in range(1, vertices) for source in range(target)]
            edge_count = rng.randint(1, min(9, len(possible)))
            selected = set(rng.sample(possible, edge_count))
            matrix = [[int((source, target) in selected) for target in range(vertices)] for source in range(vertices)]
            operations = ["input", *[rng.choice(AVAILABLE_OPS) for _ in range(vertices - 2)], "output"]
            try:
                return self.canonicalize({"matrix": matrix, "operations": operations})
            except ValueError:
                continue
        raise RuntimeError("Unable to sample a valid NAS-Bench-101 architecture")

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        for _ in range(1_000):
            matrix = [list(row) for row in architecture.spec["matrix"]]
            operations = list(architecture.spec["operations"])
            if len(operations) > 2 and rng.random() < 0.5:
                vertex = rng.randrange(1, len(operations) - 1)
                operations[vertex] = rng.choice([op for op in AVAILABLE_OPS if op != operations[vertex]])
            else:
                possible = [(source, target) for target in range(1, len(matrix)) for source in range(target)]
                source, target = rng.choice(possible)
                matrix[source][target] = 1 - matrix[source][target]
            try:
                candidate = self.canonicalize({"matrix": matrix, "operations": operations})
                if candidate.architecture_id != architecture.architecture_id:
                    return candidate
            except ValueError:
                continue
        raise RuntimeError("Unable to mutate NAS-Bench-101 architecture")

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        if len(left.spec["matrix"]) != len(right.spec["matrix"]):
            return self.mutate(rng.choice([left, right]), seed)
        for _ in range(1_000):
            matrix = [
                [rng.choice((left.spec["matrix"][source][target], right.spec["matrix"][source][target])) for target in range(len(left.spec["matrix"]))]
                for source in range(len(left.spec["matrix"]))
            ]
            operations = [
                rng.choice((left.spec["operations"][index], right.spec["operations"][index]))
                for index in range(len(left.spec["operations"]))
            ]
            try:
                return self.canonicalize({"matrix": matrix, "operations": operations})
            except ValueError:
                continue
        raise RuntimeError("Unable to crossover NAS-Bench-101 architectures")

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.nb101 import Network

        return Network(architecture.spec, num_classes)


__all__ = ["AVAILABLE_OPS", "Nb101Space", "hash_module", "prune_spec"]
