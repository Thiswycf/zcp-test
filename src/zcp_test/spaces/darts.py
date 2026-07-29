from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zcp_test.models.darts import Genotype, NetworkCIFAR, NetworkImageNet
from zcp_test.spaces.base import SearchSpace
from zcp_test.types import Architecture

DARTS_PRIMITIVES = (
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
)
DARTS_STEPS = 4
DARTS_CONCAT = (2, 3, 4, 5)


@dataclass(frozen=True)
class DartsProfile:
    name: str
    dataset: str
    input_size: int
    init_channels: int
    layers: int
    auxiliary: bool
    drop_path_prob: float
    network: str


DARTS_PROFILES: dict[str, DartsProfile] = {
    "zcp": DartsProfile("zcp", "cifar10", 32, 16, 8, False, 0.0, "cifar"),
    "cifar10": DartsProfile("cifar10", "cifar10", 32, 36, 20, True, 0.2, "cifar"),
    "cifar100": DartsProfile("cifar100", "cifar100", 32, 36, 20, True, 0.2, "cifar"),
    "imagenet": DartsProfile("imagenet", "imagenet1k", 224, 48, 14, True, 0.0, "imagenet"),
}


def get_darts_profile(name: str) -> DartsProfile:
    normalized = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "zcp": "zcp",
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "imagenet": "imagenet",
        "imagenet1k": "imagenet",
    }
    try:
        return DARTS_PROFILES[aliases[normalized]]
    except KeyError as error:
        raise ValueError(f"Unknown DARTS profile: {name}") from error


def genotype_to_spec(genotype: Genotype) -> dict[str, Any]:
    return {
        "normal": [[operation, index] for operation, index in genotype.normal],
        "normal_concat": list(genotype.normal_concat),
        "reduce": [[operation, index] for operation, index in genotype.reduce],
        "reduce_concat": list(genotype.reduce_concat),
    }


def genotype_from_spec(specification: Mapping[str, Any]) -> Genotype:
    canonical = canonicalize_genotype(specification)
    return Genotype(
        normal=[tuple(edge) for edge in canonical["normal"]],
        normal_concat=list(canonical["normal_concat"]),
        reduce=[tuple(edge) for edge in canonical["reduce"]],
        reduce_concat=list(canonical["reduce_concat"]),
    )


def _canonical_edges(value: Any, cell: str) -> list[list[Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"DARTS {cell} must be a sequence of edges")
    if len(value) != 2 * DARTS_STEPS:
        raise ValueError(f"DARTS {cell} must contain {2 * DARTS_STEPS} edges")
    canonical: list[list[Any]] = []
    for node in range(DARTS_STEPS):
        node_edges: list[list[Any]] = []
        for edge in value[2 * node : 2 * node + 2]:
            if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)) or len(edge) != 2:
                raise ValueError(f"Each DARTS {cell} edge must be [operation, input_index]")
            operation = str(edge[0])
            index = edge[1]
            if operation not in DARTS_PRIMITIVES:
                raise ValueError(f"Invalid DARTS operation: {operation}")
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("DARTS input indices must be integers")
            if not 0 <= index < node + 2:
                raise ValueError(
                    f"Invalid DARTS {cell} input {index} for intermediate node {node}"
                )
            node_edges.append([operation, index])
        if node_edges[0][1] == node_edges[1][1]:
            raise ValueError(f"DARTS {cell} node {node} must use two distinct inputs")
        canonical.extend(sorted(node_edges, key=lambda edge: (edge[1], edge[0])))
    return canonical


def _canonical_concat(value: Any, cell: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"DARTS {cell}_concat must be a sequence")
    concat = list(value)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in concat):
        raise TypeError("DARTS concat indices must be integers")
    if tuple(concat) != DARTS_CONCAT:
        raise ValueError(f"DARTS {cell}_concat must be {list(DARTS_CONCAT)}")
    return concat


def canonicalize_genotype(specification: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(specification, Mapping):
        raise TypeError("DARTS genotype must be a mapping")
    required = {"normal", "normal_concat", "reduce", "reduce_concat"}
    missing = required - set(specification)
    if missing:
        raise ValueError(f"DARTS genotype is missing: {sorted(missing)}")
    unknown = set(specification) - required
    if unknown:
        raise ValueError(f"DARTS genotype has unknown fields: {sorted(unknown)}")
    return {
        "normal": _canonical_edges(specification["normal"], "normal"),
        "normal_concat": _canonical_concat(specification["normal_concat"], "normal"),
        "reduce": _canonical_edges(specification["reduce"], "reduce"),
        "reduce_concat": _canonical_concat(specification["reduce_concat"], "reduce"),
    }


def _architecture_id(specification: Mapping[str, Any]) -> str:
    payload = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"darts:{payload}".encode()).hexdigest()[:20]


def _sample_cell(rng: random.Random) -> list[list[Any]]:
    edges: list[list[Any]] = []
    for node in range(DARTS_STEPS):
        for index in sorted(rng.sample(range(node + 2), 2)):
            edges.append([rng.choice(DARTS_PRIMITIVES), index])
    return edges


class DartsSpace(SearchSpace):
    search_space_id = "darts"
    model_family = "cnn"
    model_fidelity = "reference"

    def __init__(self, profile: str = "auto") -> None:
        if profile != "auto":
            get_darts_profile(profile)
        self.profile = profile

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        canonical = canonicalize_genotype(specification)
        return Architecture(self.search_space_id, _architecture_id(canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        return self.canonicalize(
            {
                "normal": _sample_cell(rng),
                "normal_concat": list(DARTS_CONCAT),
                "reduce": _sample_cell(rng),
                "reduce_concat": list(DARTS_CONCAT),
            }
        )

    def _validate_architecture(self, architecture: Architecture) -> None:
        if architecture.search_space_id != self.search_space_id:
            raise ValueError(
                f"Expected a {self.search_space_id} architecture, got "
                f"{architecture.search_space_id}"
            )

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        self._validate_architecture(architecture)
        rng = random.Random(seed)
        specification = canonicalize_genotype(architecture.spec)
        cell = rng.choice(["normal", "reduce"])
        edges = [list(edge) for edge in specification[cell]]
        actions: list[tuple[str, int, tuple[Any, ...]]] = []
        for edge_index, (operation, source) in enumerate(edges):
            operation_choices = tuple(choice for choice in DARTS_PRIMITIVES if choice != operation)
            actions.append(("operation", edge_index, operation_choices))
            node = edge_index // 2
            other_source = edges[edge_index - 1 if edge_index % 2 else edge_index + 1][1]
            source_choices = tuple(
                choice
                for choice in range(node + 2)
                if choice not in (source, other_source)
            )
            if source_choices:
                actions.append(("source", edge_index, source_choices))
        kind, edge_index, choices = rng.choice(actions)
        edges[edge_index][0 if kind == "operation" else 1] = rng.choice(choices)
        specification[cell] = edges
        return self.canonicalize(specification)

    def crossover(
        self,
        left: Architecture,
        right: Architecture,
        seed: int | None = None,
    ) -> Architecture:
        self._validate_architecture(left)
        self._validate_architecture(right)
        rng = random.Random(seed)
        left_spec = canonicalize_genotype(left.spec)
        right_spec = canonicalize_genotype(right.spec)
        child: dict[str, Any] = {
            "normal_concat": list(DARTS_CONCAT),
            "reduce_concat": list(DARTS_CONCAT),
        }
        for cell in ("normal", "reduce"):
            edges: list[list[Any]] = []
            for node in range(DARTS_STEPS):
                left_node = left_spec[cell][2 * node : 2 * node + 2]
                right_node = right_spec[cell][2 * node : 2 * node + 2]
                sources = list(rng.choice([left_node, right_node]))
                for edge_index, source_edge in enumerate(sources):
                    parent_edge = rng.choice(
                        [left_node[edge_index], right_node[edge_index], source_edge]
                    )
                    edges.append([parent_edge[0], source_edge[1]])
            child[cell] = edges
        return self.canonicalize(child)

    def resolve_profile(self, num_classes: int, profile: str | None = None) -> DartsProfile:
        selected = profile or self.profile
        if selected == "auto":
            if num_classes >= 1000:
                selected = "imagenet"
            elif num_classes == 100:
                selected = "cifar100"
            else:
                selected = "cifar10"
        return get_darts_profile(selected)

    def build_model(
        self,
        architecture: Architecture,
        num_classes: int,
        profile: str | None = None,
    ) -> NetworkCIFAR | NetworkImageNet:
        self._validate_architecture(architecture)
        selected = self.resolve_profile(num_classes, profile)
        genotype = genotype_from_spec(architecture.spec)
        network_class = NetworkImageNet if selected.network == "imagenet" else NetworkCIFAR
        return network_class(
            selected.init_channels,
            num_classes,
            selected.layers,
            selected.auxiliary,
            genotype,
            selected.drop_path_prob,
        )


__all__ = [
    "DARTS_CONCAT",
    "DARTS_PRIMITIVES",
    "DARTS_PROFILES",
    "DARTS_STEPS",
    "DartsProfile",
    "DartsSpace",
    "canonicalize_genotype",
    "genotype_from_spec",
    "genotype_to_spec",
    "get_darts_profile",
]
