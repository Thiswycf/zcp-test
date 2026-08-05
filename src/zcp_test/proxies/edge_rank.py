from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EdgeActivation:
    source: Hashable
    target: Hashable
    activation: Any

    def __post_init__(self) -> None:
        _validate_endpoint(self.source, "source")
        _validate_endpoint(self.target, "target")
        _validate_activation(self.activation)


@dataclass(frozen=True, slots=True)
class EdgeActivationBatch:
    edges: tuple[EdgeActivation, ...]

    def __init__(self, edges: Iterable[EdgeActivation]) -> None:
        materialized = tuple(edges)
        if not all(isinstance(edge, EdgeActivation) for edge in materialized):
            raise TypeError("edges must contain only EdgeActivation values")
        object.__setattr__(self, "edges", materialized)

    def __iter__(self):
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)


def compute_er_score(activations: Iterable[Any]) -> float:
    """Sum first-party effective ranks over 4D edge activations."""
    materialized = tuple(activations)
    if not materialized:
        raise ValueError("ER requires at least one edge activation")
    return math.fsum(_effective_rank(activation) for activation in materialized)


def compute_ter_score(
    batch: EdgeActivationBatch | Iterable[EdgeActivation],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Weight edge effective ranks by bidirectional edge-centric PageRank."""
    edges = batch.edges if isinstance(batch, EdgeActivationBatch) else tuple(batch)
    if not edges:
        raise ValueError("TER requires at least one edge activation")
    if not all(isinstance(edge, EdgeActivation) for edge in edges):
        raise TypeError("TER requires EdgeActivation values")

    endpoint_pairs = [(edge.source, edge.target) for edge in edges]
    if len(set(endpoint_pairs)) != len(endpoint_pairs):
        raise ValueError("TER requires unique directed endpoint pairs")

    outgoing: dict[Hashable, list[Hashable]] = defaultdict(list)
    incoming: dict[Hashable, list[Hashable]] = defaultdict(list)
    for source, target in endpoint_pairs:
        outgoing[source].append(target)
        incoming[target].append(source)

    forward_rank = edge_pagerank(
        outgoing, damping=damping, max_iter=max_iter, tolerance=tolerance
    )
    reverse_rank = edge_pagerank(
        incoming, damping=damping, max_iter=max_iter, tolerance=tolerance
    )
    importances = [
        forward_rank[(edge.source, edge.target)]
        * reverse_rank[(edge.target, edge.source)]
        for edge in edges
    ]
    importance_sum = math.fsum(importances)
    if not math.isfinite(importance_sum) or importance_sum <= 0:
        raise ValueError("TER PageRank produced invalid edge importance")

    return math.fsum(
        _effective_rank(edge.activation) * importance / importance_sum
        for edge, importance in zip(edges, importances, strict=True)
    )


def edge_pagerank(
    graph: Mapping[Hashable, Sequence[Hashable]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> dict[tuple[Hashable, Hashable], float]:
    """Compute the edge-centric PageRank iteration used by TER-Score."""
    if not 0 <= damping < 1:
        raise ValueError("damping must satisfy 0 <= damping < 1")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    edges = [(source, target) for source, targets in graph.items() for target in targets]
    if not edges:
        raise ValueError("edge PageRank requires at least one edge")
    if len(set(edges)) != len(edges):
        raise ValueError("edge PageRank requires unique directed edges")

    edge_count = len(edges)
    ranks = {edge: 1.0 / edge_count for edge in edges}
    teleport = (1.0 - damping) / edge_count

    for _ in range(max_iter):
        updated = {edge: teleport for edge in edges}
        for source, targets in graph.items():
            if not targets:
                continue
            incoming_rank = math.fsum(
                ranks.get((predecessor, source), 0.0)
                for predecessor, predecessor_targets in graph.items()
                if source in predecessor_targets
            )
            contribution = damping * incoming_rank / len(targets)
            for target in targets:
                updated[(source, target)] += contribution

        delta = math.fsum(abs(updated[edge] - ranks[edge]) for edge in edges)
        if delta < tolerance:
            break
        ranks = updated

    return ranks


def _effective_rank(activation: Any) -> float:
    import torch
    import torch.nn.functional as functional

    _validate_activation(activation)
    tensor = activation.detach()
    batch_size, channels, height, width = tensor.shape
    if height * width > 1024:
        tensor = functional.adaptive_avg_pool2d(tensor, (32, 32))
        height = width = 32

    tensor = tensor.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
    if tensor.shape[1] < tensor.shape[2]:
        tensor = tensor.transpose(2, 1)
    if tensor.shape[1] <= 1:
        raise ValueError("edge activation must provide at least two correlation observations")

    mean = tensor.mean(dim=(0, 1), keepdim=True)
    standard_deviation = tensor.std(dim=(0, 1), keepdim=True)
    normalized = (tensor - mean) / (standard_deviation + 1e-8)
    correlation = torch.bmm(normalized.transpose(2, 1), normalized)
    correlation = correlation / (normalized.shape[1] - 1)
    correlation = torch.nan_to_num(correlation)

    singular_values = torch.linalg.svdvals(correlation)
    totals = singular_values.sum(dim=1, keepdim=True)
    probabilities = torch.where(totals > 0, singular_values / totals, 0.0)
    entropy = torch.where(
        probabilities > 0,
        -probabilities * probabilities.log(),
        torch.zeros_like(probabilities),
    ).sum(dim=1)
    effective_ranks = torch.where(totals.squeeze(1) > 0, entropy.exp(), 0.0)
    return float(effective_ranks.mean().cpu())


def _validate_endpoint(endpoint: object, name: str) -> None:
    if not isinstance(endpoint, Hashable):
        raise TypeError(f"edge {name} must be hashable")
    try:
        hash(endpoint)
    except TypeError as error:
        raise TypeError(f"edge {name} must be hashable") from error


def _validate_activation(activation: Any) -> None:
    import torch

    if not isinstance(activation, torch.Tensor):
        raise TypeError("edge activation must be a torch.Tensor")
    if activation.ndim != 4:
        raise ValueError("edge activation must have shape [B, C, H, W]")
    if any(size <= 0 for size in activation.shape):
        raise ValueError("edge activation dimensions must be positive")
    if not activation.is_floating_point():
        raise TypeError("edge activation must use a floating-point dtype")
    if not bool(torch.isfinite(activation).all()):
        raise ValueError("edge activation must contain only finite values")


__all__ = [
    "EdgeActivation",
    "EdgeActivationBatch",
    "compute_er_score",
    "compute_ter_score",
    "edge_pagerank",
]
