from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


AZ_NAS_SOURCE = "https://github.com/cvlab-yonsei/AZ-NAS"
AZ_NAS_COMMIT = "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47"


def autoformer_components(model: Any, inputs: Any) -> dict[str, float]:
    import torch

    extractor = getattr(model, "extract_res_features", None)
    complexity = getattr(model, "official_complexity_ops", None)
    if not callable(extractor) or not callable(complexity):
        raise NotImplementedError(
            "AZ-NAS AutoFormer requires extract_res_features() and official_complexity_ops()"
        )
    features = extractor(inputs)
    if len(features) < 2:
        raise ValueError("AZ-NAS AutoFormer requires at least two residual features")
    expressivity_scores = []
    for feature in features:
        if feature.ndim != 3:
            raise ValueError("AZ-NAS AutoFormer residual features must have shape [B, N, C]")
        flattened = feature.detach().clone().reshape(-1, feature.shape[-1])
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / centered.shape[0]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total = eigenvalues.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError("AZ-NAS AutoFormer encountered a degenerate feature covariance")
        probabilities = eigenvalues / total
        entropy = (-(probabilities) * torch.log(probabilities + 1e-8)).sum()
        expressivity_scores.append(float(entropy))

    trainability_scores = []
    for index in reversed(range(1, len(features))):
        output_feature = features[index]
        input_feature = features[index - 1]
        output_gradient = torch.empty_like(output_feature).bernoulli_(0.5).mul_(2).sub_(1)
        input_gradient = torch.autograd.grad(
            outputs=output_feature,
            inputs=input_feature,
            grad_outputs=output_gradient,
            retain_graph=False,
        )[0]
        if output_gradient.shape == input_gradient.shape and torch.equal(
            input_gradient, output_gradient
        ):
            continue
        if output_gradient.shape[:2] != input_gradient.shape[:2]:
            raise NotImplementedError(
                "AZ-NAS AutoFormer does not define transitions with different token counts"
            )
        output_matrix = output_gradient.reshape(-1, output_gradient.shape[-1])
        input_matrix = input_gradient.reshape(-1, input_gradient.shape[-1])
        cross_matrix = input_matrix.T @ output_matrix / output_matrix.shape[0]
        if cross_matrix.shape[0] < cross_matrix.shape[1]:
            cross_matrix = cross_matrix.T
        maximum = torch.linalg.svdvals(cross_matrix).max()
        trainability_scores.append(float(-maximum - 1 / (maximum + 1e-6) + 2))
    if not trainability_scores:
        raise ValueError("AZ-NAS AutoFormer found no non-identity residual transitions")
    components = {
        "expressivity": float(sum(expressivity_scores)),
        "trainability": float(sum(trainability_scores) / len(trainability_scores)),
        "complexity": float(complexity()),
    }
    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("AZ-NAS AutoFormer returned a non-finite component")
    return components


def log_rank_aggregate(
    rows: Sequence[Mapping[str, float]], component_names: Sequence[str]
) -> list[float]:
    import scipy.stats

    if not rows:
        raise ValueError("AZ-NAS rank aggregation requires at least one candidate")
    names = tuple(component_names)
    if not names:
        raise ValueError("AZ-NAS rank aggregation requires component names")
    count = len(rows)
    totals = [0.0] * count
    for name in names:
        values = []
        for row in rows:
            if name not in row:
                raise ValueError(f"AZ-NAS candidate is missing component {name!r}")
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"AZ-NAS component {name!r} is not finite")
            values.append(value)
        ranks = scipy.stats.rankdata(values)
        for index, rank in enumerate(ranks):
            totals[index] += math.log(float(rank) / count)
    return totals
