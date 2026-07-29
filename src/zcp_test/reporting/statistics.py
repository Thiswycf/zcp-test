from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def ndcg_at_k(target: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, target.size)
    if k <= 0:
        return float("nan")
    relevance = stats.rankdata(target, method="average") - 1.0
    order = np.argsort(score)[::-1][:k]
    ideal = np.argsort(target)[::-1][:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[order] / discounts))
    idcg = float(np.sum(relevance[ideal] / discounts))
    return dcg / idcg if idcg else float("nan")


def correlation_summary(target: list[float], score: list[float], ndcg_k: int = 10) -> dict[str, Any]:
    if len(target) != len(score):
        raise ValueError("target and score lengths differ")
    target_array = np.asarray(target, dtype=float)
    score_array = np.asarray(score, dtype=float)
    finite = np.isfinite(target_array) & np.isfinite(score_array)
    valid_target, valid_score = target_array[finite], score_array[finite]
    result: dict[str, Any] = {
        "sample_count": len(target),
        "valid_count": int(finite.sum()),
        "invalid_count": int((~finite).sum()),
        "target_ties": int(len(valid_target) - len(np.unique(valid_target))),
        "score_ties": int(len(valid_score) - len(np.unique(valid_score))),
    }
    if valid_target.size < 2 or np.unique(valid_target).size < 2 or np.unique(valid_score).size < 2:
        return {**result, "spearman": None, "kendall_tau_b": None, "pearson": None, "ndcg": None}
    result.update(
        spearman=float(stats.spearmanr(valid_target, valid_score).statistic),
        kendall_tau_b=float(stats.kendalltau(valid_target, valid_score, variant="b").statistic),
        pearson=float(stats.pearsonr(valid_target, valid_score).statistic),
        ndcg=float(ndcg_at_k(valid_target, valid_score, ndcg_k)),
        ndcg_k=min(ndcg_k, valid_target.size),
    )
    return {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in result.items()}

