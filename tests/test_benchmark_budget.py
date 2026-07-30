from __future__ import annotations

import pytest

from zcp_test.reporting.benchmark_budget import nasbench101_budget_study


class FakeNasBench101Adapter:
    search_space_id = "nb101_dag"

    def capabilities(self):
        return {"epoch_budgets": [4, 12, 36, 108]}

    def metadata(self):
        return {"version": "fixture"}

    def canonicalize(self, specification):
        return dict(specification)

    def architecture_id(self, specification):
        return str(specification["id"])

    def query_metrics(self, architecture, metric):
        scale = int(architecture.spec["quality"])
        return {metric.metric_name: float(scale * metric.epoch_budget)}


def _rows():
    rows = []
    for index in range(1, 5):
        for proxy_id, score in (("aligned", index), ("reversed", 5 - index)):
            rows.append(
                {
                    "architecture_id": f"a{index}",
                    "architecture": {"id": f"a{index}", "quality": index},
                    "proxy_id": proxy_id,
                    "component": "score",
                    "score": float(score),
                    "direction": "maximize",
                    "status": "ok",
                }
            )
    return rows


def test_nasbench101_budget_study_queries_every_budget_and_reports_stability():
    result = nasbench101_budget_study(
        _rows(),
        FakeNasBench101Adapter(),
        budgets=(4, 12, 108),
        bootstrap_samples=0,
        top_k=(2,),
    )

    assert len(result["detailed"]) == 4 * 2 * 3
    assert set(result["correlations"]["epoch_budget"]) == {4, 12, 108}
    aligned = result["correlations"].query("proxy_id == 'aligned'")
    reversed_proxy = result["correlations"].query("proxy_id == 'reversed'")
    assert aligned["spearman"].eq(1.0).all()
    assert reversed_proxy["spearman"].eq(-1.0).all()
    assert result["rank_stability"]["top_2_jaccard"].eq(1.0).all()
    retrieval = result["top_k_retrieval"]
    aligned_retrieval = retrieval.query("proxy_id == 'aligned' and requested_k == 2")
    reversed_retrieval = retrieval.query("proxy_id == 'reversed' and requested_k == 2")
    assert aligned_retrieval["precision_at_k"].eq(1.0).all()
    assert aligned_retrieval["mean_regret"].eq(0.0).all()
    assert reversed_retrieval["precision_at_k"].eq(0.0).all()
    assert reversed_retrieval["mean_regret"].gt(0.0).all()


def test_nasbench101_budget_study_rejects_hash_and_budget_mismatch():
    adapter = FakeNasBench101Adapter()
    invalid = _rows()
    invalid[0]["architecture_id"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        nasbench101_budget_study(invalid, adapter)
    with pytest.raises(ValueError, match="Unsupported"):
        nasbench101_budget_study(_rows(), adapter, budgets=(200,))
