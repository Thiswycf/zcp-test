from __future__ import annotations

import math

import pytest

from zcp_test.reporting.benchmark_budget import nasbench101_budget_study
from zcp_test.reporting.benchmark_report import write_benchmark_study
from zcp_test.reporting.benchmark_report import _budget_plot_groups


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
    chain = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    triangle = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    specifications = (
        ("a1", 1, chain, "conv3x3-bn-relu"),
        ("a2", 2, chain, "conv1x1-bn-relu"),
        ("a3", 3, triangle, "conv3x3-bn-relu"),
        ("a4", 4, triangle, "conv1x1-bn-relu"),
    )
    rows = []
    for architecture_id, quality, matrix, operation in specifications:
        for proxy_id, score in (("aligned", quality), ("reversed", 5 - quality)):
            rows.append(
                {
                    "architecture_id": architecture_id,
                    "architecture": {
                        "id": architecture_id,
                        "quality": quality,
                        "matrix": matrix,
                        "operations": ["input", operation, "output"],
                    },
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


def test_nasbench101_budget_study_keeps_proxy_seed_separate_from_target_repeat():
    rows = _rows()
    seeded = []
    for seed in (1, 2):
        for row in rows:
            record = dict(row)
            record["seed"] = seed
            if seed == 2:
                record["score"] = -float(record["score"])
            seeded.append(record)

    result = nasbench101_budget_study(
        seeded,
        FakeNasBench101Adapter(),
        budgets=(12,),
        repeat_index=0,
        bootstrap_samples=0,
    )

    assert set(result["correlations"]["seed"]) == {1, 2}
    assert set(result["detailed"]["target_seed"]) == {0}


def test_nasbench101_budget_study_merges_shards_but_separates_evaluation_seeds():
    rows = []
    for seed in (7, 8):
        for row in _rows():
            record = dict(row)
            architecture_index = int(record["architecture_id"][1:])
            record["seed"] = seed
            record["run_id"] = f"run-{seed}-{architecture_index % 2}"
            record["source_run"] = f"shard-{architecture_index % 2}/scores.jsonl"
            if seed == 8:
                record["score"] = -float(record["score"])
            rows.append(record)

    result = nasbench101_budget_study(
        rows,
        FakeNasBench101Adapter(),
        budgets=(12,),
        bootstrap_samples=0,
        top_k=(2,),
    )

    correlations = result["correlations"]
    assert len(correlations) == 4
    assert set(correlations["seed"]) == {7, 8}
    assert correlations["sample_count"].eq(4).all()
    for table in result.values():
        assert "run_id" not in table.columns
        assert "source_run" not in table.columns


def test_nasbench101_budget_study_preserves_failed_coverage_for_correlations():
    rows = _rows()
    failed = next(row for row in rows if row["proxy_id"] == "aligned" and row["architecture_id"] == "a4")
    failed["status"] = "failed"
    failed["score"] = None

    result = nasbench101_budget_study(
        rows,
        FakeNasBench101Adapter(),
        budgets=(12,),
        bootstrap_samples=0,
        top_k=(2,),
    )

    correlation = result["correlations"].query("proxy_id == 'aligned'").iloc[0]
    assert correlation["sample_count"] == 3
    assert correlation["total_count"] == 4
    assert correlation["failed_count"] == 1
    assert correlation["coverage"] == pytest.approx(0.75)
    coverage = result["score_coverage"].query("proxy_id == 'aligned'").iloc[0]
    assert coverage["total_count"] == 4
    assert coverage["successful_count"] == 3
    assert coverage["failed_count"] == 1
    assert coverage["coverage"] == pytest.approx(0.75)
    failed_detail = result["detailed"].query(
        "proxy_id == 'aligned' and architecture_id == 'a4'"
    ).iloc[0]
    assert failed_detail["status"] == "failed"
    assert math.isnan(failed_detail["score"])


def test_nasbench101_budget_study_reports_structure_features_and_unique_neighbors():
    result = nasbench101_budget_study(
        _rows(),
        FakeNasBench101Adapter(),
        budgets=(12,),
        bootstrap_samples=0,
        top_k=(2,),
    )

    features = result["architecture_features"].set_index("architecture_id")
    assert features.loc["a1"].to_dict() == {
        "vertices": 3,
        "edges": 2,
        "longest_path_depth": 2,
        "conv3_count": 1,
        "conv1_count": 0,
        "maxpool_count": 0,
    }
    assert features.loc["a3", "edges"] == 3
    assert set(result["feature_strata"]["feature"]) == {
        "vertices",
        "edges",
        "longest_path_depth",
        "conv3_count",
        "conv1_count",
        "maxpool_count",
    }
    assert set(result["structure_controlled_correlations"]["sample_count"]) == {0}

    neighbors = result["edit_neighbors"]
    assert len(neighbors) == 4
    assert not neighbors.duplicated(
        ["architecture_id_left", "architecture_id_right"]
    ).any()
    assert set(neighbors["edit_type"]) == {"operation", "edge"}
    expected_pairs = {
        ("a1", "a2", "operation"),
        ("a1", "a3", "edge"),
        ("a2", "a4", "edge"),
        ("a3", "a4", "operation"),
    }
    assert set(
        neighbors[["architecture_id_left", "architecture_id_right", "edit_type"]].itertuples(
            index=False, name=None
        )
    ) == expected_pairs
    summaries = result["neighborhood_correlations"].set_index("proxy_id")
    assert summaries.loc["aligned", "pair_count"] == 4
    assert summaries.loc["aligned", "operation_pair_count"] == 2
    assert summaries.loc["aligned", "edge_pair_count"] == 2
    assert summaries.loc["aligned", "spearman"] == pytest.approx(1.0)
    assert summaries.loc["aligned", "kendall_tau_b"] == pytest.approx(1.0)
    assert summaries.loc["aligned", "direction_agreement_rate"] == pytest.approx(1.0)
    assert summaries.loc["reversed", "spearman"] == pytest.approx(-1.0)
    assert summaries.loc["reversed", "direction_agreement_rate"] == pytest.approx(0.0)


def test_nasbench101_budget_study_controls_within_identical_structure_strata():
    chain = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    chain_with_edge = [
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    rows = []
    for architecture_id, quality, matrix, operations in (
        ("b1", 1, chain, ["conv3x3-bn-relu", "conv1x1-bn-relu"]),
        ("b2", 2, chain, ["conv1x1-bn-relu", "conv3x3-bn-relu"]),
        ("b3", 3, chain_with_edge, ["conv3x3-bn-relu", "conv1x1-bn-relu"]),
        ("b4", 4, chain_with_edge, ["conv1x1-bn-relu", "conv3x3-bn-relu"]),
    ):
        rows.append(
            {
                "architecture_id": architecture_id,
                "architecture": {
                    "id": architecture_id,
                    "quality": quality,
                    "matrix": matrix,
                    "operations": ["input", *operations, "output"],
                },
                "proxy_id": "aligned",
                "component": "score",
                "score": float(quality),
                "direction": "maximize",
                "status": "ok",
            }
        )

    result = nasbench101_budget_study(
        rows,
        FakeNasBench101Adapter(),
        budgets=(12,),
        bootstrap_samples=0,
    )

    controlled = result["structure_controlled_correlations"].iloc[0]
    assert controlled["sample_count"] == 4
    assert controlled["informative_stratum_count"] == 2
    assert controlled["spearman"] == pytest.approx(1.0)
    assert controlled["kendall_tau_b"] == pytest.approx(1.0)


def test_nasbench101_budget_report_plots_structure_and_neighborhood(tmp_path):
    result = nasbench101_budget_study(
        _rows(),
        FakeNasBench101Adapter(),
        budgets=(4, 108),
        bootstrap_samples=0,
    )

    report = write_benchmark_study(
        result,
        tmp_path,
        view="budget",
        benchmark_id="nasbench101",
    )

    assert {
        "budget_structure_controlled.png",
        "budget_structure_controlled.svg",
        "budget_neighborhood_agreement.png",
        "budget_neighborhood_agreement.svg",
    }.issubset(report["artifacts"])


def test_budget_plot_groups_keep_proxy_versions_and_seeds_separate():
    import pandas as pd

    table = pd.DataFrame(
        [
            {"proxy_id": "synflow", "component": "score", "proxy_version": version,
             "seed": seed, "epoch_budget": budget, "spearman": 0.5}
            for version in ("1", "double-v2")
            for seed in (2026, 2027)
            for budget in (4, 108)
        ]
    )

    fields, groups = _budget_plot_groups(table)

    assert fields == ["proxy_id", "component", "proxy_version", "seed"]
    assert len(list(groups)) == 4
