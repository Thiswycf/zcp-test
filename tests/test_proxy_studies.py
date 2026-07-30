from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from zcp_test.reporting.proxy_studies import (
    OBSERVATION_NOTE,
    TIE_STRATEGY,
    plot_proxy_proxy_heatmap,
    plot_proxy_target_heatmap,
    proxy_complementarity,
    proxy_proxy_correlations,
    proxy_proxy_top_k,
    proxy_study,
    proxy_target_protocol_matrix,
)


def _row(
    architecture_id: str,
    proxy_id: str,
    score: float,
    target: float,
    *,
    budget: int = 12,
    fingerprint: str = "input-a",
    seed: int = 7,
    version: str = "1",
    direction: str = "maximize",
    target_split: str = "valid",
) -> dict[str, object]:
    return {
        "architecture_id": architecture_id,
        "proxy_id": proxy_id,
        "proxy_version": version,
        "component": "score",
        "score": score,
        "direction": direction,
        "target_value": target,
        "target_direction": "maximize",
        "target_split": target_split,
        "target_epoch_budget": budget,
        "benchmark_id": "bench",
        "search_space_id": "space",
        "dataset": "valid",
        "input_source": "batch",
        "input_fingerprint": fingerprint,
        "seed": seed,
        "status": "ok",
    }


def test_proxy_target_matrix_strictly_separates_protocol_and_proxy_version() -> None:
    rows = []
    for budget in (12, 36):
        for index in range(1, 4):
            rows.append(_row(f"a{index}", "p", index, index, budget=budget, version="1"))
    for index in range(1, 4):
        rows.append(_row(f"a{index}", "p", 4 - index, index, budget=12, version="2"))

    result = proxy_target_protocol_matrix(rows)
    spearman = result["long"].query("method == 'spearman'")

    assert len(spearman) == 3
    assert set(spearman["target_epoch_budget"]) == {12, 36}
    assert set(spearman["proxy_label"]) == {"p/score@1", "p/score@2"}
    assert spearman.set_index(["target_epoch_budget", "proxy_version"])["correlation"].to_dict() == {
        (12, "1"): 1.0,
        (12, "2"): -1.0,
        (36, "1"): 1.0,
    }
    assert {"p/score@1", "p/score@2"}.issubset(result["matrix"].columns)
    architecture_matrix = result["architecture_matrix"]
    assert len(architecture_matrix) == 6
    assert architecture_matrix.query("target_epoch_budget == 12")["p/score@2"].notna().sum() == 3


def test_input_fingerprint_and_seed_are_not_silently_mixed() -> None:
    rows = []
    for fingerprint, seed in (("first", 1), ("second", 2)):
        for index in range(1, 4):
            rows.extend(
                [
                    _row(f"a{index}", "p1", index, index, fingerprint=fingerprint, seed=seed),
                    _row(f"a{index}", "p2", index, index, fingerprint=fingerprint, seed=seed),
                ]
            )

    correlations = proxy_proxy_correlations(rows)

    assert len(correlations) == 2
    assert set(correlations["input_fingerprint"]) == {"first", "second"}
    assert set(correlations["seed"]) == {1, 2}
    assert correlations["common_sample_count"].eq(3).all()


def test_matrix_preserves_protocol_groups_with_missing_condition_values() -> None:
    rows = []
    for fingerprint in (None, "known"):
        for index in range(1, 4):
            rows.append(
                _row(f"a{index}", "p", index, index, fingerprint=fingerprint)  # type: ignore[arg-type]
            )

    result = proxy_target_protocol_matrix(rows)
    spearman = result["long"].query("method == 'spearman'")
    matrix = result["matrix"].query("method == 'spearman'")

    assert len(spearman) == 2
    assert len(matrix) == 2
    assert matrix["input_fingerprint"].isna().sum() == 1


def test_duplicate_architecture_proxy_in_protocol_is_rejected() -> None:
    row = _row("a1", "p", 1, 1)

    with pytest.raises(ValueError, match="Duplicate architecture×proxy"):
        proxy_target_protocol_matrix([row, dict(row)])


def test_proxy_proxy_correlations_use_common_architecture_join_and_coverage() -> None:
    rows = [
        _row("a1", "left", 1, 1),
        _row("a2", "left", 2, 2),
        _row("a3", "left", 3, 3),
        _row("a2", "right", 4, 2),
        _row("a3", "right", 3, 3),
        _row("a4", "right", 2, 4),
    ]

    result = proxy_proxy_correlations(rows).iloc[0]

    assert result["left_sample_count"] == 3
    assert result["right_sample_count"] == 3
    assert result["common_sample_count"] == 2
    assert result["union_sample_count"] == 4
    assert result["common_coverage"] == pytest.approx(0.5)
    assert result["spearman"] == pytest.approx(-1.0)
    assert result["kendall_tau_b"] == pytest.approx(-1.0)
    assert result["pearson"] == pytest.approx(-1.0)


def test_direction_adjustment_and_stable_top_k_ties() -> None:
    rows = []
    left_scores = {"a1": 1, "a2": 1, "a3": 3}
    right_costs = {"a1": 4, "a2": 3, "a3": 1}
    for architecture_id in ("a1", "a2", "a3"):
        target = int(architecture_id[1:])
        rows.append(_row(architecture_id, "left", left_scores[architecture_id], target))
        rows.append(
            _row(
                architecture_id,
                "right",
                right_costs[architecture_id],
                target,
                direction="minimize",
            )
        )

    correlations = proxy_proxy_correlations(rows).iloc[0]
    top_k = proxy_proxy_top_k(rows, k=(1, 2))

    assert correlations["spearman"] > 0
    assert list(top_k["intersection_count"]) == [1, 1]
    assert list(top_k["union_count"]) == [1, 3]
    assert top_k["tie_strategy"].eq(TIE_STRATEGY).all()


def test_complementarity_reports_residual_redundancy_and_top_k_union_gain() -> None:
    rows = []
    left_scores = {"a1": 7, "a2": 1, "a3": 2, "a4": 3, "a5": 4, "a6": 8}
    right_scores = {"a1": 6, "a2": 7, "a3": 2, "a4": 3, "a5": 8, "a6": 1}
    for index in range(1, 7):
        architecture_id = f"a{index}"
        rows.append(_row(architecture_id, "left", left_scores[architecture_id], index))
        rows.append(_row(architecture_id, "right", right_scores[architecture_id], index))

    result = proxy_complementarity(rows, k=2).iloc[0]

    assert result["left_top_k_recall"] == pytest.approx(0.5)
    assert result["right_top_k_recall"] == pytest.approx(0.5)
    assert result["top_k_union_recall"] == pytest.approx(1.0)
    assert result["top_k_union_marginal_gain"] == pytest.approx(0.5)
    assert -1 <= result["residual_pearson"] <= 1
    assert result["interpretation"] == OBSERVATION_NOTE


def test_validation_rank_fusion_can_improve_over_each_single_proxy() -> None:
    rows = []
    noise = (3, -3, 3, -3, 3, -3, 3, -3)
    for index, error in enumerate(noise, 1):
        architecture_id = f"a{index}"
        rows.append(_row(architecture_id, "left", index + error, index))
        rows.append(_row(architecture_id, "right", index - error, index))

    result = proxy_complementarity(rows, k=2, validation_fraction=0.5).iloc[0]

    assert result["fusion_validation_count"] == 4
    assert result["fusion_evaluation_count"] == 4
    assert result["fusion_weight_left"] == pytest.approx(0.25)
    assert result["fusion_gain_over_best_single"] > 0
    assert result["fusion_split_strategy"] == "architecture_id_ascending_prefix_validation"
    assert result["fusion_status"] == "ok"


def test_rank_fusion_never_learns_weights_from_test_targets() -> None:
    rows = []
    for index in range(1, 7):
        rows.append(_row(f"a{index}", "left", index, index, target_split="test"))
        rows.append(_row(f"a{index}", "right", 7 - index, index, target_split="test"))

    result = proxy_complementarity(rows, k=2).iloc[0]

    assert result["fusion_status"] == "unsupported_target_split"
    assert pd.isna(result["fusion_weight_left"])
    assert result["fusion_validation_count"] == 0


def test_rank_fusion_weight_does_not_depend_on_evaluation_proxy_distribution() -> None:
    rows = []
    for index in range(1, 9):
        rows.append(_row(f"a{index}", "left", index, index))
        rows.append(_row(f"a{index}", "right", 9 - index, index))
    changed = [dict(row) for row in rows]
    for row in changed:
        if str(row["architecture_id"]) >= "a5":
            row["score"] = float(row["score"]) * -100

    original = proxy_complementarity(rows, k=2).iloc[0]
    modified = proxy_complementarity(changed, k=2).iloc[0]

    assert original["fusion_weight_left"] == modified["fusion_weight_left"]


def test_complete_study_is_csv_friendly_and_heatmaps_render(tmp_path: Path) -> None:
    rows = []
    for index in range(1, 5):
        rows.append(_row(f"a{index}", "p1", index, index))
        rows.append(_row(f"a{index}", "p2", 5 - index, index))

    study = proxy_study((row for row in rows), k=2)
    target_path = tmp_path / "proxy-target.svg"
    pair_path = tmp_path / "proxy-proxy.svg"
    plot_proxy_target_heatmap(study, target_path)
    plot_proxy_proxy_heatmap(study["proxy_proxy_correlations"], pair_path)

    assert set(study) == {
        "proxy_target_long",
        "proxy_target_matrix",
        "proxy_target_observations",
        "proxy_target_architecture_matrix",
        "proxy_proxy_correlations",
        "proxy_proxy_top_k",
        "complementarity",
    }
    assert all(isinstance(table.index, pd.RangeIndex) for table in study.values())
    assert target_path.is_file()
    assert pair_path.is_file()
