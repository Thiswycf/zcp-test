from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from zcp_test.reporting.analysis import (
    bootstrap_correlation,
    build_report_bundle,
    correlation_table,
    plot_heatmap,
    proxy_cost_pareto,
    rank_aggregation,
    read_scores,
    sample_size_convergence,
    sensitivity_rank_table,
    top_k_comparison,
    transfer_correlation_table,
    validate_analysis_scores,
)
from zcp_test.reporting.monitor import read_jsonl_tolerant, refresh_once
from zcp_test.reporting.reports import curve_plot, static_html


matplotlib.use("Agg")


def _score_rows() -> list[dict[str, object]]:
    return [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "synflow",
            "component": "sum",
            "score": float(index),
            "target_value": float(index * 2),
            "target_metric": "valid_accuracy",
            "seed": index % 2,
            "status": "ok",
        }
        for index in range(1, 7)
    ]


def test_read_scores_accepts_flat_and_nested_schema(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    rows = [
        _score_rows()[0],
        {
            "architecture": {"id": "nested"},
            "proxy": {"id": "naswot", "component": "mean"},
            "result": {"score": 3.5, "status": "ok", "duration_seconds": 0.2},
            "target": {"metric": "test_accuracy", "value": 91.2},
            "run": {"id": "run-1"},
            "dataset": {"id": "cifar10"},
        },
        {
            "schema_version": "2.0",
            "architecture_id": "schema-2",
            "proxy_id": "grad_norm",
            "primary_component": "total",
            "components": {"total": 8.0, "mean": 2.0},
            "score": 8.0,
            "target_value": 88.0,
            "status": "ok",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    frame = read_scores(path)

    assert list(frame["architecture_id"]) == ["a1", "nested", "schema-2", "schema-2"]
    assert frame.loc[1, "proxy_id"] == "naswot"
    assert frame.loc[1, "target_value"] == pytest.approx(91.2)
    assert frame.loc[1, "dataset"] == "cifar10"
    assert list(frame.loc[2:, "component"]) == ["total", "mean"]
    assert list(frame.loc[2:, "score"]) == [8.0, 2.0]


def test_bootstrap_correlation_and_group_table_are_reproducible() -> None:
    first = bootstrap_correlation(range(8), range(8), samples=200, seed=7)
    second = bootstrap_correlation(range(8), range(8), samples=200, seed=7)

    assert first == second
    assert first["estimate"] == pytest.approx(1.0)
    assert first["lower"] == pytest.approx(1.0)
    table = correlation_table(_score_rows(), bootstrap_samples=50)
    assert len(table) == 2
    assert table["spearman"].eq(1.0).all()
    assert table["sample_count"].eq(3).all()
    assert table["total_count"].eq(3).all()
    assert table["invalid_count"].eq(0).all()
    assert table["coverage"].eq(1.0).all()
    assert table["correlation_status"].eq("ok").all()


def test_correlation_table_explains_constants_ties_invalid_values_and_direction() -> None:
    rows = [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "constant",
            "component": "score",
            "score": score,
            "target_value": target,
            "direction": "minimize",
            "target_direction": "maximize",
            "proxy_implementation_fidelity": "paper_formula_port_unverified",
            "status": "ok",
        }
        for index, (score, target) in enumerate(
            [(4.0, 1.0), (4.0, 1.0), (4.0, 2.0), (float("nan"), 3.0)]
        )
    ]

    record = correlation_table(rows).iloc[0]

    assert record["total_count"] == 4
    assert record["successful_count"] == 4
    assert record["failed_count"] == 0
    assert record["sample_count"] == 3
    assert record["invalid_count"] == 1
    assert record["coverage"] == pytest.approx(0.75)
    assert record["target_unique_count"] == 2
    assert record["score_unique_count"] == 1
    assert record["target_tied_observations"] == 2
    assert record["score_tied_observations"] == 3
    assert record["correlation_status"] == "constant_score"
    assert pd.isna(record["spearman"])
    assert record["score_direction"] == "minimize"
    assert record["score_direction_transform"] == "negated"
    assert record["target_direction"] == "maximize"
    assert record["target_direction_transform"] == "identity"
    assert record["proxy_implementation_fidelity"] == "paper_formula_port_unverified"


def test_legacy_params_resource_direction_is_migrated_for_accuracy_correlation() -> None:
    rows = [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "params",
            "proxy_version": "1",
            "component": "score",
            "score": float(index),
            "target_value": float(index),
            "direction": "minimize",
            "target_direction": "maximize",
            "status": "ok",
        }
        for index in range(1, 4)
    ]

    frame = read_scores(rows)
    record = correlation_table(rows).iloc[0]

    assert frame["direction"].eq("maximize").all()
    assert frame["proxy_version"].eq("count-v2").all()
    assert frame["reported_proxy_version"].eq("1").all()
    assert frame["reported_direction"].eq("minimize").all()
    assert frame["resource_direction"].eq("minimize").all()
    assert frame["direction_migration"].eq(
        "legacy-resource-direction-to-accuracy-v2"
    ).all()
    assert record["spearman"] == pytest.approx(1.0)
    assert record["score_direction_transform"] == "identity"

    without_version = [dict(row) for row in rows]
    for row in without_version:
        row.pop("proxy_version")
    missing_version_frame = read_scores(without_version)
    assert missing_version_frame["proxy_version"].eq("count-v2").all()
    assert missing_version_frame["reported_proxy_version"].eq("1").all()


def test_correlation_merges_legacy_and_native_v2_proxy_rows() -> None:
    rows = []
    for index in range(4):
        row = {
            "architecture_id": f"a{index}",
            "proxy_id": "params",
            "proxy_version": "count-v2",
            "component": "score",
            "score": float(index),
            "target_value": float(index),
            "direction": "maximize",
            "resource_direction": "minimize",
            "target_direction": "maximize",
            "status": "ok",
        }
        if index < 2:
            row.update(proxy_version="1", direction="minimize")
            row.pop("resource_direction")
        rows.append(row)

    table = correlation_table(rows)
    figure = plot_heatmap(table)

    assert len(table) == 1
    assert table.iloc[0]["sample_count"] == 4
    assert table.iloc[0]["legacy_direction_migrated_count"] == 2
    assert table.iloc[0]["legacy_reported_proxy_versions"] == "1"
    assert table.iloc[0]["direction_migrations"] == (
        "legacy-resource-direction-to-accuracy-v2"
    )
    assert table.iloc[0]["spearman"] == pytest.approx(1.0)
    assert figure.axes[0].images[0].get_array().shape == (1, 1)


def test_correlation_heatmap_treats_proxy_fidelity_as_proxy_metadata() -> None:
    rows = []
    for index in range(3):
        rows.extend(
            [
                {
                    "architecture_id": f"a{index}",
                    "proxy_id": "left",
                    "component": "score",
                    "proxy_version": "1",
                    "proxy_implementation_fidelity": "paper-port",
                    "score": float(index),
                    "target_value": float(index),
                    "status": "ok",
                },
                {
                    "architecture_id": f"a{index}",
                    "proxy_id": "right",
                    "component": "score",
                    "proxy_version": "1",
                    "proxy_implementation_fidelity": "project-extension",
                    "score": float(2 - index),
                    "target_value": float(index),
                    "status": "ok",
                },
            ]
        )

    correlations = correlation_table(rows)
    figure = plot_heatmap(correlations)

    assert len(correlations) == 2
    assert figure.axes[0].images[0].get_array().shape == (1, 2)
    assert max(len(label.get_text()) for label in figure.axes[0].get_xticklabels()) < 100


def test_correlation_table_counts_failed_invocations_and_rejects_duplicate_architectures() -> None:
    rows = _score_rows()[:3]
    for row in rows:
        row["seed"] = 0
    rows.append(
        {
            "architecture_id": "a4",
            "proxy_id": "synflow",
            "primary_component": "sum",
            "score": None,
            "target_value": 8.0,
            "target_metric": "valid_accuracy",
            "seed": 0,
            "status": "failed",
        }
    )

    record = correlation_table(rows).iloc[0]

    assert record["total_count"] == 4
    assert record["successful_count"] == 3
    assert record["failed_count"] == 1
    assert record["sample_count"] == 3
    assert record["invalid_count"] == 1
    assert record["coverage"] == pytest.approx(0.75)

    duplicated = [dict(rows[0]), dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate architecture IDs"):
        correlation_table(duplicated)


def test_correlation_table_applies_one_failed_call_to_each_multicomponent_series() -> None:
    rows = [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "composite",
            "primary_component": "first",
            "components": {"first": float(index), "second": float(index + 10)},
            "score": float(index),
            "target_value": float(index),
            "seed": 4,
            "status": "ok",
        }
        for index in (1, 2)
    ]
    rows.append(
        {
            "architecture_id": "a3",
            "proxy_id": "composite",
            "primary_component": "first",
            "components": {},
            "score": None,
            "target_value": 3.0,
            "seed": 4,
            "status": "failed",
        }
    )

    table = correlation_table(rows).set_index("component")

    assert set(table.index) == {"first", "second"}
    assert table["total_count"].eq(3).all()
    assert table["failed_count"].eq(1).all()
    assert table["sample_count"].eq(2).all()
    assert list(table["coverage"]) == pytest.approx([2 / 3, 2 / 3])


def test_bundle_writes_static_csv_png_svg_and_html(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = build_report_bundle(
        pd.DataFrame(_score_rows()),
        output,
        bootstrap_samples=25,
        top_k=(1, 3),
    )

    expected = {
        "scores.csv",
        "correlations.csv",
        "top_k.csv",
        "rank_aggregation.csv",
        "proxy_cost_pareto.csv",
        "transfer.csv",
        "scatter.png",
        "scatter.svg",
        "rank.png",
        "rank.svg",
        "heatmap.png",
        "heatmap.svg",
        "top_k.png",
        "top_k.svg",
        "sensitivity.png",
        "sensitivity.svg",
        "sensitivity_rank.csv",
        "sensitivity_rank.png",
        "sensitivity_rank.svg",
        "index.html",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert result["row_count"] == 6
    assert "scatter.svg" in (output / "index.html").read_text(encoding="utf-8")


def test_bundle_correlation_csv_retains_failed_invocations_in_coverage(tmp_path: Path) -> None:
    rows = _score_rows()[:3]
    for row in rows:
        row["seed"] = 0
    rows.append(
        {
            "architecture_id": "a4",
            "proxy_id": "synflow",
            "primary_component": "sum",
            "score": None,
            "target_value": 8.0,
            "target_metric": "valid_accuracy",
            "seed": 0,
            "status": "failed",
        }
    )

    output = tmp_path / "failed-coverage"
    build_report_bundle(rows, output, bootstrap_samples=10)
    record = pd.read_csv(output / "correlations.csv").iloc[0]

    assert record["total_count"] == 4
    assert record["failed_count"] == 1
    assert record["sample_count"] == 3
    assert record["coverage"] == pytest.approx(0.75)


def test_bundle_merges_mixed_legacy_shards_without_duplicate_heatmap_rows(
    tmp_path: Path,
) -> None:
    rows = []
    for seed in (7, 8):
        for index in range(3):
            for proxy_id in ("params", "flops"):
                row = {
                    "architecture_id": f"a{index}",
                    "proxy_id": proxy_id,
                    "proxy_version": "count-v2" if proxy_id == "params" else "thop-v2",
                    "component": "score",
                    "score": float(index + (proxy_id == "flops")),
                    "target_value": float(index),
                    "direction": "maximize",
                    "resource_direction": "minimize",
                    "target_direction": "maximize",
                    "benchmark_id": "bench",
                    "dataset": "valid",
                    "target_metric": "accuracy",
                    "target_split": "valid",
                    "input_source": "dataset",
                    "input_fingerprint": f"batch-{seed}",
                    "seed": seed,
                    "status": "ok",
                }
                if index < 2:
                    row.update(proxy_version="1", direction="minimize")
                    row.pop("resource_direction")
                rows.append(row)

    output = tmp_path / "mixed-shards"
    build_report_bundle(
        rows,
        output,
        bootstrap_samples=5,
        top_k=(1,),
        sample_sizes=(1, 3),
    )
    correlations = pd.read_csv(output / "correlations.csv")
    rank_stability = pd.read_csv(output / "sensitivity_rank.csv")

    assert len(correlations) == 4
    assert correlations["sample_count"].eq(3).all()
    assert correlations["legacy_direction_migrated_count"].eq(2).all()
    assert len(rank_stability) == 2
    assert rank_stability["common_count"].eq(3).all()
    assert rank_stability["spearman"].eq(1.0).all()
    assert len(sensitivity_rank_table(rows)) == 2
    assert (output / "heatmap.svg").exists()
    assert (output / "proxy_target_protocol_heatmap.svg").exists()
    assert "sample_size_convergence.csv" in (output / "index.html").read_text()


def test_monitor_retries_an_incomplete_trailing_record(tmp_path: Path) -> None:
    source = tmp_path / "scores.jsonl"
    source.write_bytes(b'{"score": 1}\n{"score":')

    rows, offset, partial = read_jsonl_tolerant(source)
    assert rows == [{"score": 1}]
    assert partial is True

    with source.open("ab") as handle:
        handle.write(b" 2}\n")
    result = refresh_once(source, tmp_path / "view", offset=offset, history=rows)

    assert result["rows"] == [{"score": 2}]
    assert result["row_count"] == 2
    assert result["new_row_count"] == 1
    assert result["ignored_partial_line"] is False
    monitor_html = (tmp_path / "view" / "monitor.html").read_text(encoding="utf-8")
    assert "Rows: 2" in monitor_html


def test_monitor_discovers_single_timestamped_run_and_rejects_ambiguous_root(tmp_path):
    first = tmp_path / "20260101T000000Z_first"
    first.mkdir()
    (first / "scores.jsonl").write_text('{"score": 1}\n')
    assert refresh_once(tmp_path)["source"] == str(first / "scores.jsonl")
    second = tmp_path / "20260101T000001Z_second"
    second.mkdir()
    (second / "training.jsonl").write_text('{"epoch": 0}\n')
    with pytest.raises(ValueError, match="select one"):
        refresh_once(tmp_path)


def test_legacy_html_creates_parent_directory(tmp_path):
    source = tmp_path / "scores.jsonl"
    source.write_text('{"score": 1}\n')
    destination = tmp_path / "nested" / "report.html"
    assert static_html(source, destination) == 1
    assert destination.exists()


def test_research_helper_tables_cover_aggregation_cost_convergence_and_transfer() -> None:
    rows = _score_rows()
    for index, row in enumerate(rows):
        row.update(
            {
                "benchmark_id": "bench-a" if index < 3 else "bench-b",
                "dataset": "valid",
                "target_split": "valid",
                "duration_seconds": 0.01 + index * 0.001,
                "peak_memory_mb": 10 + index,
            }
        )
    aggregation = rank_aggregation(rows)
    pareto = proxy_cost_pareto(rows)
    convergence = sample_size_convergence(rows, sizes=(2, 4), seed=3)
    transfer = transfer_correlation_table(rows)

    assert set(aggregation["benchmark_id"]) == {"bench-a", "bench-b"}
    assert set(aggregation["architecture_id"]) == {f"a{index}" for index in range(1, 7)}
    for _, protocol in aggregation.groupby(["benchmark_id", "seed"]):
        assert protocol["aggregate_rank"].is_monotonic_increasing
    assert aggregation["proxy_count"].eq(1).all()
    assert pareto.loc[0, "pareto"]
    assert set(convergence["requested_size"]) == {2, 4}
    assert convergence["sample_count"].le(convergence["requested_size"]).all()
    assert set(convergence["benchmark_id"]) == {"bench-a", "bench-b"}
    assert set(transfer["benchmark_id"]) == {"bench-a", "bench-b"}


def test_rank_aggregation_excludes_declared_proxy_aliases() -> None:
    rows = _score_rows()
    alias_rows = [
        {
            **row,
            "proxy_id": "proxy-alias",
            "proxy_alias_of": "proxy-a",
            "score": float(row["score"]) * -1,
        }
        for row in rows
    ]
    for row in rows:
        row["proxy_alias_of"] = None

    aggregation = rank_aggregation([*rows, *alias_rows])

    assert aggregation["proxy_count"].eq(1).all()


def test_ranking_and_correlation_respect_minimize_direction() -> None:
    rows = [
        {
            "architecture_id": f"a{index}",
            "proxy_id": "latency_like",
            "component": "score",
            "score": float(4 - index),
            "target_value": float(index),
            "target_split": "valid",
            "direction": "minimize",
            "status": "ok",
        }
        for index in range(1, 4)
    ]

    correlations = correlation_table(rows)
    aggregation = rank_aggregation(rows)

    assert correlations.loc[0, "spearman"] == pytest.approx(1.0)
    assert list(aggregation["architecture_id"]) == ["a3", "a2", "a1"]


def test_correlations_separate_target_protocol_and_respect_target_direction() -> None:
    rows = []
    for budget in (4, 108):
        for index in range(1, 4):
            rows.append(
                {
                    "architecture_id": f"a{index}",
                    "proxy_id": "proxy",
                    "component": "score",
                    "score": float(index),
                    "direction": "maximize",
                    "target_value": float(4 - index),
                    "target_direction": "minimize",
                    "target_epoch_budget": budget,
                    "benchmark_id": "nasbench101",
                    "status": "ok",
                }
            )

    correlations = correlation_table(rows)
    top_k = top_k_comparison(rows, k=1)

    assert len(correlations) == 2
    assert correlations["spearman"].eq(1.0).all()
    assert set(correlations["target_epoch_budget"]) == {4, 108}
    assert len(top_k) == 2
    assert top_k["overlap_fraction"].eq(1.0).all()


def test_bundle_supports_search_only_and_training_only_runs(tmp_path: Path) -> None:
    search = [
        {
            "record_kind": "generation_summary",
            "generation": generation,
            "best_so_far": 1.0 + generation,
            "mean_score": 0.5 + generation,
            "q25": 0.25 + generation,
            "q75": 0.75 + generation,
        }
        for generation in range(3)
    ]
    training = [
        {
            "epoch": epoch,
            "train_top1": 50.0 + epoch,
            "valid_top1": 48.0 + epoch,
            "train_loss": 2.0 - epoch * 0.1,
            "valid_loss": 2.1 - epoch * 0.1,
            "learning_rate": 0.01,
        }
        for epoch in range(3)
    ]

    search_output = tmp_path / "search-report"
    training_output = tmp_path / "training-report"
    build_report_bundle([], search_output, search=search, bootstrap_samples=10)
    build_report_bundle([], training_output, training=training, bootstrap_samples=10)

    assert (search_output / "search.png").exists()
    assert (search_output / "search.svg").exists()
    assert (training_output / "training.png").exists()
    assert (training_output / "training.svg").exists()


def test_bundle_ignores_an_incomplete_trailing_score_record(tmp_path: Path) -> None:
    source = tmp_path / "scores.jsonl"
    source.write_bytes(
        json.dumps(_score_rows()[0]).encode("utf-8") + b'\n{"architecture_id":"partial"'
    )

    result = build_report_bundle(source, tmp_path / "partial-report", bootstrap_samples=10)

    assert result["row_count"] == 1
    assert (tmp_path / "partial-report" / "index.html").exists()


def test_sensitivity_accepts_mixed_parameter_types(tmp_path: Path) -> None:
    from zcp_test.reporting.analysis import plot_sensitivity

    rows = _score_rows()[:2]
    rows[0]["seed"] = 1
    rows[1]["seed"] = "two"

    figure = plot_sensitivity(rows, tmp_path / "mixed-seed.png", parameter="seed")

    assert figure is not None
    assert (tmp_path / "mixed-seed.png").exists()


def test_analysis_validation_rejects_missing_targets_and_single_sensitivity_value() -> None:
    missing_target = pd.DataFrame(_score_rows()).assign(target_value=None)
    with pytest.raises(ValueError, match="target_value"):
        validate_analysis_scores(missing_target, "correlation")

    single_seed = pd.DataFrame(_score_rows()).assign(seed=1)
    with pytest.raises(ValueError, match="at least two seed"):
        validate_analysis_scores(single_seed, "sensitivity", sensitivity_parameter="seed")


def test_read_scores_preserves_source_run_for_multiple_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(_score_rows()[0]) + "\n", encoding="utf-8")
    second.write_text(json.dumps(_score_rows()[1]) + "\n", encoding="utf-8")

    frame = read_scores([first, second])

    assert set(frame["source_run"]) == {str(first), str(second)}


def test_legacy_search_curve_accepts_candidate_and_summary_records(tmp_path: Path) -> None:
    source = tmp_path / "search.jsonl"
    rows = [
        {"record_kind": "candidate", "generation": 0, "score": 1.0},
        {
            "record_kind": "generation_summary",
            "generation": 0,
            "best_so_far": 1.0,
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert curve_plot(source, tmp_path / "legacy-search.png", "search") == 2
    assert (tmp_path / "legacy-search.png").exists()
