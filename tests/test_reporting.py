from __future__ import annotations

import csv
import json

import matplotlib
import pandas as pd
import pytest

from zcp_test.reporting.benchmark_report import write_benchmark_study
from zcp_test.reporting.reports import curve_plot, jsonl_to_csv, static_html


matplotlib.use("Agg")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_jsonl_to_csv_handles_empty_and_nested_records(tmp_path):
    empty_source = tmp_path / "empty.jsonl"
    empty_source.write_text("", encoding="utf-8")
    empty_destination = tmp_path / "empty" / "records.csv"

    assert jsonl_to_csv(empty_source, empty_destination) == 0
    assert empty_destination.read_text(encoding="utf-8") in {"\n", "\r\n"}

    source = tmp_path / "records.jsonl"
    destination = tmp_path / "nested" / "records.csv"
    _write_jsonl(
        source,
        [
            {"name": "第一条", "metadata": {"depth": 2}, "values": [1, "二"]},
            {"extra": True, "name": "second"},
        ],
    )

    assert jsonl_to_csv(source, destination) == 2
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["extra", "metadata", "name", "values"]
    assert json.loads(rows[0]["metadata"]) == {"depth": 2}
    assert json.loads(rows[0]["values"]) == [1, "二"]
    assert rows[1] == {"extra": "True", "metadata": "", "name": "second", "values": ""}


def test_static_html_escapes_title_headers_and_nested_values(tmp_path):
    source = tmp_path / "records.jsonl"
    destination = tmp_path / "site" / "index.html"
    _write_jsonl(
        source,
        [{"<script>key</script>": {"markup": "<img src=x onerror=alert(1)>"}}],
    )

    assert static_html(source, destination, title="Report <unsafe> & results") == 1
    document = destination.read_text(encoding="utf-8")
    assert "<h1>Report &lt;unsafe&gt; &amp; results</h1>" in document
    assert "&lt;script&gt;key&lt;/script&gt;" in document
    assert "&lt;img src=x onerror=alert(1)&gt;" in document
    assert "<script>key</script>" not in document
    assert "<img src=x onerror=alert(1)>" not in document


def test_curve_plot_supports_training_and_raw_search_records(tmp_path):
    training = tmp_path / "training.jsonl"
    search = tmp_path / "search.jsonl"
    _write_jsonl(
        training,
        [
            {"epoch": 1, "train_top1": 60, "valid_loss": 1.2, "learning_rate": 0.01},
            {"epoch": 0, "train_top1": 50, "valid_loss": 1.5, "learning_rate": 0.02},
        ],
    )
    _write_jsonl(
        search,
        [
            {"generation": 1, "score": 2.0},
            {"generation": 0, "score": 1.0},
            {"generation": 1, "score": 3.0},
        ],
    )

    assert curve_plot(training, tmp_path / "curves" / "training.png", "training") == 2
    assert curve_plot(search, tmp_path / "curves" / "search.png", "search") == 3
    assert (tmp_path / "curves" / "training.png").is_file()
    assert (tmp_path / "curves" / "search.png").is_file()


def test_curve_plot_rejects_empty_unknown_and_invalid_curve_data(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    record = tmp_path / "record.jsonl"
    _write_jsonl(record, [{"epoch": 0}])

    with pytest.raises(ValueError, match="empty JSONL"):
        curve_plot(empty, tmp_path / "empty.png", "training")
    with pytest.raises(ValueError, match="Unknown plot kind: mystery"):
        curve_plot(record, tmp_path / "unknown.png", "mystery")
    with pytest.raises(ValueError, match="no supported metrics"):
        curve_plot(record, tmp_path / "training.png", "training")
    with pytest.raises(ValueError, match="generation and score"):
        curve_plot(record, tmp_path / "search.png", "search")


def test_benchmark_report_budget_view_uses_both_tables(tmp_path):
    tables = {
        "correlations": pd.DataFrame(
            {
                "proxy_id": ["p", "p"],
                "component": ["score", "score"],
                "epoch_budget": [1, 2],
                "spearman": [0.2, 0.6],
            }
        ),
        "top_k_retrieval": pd.DataFrame(
            {
                "proxy_id": ["p", "p"],
                "component": ["score", "score"],
                "epoch_budget": [1, 2],
                "requested_k": [5, 5],
                "precision_at_k": [0.4, 0.8],
            }
        ),
    }

    report = write_benchmark_study(
        tables,
        tmp_path / "budget",
        view="budget",
        benchmark_id="nasbench101",
    )

    assert report["tables"] == {"correlations": 2, "top_k_retrieval": 2}
    assert {
        "budget_correlation.png",
        "budget_correlation.svg",
        "budget_top_k_retrieval.png",
        "budget_top_k_retrieval.svg",
    }.issubset(report["artifacts"])


@pytest.mark.parametrize(
    ("view", "tables", "expected_artifacts"),
    [
        (
            "topology",
            {
                "operations": pd.DataFrame(
                    {"operation": ["skip_connect"], "edge_fraction": [1.0]}
                ),
                "correlations": pd.DataFrame(
                    {
                        "feature": ["skip_fraction"],
                        "correlation": [0.5],
                        "method": ["spearman"],
                        "proxy_id": ["p"],
                    }
                ),
            },
            {"topology_operations.png", "topology_feature_correlations.png"},
        ),
        (
            "size",
            {
                "stages": pd.DataFrame({"stage": [0, 1], "channel": [16, 32]}),
                "correlations": pd.DataFrame(
                    {"feature": ["channel_sum"], "correlation": [0.4]}
                ),
            },
            {"size_stages.png", "size_feature_correlations.png"},
        ),
        (
            "architecture",
            {
                "correlations": pd.DataFrame(
                    {"feature": ["depth"], "correlation": [0.3]}
                )
            },
            {"architecture_features.png"},
        ),
        (
            "darts",
            {
                "correlations": pd.DataFrame(
                    {"feature": ["normal_edges"], "correlation": [0.2]}
                ),
                "operation_topology_interactions": pd.DataFrame(
                    {
                        "cell": ["normal"],
                        "node": [2],
                        "operation": ["skip_connect"],
                        "target_delta_from_interaction_mean": [0.0],
                        "proxy_id": ["p"],
                        "component": ["score"],
                    }
                ),
            },
            {"darts_feature_correlations.png", "darts_operation_topology_interactions.png"},
        ),
        (
            "transfer",
            {
                "task_quality": pd.DataFrame(
                    {
                        "dataset": ["task-a"],
                        "correlation": [0.7],
                        "method": ["spearman"],
                        "proxy_id": ["p"],
                    }
                ),
                "task_transfer": pd.DataFrame(
                    {
                        "search_space_id": ["space", "space"],
                        "proxy_id": ["p", "p"],
                        "component": ["score", "score"],
                        "source_task": ["task-a", "task-b"],
                        "target_task": ["task-b", "task-a"],
                        "correlation": [0.5, 0.6],
                        "method": ["spearman", "spearman"],
                    }
                ),
            },
            {"task_quality.png", "task_transfer_0.png"},
        ),
    ],
)
def test_benchmark_report_views_accept_minimal_valid_tables(
    tmp_path, view, tables, expected_artifacts
):
    report = write_benchmark_study(
        tables,
        tmp_path / view,
        view=view,
        benchmark_id="benchmark <unsafe>",
    )

    assert expected_artifacts.issubset(report["artifacts"])
    assert {"study.json", "index.html"}.issubset(report["artifacts"])
    document = (tmp_path / view / "index.html").read_text(encoding="utf-8")
    assert "benchmark &lt;unsafe&gt;" in document
    assert "benchmark <unsafe>" not in document


def test_benchmark_report_skips_plots_for_empty_or_incomplete_tables(tmp_path):
    report = write_benchmark_study(
        {"correlations": pd.DataFrame({"epoch_budget": [1]})},
        tmp_path / "incomplete",
        view="budget",
        benchmark_id="minimal",
    )
    unknown = write_benchmark_study(
        {},
        tmp_path / "unknown",
        view="custom",
        benchmark_id="minimal",
    )

    assert report["artifacts"] == ["correlations.csv", "study.json", "index.html"]
    assert unknown["artifacts"] == ["study.json", "index.html"]
