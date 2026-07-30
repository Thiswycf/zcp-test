from __future__ import annotations

import json

import pandas as pd

from zcp_test.cli import main
from zcp_test.reporting.benchmark_report import write_benchmark_study
from zcp_test.reporting.benchmark_studies import (
    nats_size_study,
    topology_study,
    transnas_transfer_study,
    vit_architecture_study,
)


def test_topology_and_nats_size_studies_extract_canonical_factors(tmp_path):
    topologies = pd.DataFrame(
        {
            "architecture_id": ["a", "b"],
            "architecture": [
                {"architecture": "|nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|skip_connect~2|"},
                {"architecture": "|skip_connect~0|+|none~0|nor_conv_3x3~1|+|avg_pool_3x3~0|none~1|nor_conv_1x1~2|"},
            ],
        }
    )
    topology = topology_study(topologies)
    assert len(topology["edges"]) == 12
    assert set(topology["edges"]["edge"]) == {"0->1", "0->2", "1->2", "0->3", "1->3", "2->3"}

    sizes = pd.DataFrame(
        {
            "architecture_id": ["a", "b"],
            "architecture": [{"architecture": "8:16:24:32:40"}, {"architecture": "16:16:32:48:64"}],
        }
    )
    size = nats_size_study(sizes)
    assert len(size["stages"]) == 10
    assert size["architectures"]["size_channel_sum"].tolist() == [120, 176]
    report = write_benchmark_study(size, tmp_path / "size", view="size", benchmark_id="nats_sss")
    assert {"stages.csv", "size_stages.png", "index.html"}.issubset(report["artifacts"])


def test_topology_and_size_studies_link_structure_to_scores_and_targets():
    topology_rows = []
    size_rows = []
    for index, operation in enumerate(("none", "avg_pool_3x3", "skip_connect", "nor_conv_3x3"), 1):
        topology_rows.append(
            {
                "architecture_id": f"t{index}",
                "architecture": {
                    "architecture": f"|{operation}~0|+|skip_connect~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|skip_connect~2|"
                },
                "proxy_id": "proxy",
                "component": "score",
                "score": float(index),
                "target_value": float(index * 2),
                "direction": "maximize",
            }
        )
        channels = [8 * index, 16 * index, 24 * index, 32 * index, 40 * index]
        size_rows.append(
            {
                "architecture_id": f"s{index}",
                "architecture": {"architecture": ":".join(map(str, channels))},
                "proxy_id": "proxy",
                "component": "score",
                "score": float(index),
                "target_value": float(index * 2),
                "direction": "maximize",
            }
        )

    topology = topology_study(pd.DataFrame(topology_rows))
    assert {"correlations", "operation_effects"}.issubset(topology)
    assert set(topology["operation_effects"]["edge"]) == {
        "0->1", "0->2", "1->2", "0->3", "1->3", "2->3"
    }
    assert {
        "target_delta_from_edge_mean", "score_delta_from_edge_mean"
    }.issubset(topology["operation_effects"])

    size = nats_size_study(pd.DataFrame(size_rows))
    channel_sum = size["correlations"].query(
        "feature == 'size_channel_sum' and outcome == 'target' and method == 'spearman'"
    )
    assert channel_sum.iloc[0]["correlation"] == 1.0
    assert set(size["stage_sensitivity"]["feature"]) == {
        f"stage_{index}_channel" for index in range(5)
    }


def test_vit_and_transnas_studies_report_feature_and_task_correlations():
    vit_rows = []
    for index, hidden in enumerate((192, 216, 240), 1):
        vit_rows.append(
            {
                "architecture_id": f"a{index}",
                "architecture": {"depth": 2, "hidden_dim": hidden, "num_heads": [3, 4], "mlp_ratio": [3.5, 4.0]},
                "search_space_id": "autoformer",
                "proxy_id": "proxy",
                "component": "score",
                "score": float(index),
                "target_value": float(index * 2),
                "direction": "maximize",
            }
        )
    vit = vit_architecture_study(pd.DataFrame(vit_rows))
    hidden_target = vit["correlations"].query(
        "feature == 'vit_dimension' and outcome == 'target' and method == 'spearman'"
    )
    assert hidden_target.iloc[0]["correlation"] == 1.0

    transfer_rows = []
    for task in ("class_object", "segmentsemantic"):
        for index in range(1, 4):
            transfer_rows.append(
                {
                    "search_space_id": "transnas_micro",
                    "dataset": task,
                    "architecture_id": f"a{index}",
                    "proxy_id": "proxy",
                    "component": "score",
                    "score": float(index),
                    "target_value": float(index),
                    "direction": "maximize",
                }
            )
    transfer = transnas_transfer_study(pd.DataFrame(transfer_rows))
    assert set(transfer["task_quality"]["dataset"]) == {"class_object", "segmentsemantic"}
    assert transfer["task_transfer"]["sample_count"].eq(3).all()


def test_benchmark_analysis_cli_auto_dispatches_topology(tmp_path):
    scores = tmp_path / "scores.jsonl"
    rows = []
    for index, operation in enumerate(("none", "skip_connect"), 1):
        rows.append(
            {
                "benchmark_id": "nasbench201",
                "architecture_id": f"a{index}",
                "architecture": {
                    "architecture": f"|{operation}~0|+|skip_connect~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|skip_connect~2|"
                },
                "proxy_id": "proxy",
                "component": "score",
                "score": float(index),
                "target_value": float(index),
                "direction": "maximize",
                "status": "ok",
            }
        )
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "report"

    main(["analyze", "benchmark", "--scores", str(scores), "--output", str(output)])

    assert (output / "edges.csv").is_file()
    assert (output / "topology_operations.svg").is_file()
    assert json.loads((output / "study.json").read_text())["view"] == "topology"
