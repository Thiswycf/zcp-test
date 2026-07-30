from __future__ import annotations

import json

import pandas as pd

from zcp_test.cli import main
from zcp_test.reporting.benchmark_darts import nasbench301_darts_study
from zcp_test.reporting.benchmark_report import write_benchmark_study
from zcp_test.reporting.benchmark_studies import (
    nats_size_study,
    topology_study,
    transnas_transfer_study,
    vit_architecture_study,
)
from zcp_test.spaces.darts import DartsSpace


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
    assert {"correlations", "operation_effects", "matched_pairs", "matched_pair_summary"}.issubset(topology)
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
    assert {"size_controlled_correlations", "size_strata"}.issubset(size)


def test_benchmark_studies_merge_shards_but_keep_evaluation_seeds_separate():
    rows = []
    for seed in (2026, 2027):
        for index in range(4):
            rows.append(
                {
                    "architecture_id": f"s{index}",
                    "architecture": {"architecture": f"8:16:24:32:{40 + 8 * index}"},
                    "proxy_id": "proxy",
                    "component": "score",
                    "score": float(index + seed - 2026),
                    "target_value": float(index * 2),
                    "direction": "maximize",
                    "seed": seed,
                    "run_id": f"shard-{index % 2}",
                    "source_run": f"/runs/shard-{index % 2}",
                }
            )

    tables = nats_size_study(pd.DataFrame(rows))
    correlation = tables["correlations"].query(
        "feature == 'size_last_channel' and outcome == 'target' and method == 'spearman'"
    )

    assert correlation["sample_count"].tolist() == [4, 4]
    assert correlation["seed"].tolist() == [2026, 2027]
    assert "run_id" not in correlation
    assert "source_run" not in correlation


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
    assert len(vit["layers"]) == 6

    transfer_rows = []
    encodings = ("64-41414-1_02_333", "64-41414-2_13_001", "64-41414-3_21_012")
    for task in ("class_object", "segmentsemantic"):
        for index in range(1, 4):
            transfer_rows.append(
                {
                    "search_space_id": "transnas_micro",
                    "dataset": task,
                    "architecture_id": f"a{index}",
                    "architecture": {"architecture": encodings[index - 1]},
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
    assert {"architecture_features", "architecture_factors", "factor_effects"}.issubset(
        transfer
    )
    assert len(transfer["architecture_factors"]) == 33


def test_pit_features_use_stage_embedding_dimensions_and_task_transfer_ignores_target_fields():
    pit_rows = [
        {
            "architecture_id": "pit-a",
            "architecture": {"depth": [1, 1], "base_dim": 16, "num_heads": [2, 4], "mlp_ratio": 4},
            "search_space_id": "pit",
            "proxy_id": "proxy",
            "component": "score",
            "score": 1.0,
            "target_value": 1.0,
            "direction": "maximize",
        }
    ]
    vit = vit_architecture_study(pd.DataFrame(pit_rows))
    assert list(vit["layers"]["dimension"]) == [32.0, 64.0]
    assert list(vit["layers"]["head_dimension"]) == [16.0, 16.0]

    transfer_rows = []
    for task, metric, fingerprint in (
        ("class_object", "valid_top1", "class-batch"),
        ("segmentsemantic", "valid_mIoU", "segmentation-batch"),
    ):
        for index in range(1, 4):
            transfer_rows.append(
                {
                    "search_space_id": "transnas_micro",
                    "dataset": task,
                    "target_metric": metric,
                    "target_split": "valid",
                    "input_fingerprint": fingerprint,
                    "architecture_id": f"a{index}",
                    "proxy_id": "proxy",
                    "component": "score",
                    "score": float(index),
                    "target_value": float(index),
                    "direction": "maximize",
                }
            )
    transfer = transnas_transfer_study(pd.DataFrame(transfer_rows))
    assert len(transfer["task_transfer"]) == 2 * 2 * 2


def test_nasbench301_darts_study_reports_joint_operation_topology_effects(tmp_path):
    space = DartsSpace()
    rows = []
    for index in range(4):
        architecture = space.sample(seed=index)
        rows.append(
            {
                "benchmark_id": "nasbench301_surrogate",
                "search_space_id": "darts",
                "architecture_id": architecture.architecture_id,
                "architecture": architecture.spec,
                "proxy_id": "proxy",
                "component": "score",
                "score": float(index),
                "target_value": float(index * 2),
                "direction": "maximize",
            }
        )

    tables = nasbench301_darts_study(pd.DataFrame(rows))
    assert {"architectures", "edges", "correlations", "operation_topology_interactions"} == set(
        tables
    )
    assert set(tables["edges"]["cell"]) == {"normal", "reduce"}
    assert {"node", "source_class", "operation"}.issubset(
        tables["operation_topology_interactions"]
    )
    report = write_benchmark_study(
        tables, tmp_path / "nb301", view="darts", benchmark_id="nasbench301_surrogate"
    )
    assert "darts_operation_topology_interactions.svg" in report["artifacts"]


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
