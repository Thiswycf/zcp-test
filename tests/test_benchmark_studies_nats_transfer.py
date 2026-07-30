from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zcp_test.reporting.benchmark_studies import nats_size_study


def _nats_rows() -> pd.DataFrame:
    rows = []
    targets = {
        "cifar10": [1.0, 2.0, 2.0, 4.0, 5.0],
        "cifar100": [5.0, 4.0, 3.0, 2.0, 1.0],
        "ImageNet16-120": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    proxy_scores = {
        "proxy_a": {
            "cifar10": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cifar100": [5.0, 4.0, 3.0, 2.0, 1.0],
            "ImageNet16-120": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        "proxy_b": {
            "cifar10": [1.0, 1.0, 2.0, 2.0, 3.0],
            "cifar100": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ImageNet16-120": [3.0, 2.0, 2.0, 1.0, 1.0],
        },
    }
    for proxy_id, datasets in proxy_scores.items():
        for dataset, scores in datasets.items():
            for index, (score, target) in enumerate(zip(scores, targets[dataset], strict=True)):
                channels = [8 + index, 16 + 2 * index, 24 + index, 32, 40 + 3 * index]
                rows.append(
                    {
                        "benchmark_id": "nats_sss",
                        "search_space_id": "nats_size",
                        "dataset": dataset,
                        "architecture_id": f"a{index}",
                        "architecture": {"architecture": ":".join(map(str, channels))},
                        "proxy_id": proxy_id,
                        "component": "score",
                        "score": score,
                        "direction": "maximize",
                        "input_source": "dataset",
                        "input_fingerprint": f"batch-{dataset}",
                        "target_value": target,
                        "target_metric": "final_accuracy",
                        "target_split": "test",
                        "target_direction": "maximize",
                        "target_epoch_budget": 90,
                    }
                )
    return pd.DataFrame(rows)


def test_nats_cross_dataset_tables_separate_proxy_and_target_semantics():
    tables = nats_size_study(_nats_rows())

    matrix = tables["dataset_proxy_target_matrix"]
    assert len(matrix) == 2 * 3 * 3 * 3
    direct = matrix.query(
        "proxy_id == 'proxy_a' and source_dataset == 'cifar10' "
        "and target_dataset == 'ImageNet16-120' and method == 'spearman'"
    ).iloc[0]
    assert direct["correlation"] == pytest.approx(1.0)
    assert direct["source_input_fingerprint"] == "batch-cifar10"

    stability = tables["proxy_dataset_stability"]
    assert len(stability) == 2 * 3 * 3
    reversed_rank = stability.query(
        "proxy_id == 'proxy_a' and dataset_left == 'cifar10' "
        "and dataset_right == 'cifar100' and method == 'spearman'"
    ).iloc[0]
    assert reversed_rank["correlation"] == pytest.approx(-1.0)
    assert reversed_rank["input_fingerprint_left"] != reversed_rank["input_fingerprint_right"]

    target_transfer = tables["target_dataset_transfer"]
    assert len(target_transfer) == 3 * 3
    assert "proxy_id" not in target_transfer
    target_rank = target_transfer.query(
        "dataset_left == 'cifar10' and dataset_right == 'cifar100' "
        "and method == 'kendall_tau_b'"
    ).iloc[0]
    assert target_rank["sample_count"] == 5
    assert target_rank["correlation"] < 0

    controlled = tables["controlled_proxy_target_transfer"]
    assert set(controlled["control"]) == {
        "size_channel_sum",
        "stage_0_channel",
        "stage_1_channel",
        "stage_2_channel",
        "stage_3_channel",
        "stage_4_channel",
    }
    assert controlled["method"].eq("partial_spearman").all()


def test_nats_cross_dataset_filters_nonfinite_values_and_preserves_ties():
    frame = _nats_rows()
    frame.loc[
        (frame["proxy_id"] == "proxy_a")
        & (frame["dataset"] == "cifar10")
        & (frame["architecture_id"] == "a4"),
        "score",
    ] = np.nan
    frame.loc[
        (frame["proxy_id"] == "proxy_a")
        & (frame["dataset"] == "cifar100")
        & (frame["architecture_id"] == "a3"),
        "score",
    ] = np.inf
    tables = nats_size_study(frame)
    row = tables["proxy_dataset_stability"].query(
        "proxy_id == 'proxy_a' and dataset_left == 'cifar10' "
        "and dataset_right == 'cifar100' and method == 'kendall_tau_b'"
    ).iloc[0]
    assert row["common_architecture_count"] == 5
    assert row["valid_pair_count"] == 3
    assert row["sample_count"] == 3
    assert row["valid_coverage"] == pytest.approx(3 / 5)

    frame.loc[frame["proxy_id"] == "proxy_b", "score"] = 1.0
    constant = nats_size_study(frame)["proxy_dataset_stability"]
    values = constant.query("proxy_id == 'proxy_b'")["correlation"]
    assert values.isna().all()


def test_nats_cross_dataset_rejects_duplicate_scores_and_conflicting_targets():
    frame = _nats_rows()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per protocol/proxy/component/architecture"):
        nats_size_study(duplicate)

    conflicting = frame.copy()
    selector = (
        (conflicting["proxy_id"] == "proxy_b")
        & (conflicting["dataset"] == "cifar10")
        & (conflicting["architecture_id"] == "a0")
    )
    conflicting.loc[selector, "target_value"] = 99.0
    with pytest.raises(ValueError, match="conflicting target values"):
        nats_size_study(conflicting)


@pytest.mark.parametrize(
    "field", ["target_metric", "target_split", "target_direction", "input_fingerprint"]
)
def test_nats_cross_dataset_requires_complete_protocol(field: str):
    frame = _nats_rows()
    frame.loc[0, field] = None
    with pytest.raises(ValueError, match=field):
        nats_size_study(frame)
