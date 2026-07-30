from __future__ import annotations

from pathlib import Path

from zcp_test.cli import main
from zcp_test.reporting.analysis import read_scores
from zcp_test.reporting.benchmark_darts import nasbench301_darts_study
from zcp_test.reporting.benchmark_studies import (
    nats_size_study,
    topology_study,
    transnas_transfer_study,
    vit_architecture_study,
)
from zcp_test.reporting.proxy_studies import proxy_study
from zcp_test.spaces.nb101 import Nb101Space


EXAMPLES = Path(__file__).parents[1] / "examples" / "studies" / "data"


def test_generic_multi_proxy_example_is_protocol_separated(tmp_path: Path) -> None:
    source = EXAMPLES / "generic_multi_proxy.jsonl"
    study = proxy_study(source, k=(1, 3))
    assert len(study["proxy_target_long"].query("method == 'spearman'")) == 6
    assert len(study["proxy_proxy_correlations"]) == 6

    main(
        [
            "analyze",
            "compare",
            "--scores",
            str(source),
            "--bootstrap-samples",
            "10",
            "--top-k",
            "1",
            "3",
            "--output",
            str(tmp_path / "generic"),
        ]
    )
    assert (tmp_path / "generic" / "proxy_target_protocol_heatmap.svg").is_file()


def test_all_benchmark_examples_match_their_custom_studies() -> None:
    topology = topology_study(read_scores(EXAMPLES / "nats_tss_topology.jsonl"))
    size = nats_size_study(read_scores(EXAMPLES / "nats_sss_size.jsonl"))
    darts = nasbench301_darts_study(read_scores(EXAMPLES / "nasbench301_darts.jsonl"))
    transnas = transnas_transfer_study(read_scores(EXAMPLES / "transnas_tasks.jsonl"))
    vit = vit_architecture_study(read_scores(EXAMPLES / "vit_autoformer.jsonl"))

    assert not topology["matched_pairs"].empty
    assert not size["size_controlled_correlations"].empty
    assert not darts["operation_topology_interactions"].empty
    assert not transnas["task_transfer"].empty
    assert not vit["layers"].empty


def test_nb101_example_contains_official_canonical_specs() -> None:
    space = Nb101Space()
    frame = read_scores(EXAMPLES / "nasbench101_budget.jsonl")
    unique = frame.drop_duplicates("architecture_id")
    for _, row in unique.iterrows():
        architecture = space.canonicalize(row["architecture"])
        assert architecture.architecture_id == row["architecture_id"]
