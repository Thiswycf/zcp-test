from __future__ import annotations

import math

import pytest

from zcp_test.acceptance import plan_adaptive_feasibility_sweep
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.edge_adapters import capture_semantic_edge_activations
from zcp_test.proxies.edge_rank import EdgeActivation, EdgeActivationBatch
from zcp_test.proxies.evaluator import evaluate_proxy
from zcp_test.proxies.scalarizers import aggregate_rank, select_pointwise
from zcp_test.types import ProxyContext


def test_registry_contains_only_current_audited_ids() -> None:
    load_builtin_proxies()
    assert len(PROXIES.names()) == 23
    for proxy_id in ("ntkt", "er_pr", "er_conn", "er_deg", "er_dist"):
        with pytest.raises(KeyError, match="Retired proxy"):
            PROXIES.create(proxy_id)
    assert {"ac", "hi", "hc", "er", "ter"}.issubset(PROXIES.names())
    assert "dss_pp" not in PROXIES.names()


def test_er_and_ter_require_explicit_edge_context() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Identity()
    activation = torch.randn(2, 3, 4, 4)
    batch = EdgeActivationBatch([EdgeActivation("source", "sink", activation)])
    missing = evaluate_proxy("er", model, model_family="cnn")
    assert missing.status.value == "unsupported"
    er = evaluate_proxy(
        "er",
        model,
        context=ProxyContext(
            model_family="cnn", edge_activations=batch, inputs=None
        ),
    )
    ter = evaluate_proxy(
        "ter",
        model,
        context=ProxyContext(
            model_family="cnn", edge_activations=batch, inputs=None
        ),
    )
    assert er.status.value == ter.status.value == "ok"
    assert er.primary_component == ter.primary_component == "score"
    assert set(er.components) == set(ter.components) == {"score"}


def test_nb201_semantic_edge_adapter_has_six_unique_edges() -> None:
    torch = pytest.importorskip("torch")
    from zcp_test.models.nb201 import InferCell

    specification = "|skip_connect~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_3x3~0|skip_connect~1|nor_conv_1x1~2|"
    cell = InferCell(specification, 4, 4)
    batch = capture_semantic_edge_activations(cell, torch.randn(2, 4, 8, 8))
    assert len(batch) == 6
    assert len({(edge.source, edge.target) for edge in batch}) == 6
    assert all(edge.activation.ndim == 4 for edge in batch)


def test_transformer_attention_proxies_expose_raw_and_normalized() -> None:
    torch = pytest.importorskip("torch")
    from zcp_test.models.pit import PitAttention

    class ToyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = PitAttention(8, 2, 0.0, 0.0)
            self.head = torch.nn.Linear(8, 3)

        def forward(self, inputs):
            return self.head(self.attention(inputs).mean(1))

    inputs = torch.randn(2, 5, 8)
    for proxy_id in ("ac", "hi", "hc"):
        result = evaluate_proxy(
            proxy_id, ToyAttention(), inputs, model_family="transformer"
        )
        assert result.status.value == "ok", result.error_message
        assert result.primary_component == "raw"
        assert set(result.components) == {"raw", "normalized"}
        assert all(math.isfinite(value) for value in result.components.values())


def test_score_selectors_distinguish_pointwise_and_cohort_scores() -> None:
    row = {"score": 3.0, "components": {"a": 1.0, "b": 4.0}}
    assert select_pointwise(row, "primary") == 3.0
    assert select_pointwise(row, "component:b") == 4.0
    result = aggregate_rank(
        [{"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 2.0}],
        ("a", "b"),
        method="az_nas_log_rank",
        architecture_ids=("left", "right"),
    )
    assert result.cohort_size == 2
    assert len(result.cohort_digest or "") == 64
    assert result.tie_method == "average"


def test_te_nas_is_single_scalar_only() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(2, 2),
    )
    result = evaluate_proxy("te_nas", model, torch.randn(2, 1, 4, 4))
    assert result.status.value == "ok", result.error_message
    assert result.primary_component == "score"
    assert set(result.components) == {"score"}


@pytest.mark.parametrize(
    ("seconds_per_architecture", "label"),
    [(0.1, "one_percent"), (2.0, "one_per_mille"), (10.0, "one_per_ten_thousand")],
)
def test_adaptive_feasibility_ladder(seconds_per_architecture: float, label: str) -> None:
    plan = plan_adaptive_feasibility_sweep(
        total_architectures=100_000,
        pilot_architectures=1,
        pilot_seconds=seconds_per_architecture,
    )
    assert plan["fraction_label"] == label
    assert plan["projected_seconds"] <= 600
    assert plan["coverage_claim"] is False


def test_adaptive_feasibility_times_out_single_architecture() -> None:
    plan = plan_adaptive_feasibility_sweep(
        total_architectures=100,
        pilot_architectures=1,
        pilot_seconds=601,
    )
    assert plan["status"] == "timeout_feasibility"
    assert plan["architecture_count"] == 1
