from __future__ import annotations

import math

import pytest
import torch

from zcp_test.proxies.edge_rank import (
    EdgeActivation,
    EdgeActivationBatch,
    compute_er_score,
    compute_ter_score,
    edge_pagerank,
)


def _rank_one_activation() -> torch.Tensor:
    return torch.tensor([[[[-1.0, -1.0, 1.0, 1.0]], [[-1.0, -1.0, 1.0, 1.0]]]])


def _rank_two_activation() -> torch.Tensor:
    return torch.tensor([[[[-1.0, -1.0, 1.0, 1.0]], [[-1.0, 1.0, -1.0, 1.0]]]])


def test_er_matches_hand_computed_effective_rank_sum() -> None:
    assert compute_er_score([_rank_one_activation(), _rank_two_activation()]) == pytest.approx(3.0)


def test_er_is_invariant_to_per_channel_shift_and_nonzero_scale() -> None:
    activation = _rank_two_activation()
    shift = torch.tensor([2.0, -7.0]).view(1, 2, 1, 1)
    scale = torch.tensor([-3.0, 5.0]).view(1, 2, 1, 1)

    assert compute_er_score([activation * scale + shift]) == pytest.approx(
        compute_er_score([activation])
    )


def test_er_adds_edges_and_degenerate_edge_contributes_zero() -> None:
    rank_two = compute_er_score([_rank_two_activation()])
    assert compute_er_score([_rank_two_activation(), _rank_two_activation()]) == pytest.approx(
        2 * rank_two
    )
    assert compute_er_score([torch.ones(1, 2, 1, 4)]) == 0.0


def test_edge_pagerank_matches_first_party_chain_iteration() -> None:
    ranks = edge_pagerank({"a": ["b"], "b": ["c"]})

    assert ranks[("a", "b")] == pytest.approx(0.075)
    assert ranks[("b", "c")] == pytest.approx(0.13875)


def test_ter_matches_hand_computed_bidirectional_chain_weighting() -> None:
    batch = EdgeActivationBatch(
        [
            EdgeActivation("a", "b", _rank_one_activation()),
            EdgeActivation("b", "c", _rank_two_activation()),
        ]
    )

    assert compute_ter_score(batch) == pytest.approx(1.5)


def test_ter_is_invariant_to_edge_order_and_endpoint_relabeling() -> None:
    first = EdgeActivation("a", "b", _rank_one_activation())
    second = EdgeActivation("b", "c", _rank_two_activation())
    relabeled = EdgeActivationBatch(
        [
            EdgeActivation(20, 30, _rank_two_activation()),
            EdgeActivation(10, 20, _rank_one_activation()),
        ]
    )

    assert compute_ter_score([second, first]) == pytest.approx(compute_ter_score([first, second]))
    assert compute_ter_score(relabeled) == pytest.approx(compute_ter_score([first, second]))


@pytest.mark.parametrize(
    "activation, error",
    [
        (torch.ones(2, 3), ValueError),
        (torch.ones(1, 1, 1, 1), ValueError),
        (torch.ones(1, 2, 1, 2, dtype=torch.int64), TypeError),
        (torch.tensor([[[[math.inf, 0.0]]]]), ValueError),
        ("not a tensor", TypeError),
    ],
)
def test_er_rejects_invalid_activations(activation: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        compute_er_score([activation])


def test_er_and_ter_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_er_score([])
    with pytest.raises(ValueError, match="at least one"):
        compute_ter_score(EdgeActivationBatch([]))


def test_edge_data_structures_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="hashable"):
        EdgeActivation([], "b", _rank_one_activation())
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        EdgeActivation("a", "b", torch.ones(2, 3))
    with pytest.raises(TypeError, match="EdgeActivation"):
        EdgeActivationBatch([object()])


def test_ter_rejects_duplicate_directed_edges() -> None:
    edges = [
        EdgeActivation("a", "b", _rank_one_activation()),
        EdgeActivation("a", "b", _rank_two_activation()),
    ]

    with pytest.raises(ValueError, match="unique"):
        compute_ter_score(edges)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"damping": -0.1},
        {"damping": 1.0},
        {"max_iter": 0},
        {"max_iter": True},
        {"tolerance": 0.0},
        {"tolerance": math.inf},
    ],
)
def test_edge_pagerank_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        edge_pagerank({"a": ["b"]}, **kwargs)
