from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from zcp_test.proxies.official import (
    OFFICIAL_IMPLEMENTATIONS,
    effective_rank,
    gradnorm,
    jacob_cov,
    naswot,
    near,
    official_proxy_factories,
    swap,
    synflow,
    zen,
    zico,
)


def test_gradnorm_golden_sums_layer_weight_norms_only() -> None:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.bias.fill_(3.0)
    inputs = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    labels = torch.tensor([[0.5], [-2.0]])

    reference = copy.deepcopy(model)
    nn.MSELoss()(reference(inputs), labels).backward()
    expected = float(reference.weight.grad.norm())

    assert gradnorm(model, inputs, labels, nn.MSELoss()) == pytest.approx(expected)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_synflow_golden_and_model_state_is_restored() -> None:
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, 1, bias=False),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
        model[2].weight.copy_(torch.tensor([[5.0, 6.0]]))
    original = copy.deepcopy(model.state_dict())

    assert synflow(model, torch.zeros(4, 2)) == pytest.approx(114.0)
    assert next(model.parameters()).dtype == torch.float32
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_naswot_golden_and_depth_ordering_invariant() -> None:
    inputs = torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]])
    shallow = nn.Sequential(nn.ReLU())
    deep = nn.Sequential(nn.ReLU(), nn.ReLU())
    expected_kernel = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 2.0]])
    expected = np.linalg.slogdet(expected_kernel)[1]

    assert naswot(shallow, inputs) == pytest.approx(expected)
    assert naswot(deep, inputs) > naswot(shallow, inputs)


class _Quadratic(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs.square()).sum(dim=1, keepdim=True)


def test_jacob_cov_golden_matches_upstream_numpy_formula() -> None:
    inputs = torch.tensor([[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0]])
    jacobian = (2.0 * inputs).numpy()
    eigenvalues = np.linalg.eigvals(np.corrcoef(jacobian))
    expected = -np.sum(np.log(eigenvalues + 1e-5) + 1.0 / (eigenvalues + 1e-5))

    assert jacob_cov(_Quadratic(), inputs) == pytest.approx(float(np.real_if_close(expected)))
    assert inputs.requires_grad is False


def test_near_effective_rank_golden_and_layer_sum() -> None:
    matrix = torch.diag(torch.tensor([3.0, 1.0]))
    probabilities = np.array([0.75, 0.25])
    expected = math.exp(float(-(probabilities * np.log(probabilities)).sum()))
    assert effective_rank(matrix) == pytest.approx(expected)

    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    assert near(model, torch.eye(2)) == pytest.approx(2.0)


def test_swap_golden_counts_neurons_not_samples() -> None:
    inputs = torch.tensor(
        [
            [[[1.0, -1.0]]],
            [[[1.0, 1.0]]],
            [[[-1.0, -1.0]]],
        ]
    )
    assert swap(nn.Sequential(nn.ReLU()), inputs) == 2.0


class _ZenNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)

    def forward_pre_GAP(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_pre_GAP(inputs).mean(dim=(2, 3))


def _reference_zen_once(model: _ZenNetwork, inputs: torch.Tensor) -> float:
    with torch.no_grad():
        nn.init.normal_(model.conv.weight)
        nn.init.zeros_(model.conv.bias)
        nn.init.ones_(model.bn.weight)
        nn.init.zeros_(model.bn.bias)
        first_input = torch.randn_like(inputs)
        second_input = torch.randn_like(inputs)
        first = model.forward_pre_GAP(first_input)
        mixed = model.forward_pre_GAP(first_input + 1e-2 * second_input)
        difference = (first - mixed).abs().sum(dim=(1, 2, 3)).mean()
        return float(torch.log(difference) + torch.log(torch.sqrt(model.bn.running_var.mean())))


def test_zen_golden_matches_fixed_source_protocol_and_restores_model() -> None:
    model = _ZenNetwork()
    reference = copy.deepcopy(model)
    original = copy.deepcopy(model.state_dict())
    inputs = torch.empty(3, 1, 2, 2)

    torch.manual_seed(17)
    expected = _reference_zen_once(reference, inputs)
    torch.manual_seed(17)
    assert zen(model, inputs) == pytest.approx(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_zico_golden_uses_gradients_across_batches() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
    batches = [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[2.0, -1.0]]), torch.tensor([[1.0]])),
        (torch.tensor([[-1.0, 3.0]]), torch.tensor([[2.0]])),
    ]

    reference = copy.deepcopy(model)
    gradients = []
    for batch_inputs, batch_labels in batches:
        reference.zero_grad()
        nn.MSELoss()(reference(batch_inputs), batch_labels).backward()
        gradients.append(reference.weight.grad.reshape(-1).numpy().copy())
    values = np.asarray(gradients)
    standard_deviation = values.std(axis=0)
    nonzero = np.nonzero(standard_deviation)[0]
    expected = np.log(
        np.sum(np.mean(np.abs(values), axis=0)[nonzero] / standard_deviation[nonzero])
    )

    assert zico(model, batches, loss_fn=nn.MSELoss()) == pytest.approx(expected)


def test_official_factories_are_complete_and_side_effect_free() -> None:
    expected = {"gradnorm", "synflow", "naswot", "jacob_cov", "near", "swap", "zen", "zico"}
    factories = official_proxy_factories()

    assert set(OFFICIAL_IMPLEMENTATIONS) == expected
    assert set(factories) == expected
    for name, factory in factories.items():
        proxy = factory()
        assert proxy.capability.proxy_id == name
        assert proxy.capability.implementation_fidelity == "fixed_source_formula_port"
