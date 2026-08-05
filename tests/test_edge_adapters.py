from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from zcp_test.models.darts import Genotype, NetworkCIFAR
from zcp_test.models.nb101 import Module as Nb101Module
from zcp_test.models.nb101 import Network as Nb101Network
from zcp_test.models.nb201 import InferCell as Nb201Cell
from zcp_test.models.nb201 import TinyNetwork, build_nats_sss
from zcp_test.models.transnas import MicroCell as TransNasMicroCell
from zcp_test.models.transnas import TransNasNetwork
from zcp_test.proxies.edge_adapters import capture_semantic_edge_activations


NB101_SPECIFICATION = {
    "matrix": [
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ],
    "operations": ["input", "conv1x1-bn-relu", "conv3x3-bn-relu", "output"],
}
NB201_SPECIFICATION = (
    "|skip_connect~0|+|nor_conv_1x1~0|skip_connect~1|+"
    "|nor_conv_3x3~0|skip_connect~1|nor_conv_1x1~2|"
)
DARTS_GENE = [
    ("skip_connect", 0),
    ("skip_connect", 1),
    ("sep_conv_3x3", 0),
    ("skip_connect", 2),
    ("skip_connect", 1),
    ("sep_conv_3x3", 2),
    ("skip_connect", 0),
    ("skip_connect", 3),
]


def _nb101_model() -> nn.Module:
    return Nb101Network(NB101_SPECIFICATION, num_classes=3).eval()


def _nb201_model() -> nn.Module:
    return TinyNetwork(
        NB201_SPECIFICATION,
        num_classes=3,
        channels=2,
        cells_per_stage=1,
    ).eval()


def _nats_model() -> nn.Module:
    return build_nats_sss([2, 2, 2, 2, 2], num_classes=3).eval()


def _darts_model() -> nn.Module:
    genotype = Genotype(DARTS_GENE, [2, 3, 4, 5], DARTS_GENE, [2, 3, 4, 5])
    return NetworkCIFAR(2, 3, 2, False, genotype).eval()


def _transnas_model() -> nn.Module:
    return TransNasNetwork("8-4111-3_33_333", num_classes=3).eval()


def _hook_snapshot(model: nn.Module) -> dict[int, tuple[int, ...]]:
    return {
        id(module): tuple(module._forward_hooks)
        for module in model.modules()
    }


@pytest.mark.parametrize(
    ("model_factory", "input_shape", "expected_edges", "cell_type", "edges_per_cell"),
    [
        (_nb101_model, (1, 3, 8, 8), 36, Nb101Module, 4),
        (_nb201_model, (1, 3, 8, 8), 18, Nb201Cell, 6),
        (_nats_model, (1, 3, 8, 8), 18, Nb201Cell, 6),
        (_darts_model, (1, 3, 16, 16), 16, None, 8),
        (_transnas_model, (1, 3, 16, 16), 48, TransNasMicroCell, 6),
    ],
    ids=["nb101", "nb201", "nats-sss", "darts", "transnas-cnn"],
)
def test_cnn_adapters_capture_semantic_unique_4d_edges(
    model_factory: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    expected_edges: int,
    cell_type: type[nn.Module] | None,
    edges_per_cell: int,
) -> None:
    model = model_factory()
    if cell_type is not None:
        assert sum(isinstance(module, cell_type) for module in model.modules()) * edges_per_cell == (
            expected_edges
        )

    with torch.no_grad():
        batch = capture_semantic_edge_activations(model, torch.randn(*input_shape))

    endpoints = [(edge.source, edge.target) for edge in batch]
    assert len(batch) == expected_edges
    assert len(set(endpoints)) == expected_edges
    assert all(edge.activation.ndim == 4 for edge in batch)
    assert all(edge.activation.shape[0] == input_shape[0] for edge in batch)


@pytest.mark.parametrize(
    ("model_factory", "input_shape"),
    [
        (_nb201_model, (1, 3, 8, 8)),
        (_darts_model, (1, 3, 16, 16)),
        (_transnas_model, (1, 3, 16, 16)),
    ],
    ids=["nb201", "darts", "transnas-cnn"],
)
def test_adapter_removes_only_its_forward_hooks(
    model_factory: Callable[[], nn.Module], input_shape: tuple[int, ...]
) -> None:
    model = model_factory()
    semantic_module = next(
        module
        for module in model.modules()
        if isinstance(module, (Nb201Cell, TransNasMicroCell)) or hasattr(module, "_ops")
    )
    hooked_module = next(module for module in semantic_module.modules() if module is not semantic_module)
    persistent_handle = hooked_module.register_forward_hook(lambda *_arguments: None)
    before = _hook_snapshot(model)

    try:
        with torch.no_grad():
            capture_semantic_edge_activations(model, torch.randn(*input_shape))
        assert _hook_snapshot(model) == before
    finally:
        persistent_handle.remove()


def test_adapter_cleans_hooks_and_nb101_temporary_attributes_when_forward_fails() -> None:
    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nb101 = Nb101Module(NB101_SPECIFICATION, 4, 4)
            self.nb201 = Nb201Cell(NB201_SPECIFICATION, 4, 4)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("intentional forward failure")

    model = FailingModel()
    before = _hook_snapshot(model)
    assert not hasattr(model.nb101, "_edge_activation_capture")
    assert not hasattr(model.nb101, "_edge_activation_prefix")

    with pytest.raises(RuntimeError, match="intentional forward failure"):
        capture_semantic_edge_activations(model, torch.randn(1, 4, 8, 8))

    assert _hook_snapshot(model) == before
    assert not hasattr(model.nb101, "_edge_activation_capture")
    assert not hasattr(model.nb101, "_edge_activation_prefix")


def test_transformer_is_explicitly_unsupported_without_running_forward() -> None:
    from zcp_test.models.autoformer import StaticAutoFormer, VITBENCH_AUTOPROX_PROFILE

    model = StaticAutoFormer(
        profile=VITBENCH_AUTOPROX_PROFILE,
        image_size=8,
        patch_size=4,
        num_classes=3,
        embed_dim=8,
        depth=1,
        num_heads=[2],
        mlp_ratio=[2.0],
    )
    before = _hook_snapshot(model)

    with pytest.raises(NotImplementedError, match="semantic edge provider"):
        capture_semantic_edge_activations(model, torch.randn(1, 3, 8, 8))

    assert _hook_snapshot(model) == before
