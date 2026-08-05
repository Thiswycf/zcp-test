from __future__ import annotations

import copy
import math

import pytest
import torch

from zcp_test.models.autoformer import (
    AZNAS_SCRATCH_PROFILE,
    AutoFormerAttention,
    AutoFormerMlp,
    StaticAutoFormer,
)
from zcp_test.models.pit import PitAttention, PitMlp, PitPooling, StaticPiT
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.evaluator import evaluate_proxy


def _manual_dss(model: torch.nn.Module, inputs: torch.Tensor) -> dict[str, float]:
    model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.abs_()
    model.zero_grad(set_to_none=True)
    model(torch.ones((1, *inputs.shape[1:]), dtype=inputs.dtype)).sum().backward()

    attention = 0.0
    saliency = 0.0
    auxiliary = float((model.patch_embed.weight.grad * model.patch_embed.weight).abs().sum())
    auxiliary += float((model.head.weight.grad * model.head.weight).abs().sum())
    for module in model.modules():
        if isinstance(module, (AutoFormerAttention, PitAttention)):
            for weight in (module.qkv.weight, module.projection.weight):
                attention += float(
                    torch.linalg.matrix_norm(weight.grad.double(), ord="nuc")
                    * torch.linalg.matrix_norm(weight.double(), ord="nuc")
                )
        elif isinstance(module, (AutoFormerMlp, PitMlp)):
            for child in module.modules():
                if isinstance(child, torch.nn.Linear):
                    saliency += float((child.weight.grad * child.weight).abs().sum())
        elif isinstance(module, PitPooling):
            weight = module.class_projection.weight
            auxiliary += float((weight.grad * weight).abs().sum())
    return {
        "score": attention + saliency + auxiliary,
        "attention_diversity": attention,
        "mlp_saliency": saliency,
        "auxiliary_saliency": auxiliary,
    }


@pytest.mark.parametrize("model_kind", ["autoformer", "pit"])
def test_dss_matches_tf_tas_paper_formula_and_restores_model(model_kind: str) -> None:
    torch.manual_seed(41)
    if model_kind == "autoformer":
        model = StaticAutoFormer(
            profile=AZNAS_SCRATCH_PROFILE,
            image_size=32,
            patch_size=16,
            num_classes=3,
            embed_dim=24,
            depth=2,
            num_heads=[2, 2],
            mlp_ratio=[2.0, 2.0],
            super_depth=14,
        )
    else:
        model = StaticPiT(
            image_size=32,
            patch_size=8,
            patch_stride=4,
            num_classes=3,
            base_dim=8,
            depth=[1, 1, 1],
            num_heads=[1, 2, 4],
            mlp_ratio=2.0,
            drop_path_rate=0.0,
        )
    inputs = torch.randn(2, 3, 32, 32)
    expected = _manual_dss(model, inputs)
    state = copy.deepcopy(model.state_dict())

    result = evaluate_proxy("dss", model, inputs, model_family="transformer")

    assert result.status.value == "ok"
    assert result.proxy_version == "tf-tas-42616bc-code-protocol-port-v2"
    assert result.implementation_fidelity == "paper_formula_port_stabilized"
    assert result.components == pytest.approx(expected, rel=1e-6)
    assert result.score == pytest.approx(expected["score"], rel=1e-6)
    assert math.isfinite(result.score)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])


def test_dss_contract_is_transformer_only_and_source_pinned() -> None:
    load_builtin_proxies()
    capability = PROXIES.create("dss").capability
    assert capability.model_families == ("transformer",)
    assert capability.requires_data is False
    assert capability.components == (
        "score",
        "attention_diversity",
        "mlp_saliency",
        "auxiliary_saliency",
    )
    assert capability.source is not None and "42616bc" in capability.source

    result = evaluate_proxy(
        "dss",
        torch.nn.Linear(4, 2),
        torch.ones(1, 4),
        model_family="cnn",
    )
    assert result.status.value == "unsupported"
