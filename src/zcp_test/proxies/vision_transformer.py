from __future__ import annotations

from typing import Any

from zcp_test.types import ProxyOutput


def dss_indicator(
    model: Any,
    inputs: Any,
    _labels: Any | None = None,
    _loss_fn: Any | None = None,
) -> ProxyOutput:
    """Paper-formula port of the TF-TAS DSS indicator for static ViTs."""
    import torch

    from zcp_test.models.autoformer import (
        AutoFormerAttention,
        AutoFormerMlp,
        StaticAutoFormer,
    )
    from zcp_test.models.pit import PitAttention, PitMlp, PitPooling, StaticPiT

    if not isinstance(model, (StaticAutoFormer, StaticPiT)):
        raise NotImplementedError(
            "dss supports zcp-test StaticAutoFormer and StaticPiT models only"
        )
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
        raise ValueError("dss requires an NCHW image tensor to determine input shape")

    attention_weights: list[torch.nn.Parameter] = []
    mlp_weights: list[torch.nn.Parameter] = []
    auxiliary_weights: list[torch.nn.Parameter] = [
        model.patch_embed.weight,
        model.head.weight,
    ]
    for module in model.modules():
        if isinstance(module, (AutoFormerAttention, PitAttention)):
            attention_weights.extend((module.qkv.weight, module.projection.weight))
        elif isinstance(module, (AutoFormerMlp, PitMlp)):
            mlp_weights.extend(
                child.weight for child in module.modules() if isinstance(child, torch.nn.Linear)
            )
        elif isinstance(module, PitPooling):
            auxiliary_weights.append(module.class_projection.weight)

    if not attention_weights or not mlp_weights:
        raise NotImplementedError("dss found no supported attention/MLP weight groups")

    signs = {name: torch.sign(parameter.detach()) for name, parameter in model.named_parameters()}
    try:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.abs_()
        model.zero_grad(set_to_none=True)
        probe = torch.ones(
            (1, *inputs.shape[1:]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        output = model(probe)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        output.sum().backward()

        attention_diversity = torch.zeros((), device=inputs.device, dtype=torch.float64)
        for weight in attention_weights:
            if weight.grad is not None:
                attention_diversity += (
                    torch.linalg.matrix_norm(weight.grad.double(), ord="nuc")
                    * torch.linalg.matrix_norm(weight.double(), ord="nuc")
                ).abs()

        mlp_saliency = torch.zeros((), device=inputs.device, dtype=torch.float64)
        for weight in mlp_weights:
            if weight.grad is not None:
                mlp_saliency += (weight.grad * weight).abs().double().sum()

        auxiliary_saliency = torch.zeros((), device=inputs.device, dtype=torch.float64)
        for weight in auxiliary_weights:
            if weight.grad is not None:
                auxiliary_saliency += (weight.grad * weight).abs().double().sum()

        components = {
            "attention_diversity": float(attention_diversity.detach()),
            "mlp_saliency": float(mlp_saliency.detach()),
            "auxiliary_saliency": float(auxiliary_saliency.detach()),
        }
        components["score"] = sum(components.values())
        return ProxyOutput(
            score=components["score"],
            primary_component="score",
            components=components,
        )
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.mul_(signs[name])
        model.zero_grad(set_to_none=True)
