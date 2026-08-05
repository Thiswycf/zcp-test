from __future__ import annotations

from typing import Any


def _attention_types() -> tuple[type[Any], type[Any]]:
    from zcp_test.models.autoformer import AutoFormerAttention
    from zcp_test.models.pit import PitAttention

    return AutoFormerAttention, PitAttention


def _attention_maps(module: Any, inputs: Any) -> tuple[Any, Any]:
    import torch

    tensor = inputs[0]
    batch, tokens, _ = tensor.shape
    if hasattr(module, "head_dim"):
        head_dimension = int(module.head_dim)
    else:
        head_dimension = int(module.head_dimension)
    qkv = module.qkv(tensor).reshape(
        batch, tokens, 3, int(module.num_heads), head_dimension
    )
    query, key, _value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    logits = (query @ key.transpose(-2, -1)).mul(float(module.scale))
    relative_key = getattr(module, "relative_key", None)
    if relative_key is not None:
        relative = relative_key(tokens)
        relative_logits = (
            query.permute(2, 0, 1, 3).reshape(
                tokens, int(module.num_heads) * batch, -1
            )
            @ relative.transpose(2, 1)
        )
        logits = logits + relative_logits.transpose(1, 0).reshape(
            batch, int(module.num_heads), tokens, tokens
        ).mul(float(module.scale))
    return logits, torch.softmax(logits, dim=-1)


def _collect_attention(model: Any, inputs: Any) -> tuple[list[Any], list[Any]]:
    logits: list[Any] = []
    probabilities: list[Any] = []
    handles = []

    def hook(module: Any, module_inputs: Any) -> None:
        raw, softmax = _attention_maps(module, module_inputs)
        logits.append(raw)
        probabilities.append(softmax)

    for module in model.modules():
        if isinstance(module, _attention_types()):
            handles.append(module.register_forward_pre_hook(hook))
    if not handles:
        raise NotImplementedError(
            "AC/HI/HC require an explicit StaticAutoFormer or StaticPiT attention probe"
        )
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return logits, probabilities


def _confidence(values: list[Any]) -> dict[str, float]:
    import torch

    if not values:
        raise ValueError("attention probe produced no values")
    layer_scores = [torch.nanmean(torch.amax(value, dim=-1)) for value in values]
    raw = torch.stack(layer_scores).sum()
    normalized = raw / len(layer_scores)
    return {"raw": float(raw.detach()), "normalized": float(normalized.detach())}


def attention_confidence(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    logits, _probabilities = _collect_attention(model, inputs)
    return _confidence(logits)


def head_confidence(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    _logits, probabilities = _collect_attention(model, inputs)
    return _confidence(probabilities)


def head_importance(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    import torch

    attention_types = _attention_types()
    linears = []
    for module in model.modules():
        if isinstance(module, attention_types):
            linears.extend(
                child for child in module.modules() if isinstance(child, torch.nn.Linear)
            )
    if not linears:
        raise NotImplementedError(
            "HI requires an explicit StaticAutoFormer or StaticPiT attention probe"
        )
    model.zero_grad(set_to_none=True)
    output = model(inputs)
    if isinstance(output, (tuple, list)):
        output = output[-1]
    output.sum().backward()
    values = [
        torch.nansum(torch.abs(layer.weight.detach() * layer.weight.grad.detach()))
        for layer in linears
        if layer.weight.grad is not None
    ]
    if not values:
        raise ValueError("attention importance produced no gradients")
    raw = torch.stack(values).sum()
    normalized = raw / len(values)
    return {"raw": float(raw.detach()), "normalized": float(normalized.detach())}
