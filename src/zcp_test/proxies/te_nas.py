from __future__ import annotations

from typing import Any


def _linear_regions(model: Any, inputs: Any) -> float:
    import torch

    activations: list[Any] = []
    handles = []

    def hook(_module: Any, _module_inputs: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            activations.append(output.detach().flatten(1))

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            handles.append(module.register_forward_hook(hook))
    if not handles:
        raise NotImplementedError("TE-NAS RN requires ReLU activations")
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    patterns = torch.sign(torch.cat(activations, dim=1)).half()
    differences = patterns @ (1 - patterns).T
    differences = differences + differences.T
    matches = (1 - torch.sign(differences)).sum(dim=1).float()
    return float((1.0 / matches.clamp_min(1.0)).sum())


def _ntk_condition(model: Any, inputs: Any) -> float:
    import torch

    gradients = []
    model.zero_grad(set_to_none=True)
    output = model(inputs)
    if isinstance(output, (tuple, list)):
        output = output[-1]
    for index in range(output.shape[0]):
        model.zero_grad(set_to_none=True)
        output[index : index + 1].backward(
            torch.ones_like(output[index : index + 1]), retain_graph=True
        )
        values = [
            parameter.grad.detach().flatten()
            for name, parameter in model.named_parameters()
            if "weight" in name and parameter.grad is not None
        ]
        if not values:
            raise ValueError("TE-NAS NTK found no weight gradients")
        gradients.append(torch.cat(values))
    matrix = torch.stack(gradients)
    kernel = matrix @ matrix.T
    eigenvalues = torch.linalg.eigvalsh(kernel.float())
    smallest = eigenvalues[0].abs().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    return float((eigenvalues[-1].abs() / smallest).detach())


def te_nas_score(model: Any, inputs: Any, *_: Any) -> float:
    """TER-Score first-party adaptation: RN minus NTK condition number."""

    return _linear_regions(model, inputs) - _ntk_condition(model, inputs)
