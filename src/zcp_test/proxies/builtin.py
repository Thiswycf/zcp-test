from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable

from zcp_test.proxies import PROXIES
from zcp_test.proxies.base import ZeroCostProxy
from zcp_test.types import ProxyCapability, ScoreDirection


class FunctionProxy(ZeroCostProxy):
    def __init__(self, capability: ProxyCapability, function: Callable[..., float | dict[str, float]]) -> None:
        self.capability = capability
        self._function = function

    def compute(self, model: Any, inputs: Any, labels: Any | None = None, loss_fn: Any | None = None) -> float | dict[str, float]:
        return self._function(model, inputs, labels, loss_fn)


def _params(model: Any, *_: Any) -> float:
    return float(sum(parameter.numel() for parameter in model.parameters()))


def _flops(model: Any, inputs: Any, *_: Any) -> float:
    from thop import profile

    value, _ = profile(model, inputs=(inputs,), verbose=False)
    return float(value)


def _gradnorm(model: Any, inputs: Any, labels: Any, loss_fn: Any) -> float:
    if labels is None or loss_fn is None:
        raise ValueError("gradnorm requires labels and loss_fn")
    model.zero_grad(set_to_none=True)
    output = model(inputs)
    if isinstance(output, (tuple, list)):
        output = output[-1]
    loss_fn(output, labels).backward()
    total = sum(float(parameter.grad.norm().item()) ** 2 for parameter in model.parameters() if parameter.grad is not None)
    return math.sqrt(total)


def _synflow(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    parameter_dtypes = {
        name: parameter.dtype for name, parameter in model.named_parameters()
    }
    buffer_dtypes = {
        name: buffer.dtype
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point() or buffer.is_complex()
    }
    signs: dict[str, Any] = {}
    try:
        model.double()
        signs = {
            name: torch.sign(parameter.data)
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.data.abs_()
        model.zero_grad(set_to_none=True)
        shape = list(inputs.shape)
        shape[0] = 1
        output = model(torch.ones(shape, device=inputs.device, dtype=torch.float64))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        output.sum().backward()
        return sum(
            float((parameter.grad * parameter).abs().sum().detach())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    finally:
        for name, parameter in model.named_parameters():
            if name in signs:
                parameter.data.mul_(signs[name])
            parameter.data = parameter.data.to(dtype=parameter_dtypes[name])
        for name, buffer in model.named_buffers():
            if name in buffer_dtypes:
                buffer.data = buffer.data.to(dtype=buffer_dtypes[name])
        model.zero_grad(set_to_none=True)


def _naswot(model: Any, inputs: Any, *_: Any) -> float:
    import numpy as np
    import torch

    kernels: list[np.ndarray] = []
    handles = []

    def hook(_module: Any, _input: Any, output: Any) -> None:
        activation = (output.detach().flatten(1) > 0).float()
        kernel = activation @ activation.T + (1 - activation) @ (1 - activation).T
        kernels.append(kernel.cpu().numpy())

    activation_types = (
        torch.nn.ELU,
        torch.nn.GELU,
        torch.nn.Hardswish,
        torch.nn.LeakyReLU,
        torch.nn.PReLU,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.SELU,
        torch.nn.SiLU,
    )
    for module in model.modules():
        if isinstance(module, activation_types):
            handles.append(module.register_forward_hook(hook))
    model(inputs)
    for handle in handles:
        handle.remove()
    if not kernels:
        raise ValueError("NASWOT found no supported activation modules")
    _, logdet = np.linalg.slogdet(sum(kernels))
    return float(logdet)


def _effective_rank(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    import torch

    values: list[float] = []
    handles = []

    def hook(_module: Any, _input: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim < 2:
            return
        matrix = output.detach().flatten(1).float()
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(matrix)
        probability = singular / singular.sum().clamp_min(1e-12)
        values.append(float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())))

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(inputs)
    for handle in handles:
        handle.remove()
    if not values:
        raise ValueError("ER found no eligible activations")
    return {"mean": float(sum(values) / len(values)), "sum": float(sum(values))}


def _activation_matrix(model: Any, inputs: Any) -> Any:
    import torch

    activations = []
    handles = []

    def hook(_module: Any, _input: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim >= 2:
            activations.append(output.detach().flatten(1).float())

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(inputs)
    for handle in handles:
        handle.remove()
    if not activations:
        raise ValueError("No eligible activations")
    width = min(value.shape[1] for value in activations)
    return torch.cat([value[:, :width] for value in activations], dim=1)


def _meco(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    matrix = _activation_matrix(model, inputs)
    matrix = matrix - matrix.mean(0, keepdim=True)
    matrix = matrix / matrix.std(0, keepdim=True).clamp_min(1e-6)
    correlation = matrix @ matrix.T / max(1, matrix.shape[1])
    return float(torch.linalg.slogdet(correlation + torch.eye(correlation.shape[0], device=correlation.device) * 1e-5).logabsdet)


def _input_jacobian(model: Any, inputs: Any) -> Any:
    import torch

    sample = inputs.detach().clone().requires_grad_(True)
    output = model(sample)
    if isinstance(output, (tuple, list)):
        output = output[-1]
    gradient = torch.autograd.grad(output.sum(), sample, create_graph=False)[0]
    return gradient.flatten(1).float()


def _jacob_cov(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    jacobian = _input_jacobian(model, inputs)
    covariance = torch.corrcoef(jacobian)
    eigenvalues = torch.linalg.eigvalsh(torch.nan_to_num(covariance)).clamp_min(1e-6)
    return float(-(torch.log(eigenvalues) + 1.0 / eigenvalues).sum())


def _vkdnw(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    singular = torch.linalg.svdvals(_input_jacobian(model, inputs)).clamp_min(1e-8)
    return float(torch.log(singular).sum() - torch.log(singular.max() / singular.min()))


def _near(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    matrix = _activation_matrix(model, inputs)
    return float(torch.linalg.matrix_rank(matrix.float()))


def _swap(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    signatures = (_activation_matrix(model, inputs) > 0).to(torch.uint8)
    return float(torch.unique(signatures, dim=0).shape[0])


def _zen(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    noise = torch.randn_like(inputs)
    first, second = model(inputs), model(inputs + 1e-2 * noise)
    if isinstance(first, (tuple, list)):
        first, second = first[-1], second[-1]
    return float(torch.log((first - second).abs().mean().clamp_min(1e-12)))


def _ntkt(model: Any, inputs: Any, *_: Any) -> float:
    import torch

    gradients = []
    for sample in inputs:
        model.zero_grad(set_to_none=True)
        output = model(sample.unsqueeze(0))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        output.sum().backward()
        gradients.append(torch.cat([parameter.grad.flatten() for parameter in model.parameters() if parameter.grad is not None]))
    kernel = torch.stack(gradients) @ torch.stack(gradients).T
    return float(torch.linalg.slogdet(kernel + torch.eye(kernel.shape[0], device=kernel.device) * 1e-5).logabsdet)


def _zico(model: Any, inputs: Any, labels: Any, loss_fn: Any) -> float:
    import torch

    if labels is None or loss_fn is None:
        raise ValueError("zico requires labels and loss_fn")
    gradients = []
    for sample, label in zip(inputs, labels, strict=True):
        model.zero_grad(set_to_none=True)
        output = model(sample.unsqueeze(0))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        loss_fn(output, label.unsqueeze(0)).backward()
        gradients.append(torch.cat([parameter.grad.flatten() for parameter in model.parameters() if parameter.grad is not None]))
    values = torch.stack(gradients).abs()
    ratio = values.mean(0) / values.std(0, unbiased=False).clamp_min(1e-8)
    return float(torch.log(ratio.clamp_min(1e-8)).mean())


def _te_nas(model: Any, inputs: Any, labels: Any, loss_fn: Any) -> dict[str, float]:
    return {"synflow": _synflow(model, inputs), "naswot": _naswot(model, inputs), "gradnorm": _gradnorm(model, inputs, labels, loss_fn)}


def _az_nas(model: Any, inputs: Any, labels: Any, loss_fn: Any) -> dict[str, float]:
    return {"expressivity": _naswot(model, inputs), "trainability": _gradnorm(model, inputs, labels, loss_fn), "complexity": math.log1p(_params(model))}


def _az_nas_autoformer(model: Any, inputs: Any, *_: Any) -> dict[str, float]:
    from zcp_test.proxies.az_nas import autoformer_components

    return autoformer_components(model, inputs)


def _topology_er(mode: str) -> Callable[..., float]:
    def compute(model: Any, inputs: Any, *_: Any) -> float:
        import inspect

        import torch
        from torch.fx import symbolic_trace

        concrete_args = (
            {"return_auxiliary": False}
            if "return_auxiliary" in inspect.signature(model.forward).parameters
            else None
        )
        traced = symbolic_trace(model, concrete_args=concrete_args)
        module_nodes = [node for node in traced.graph.nodes if node.op == "call_module"]
        if not module_nodes:
            raise ValueError("Topology proxy found no traceable modules")
        index = {node: position for position, node in enumerate(module_nodes)}
        incoming = {position: set() for position in range(len(module_nodes))}
        outgoing = {position: set() for position in range(len(module_nodes))}
        for node in module_nodes:
            target = index[node]
            pending = list(node.all_input_nodes)
            visited = set()
            while pending:
                source_node = pending.pop()
                if source_node in visited:
                    continue
                visited.add(source_node)
                if source_node in index:
                    source = index[source_node]
                    incoming[target].add(source)
                    outgoing[source].add(target)
                else:
                    pending.extend(source_node.all_input_nodes)
        ranks: dict[str, float] = {}
        handles = []

        def make_hook(name: str) -> Callable[..., None]:
            def hook(_module: Any, _input: Any, output: Any) -> None:
                if not isinstance(output, torch.Tensor) or output.ndim < 2:
                    return
                matrix = output.detach().flatten(1).float()
                singular = torch.linalg.svdvals(matrix - matrix.mean(0, keepdim=True))
                probability = singular / singular.sum().clamp_min(1e-12)
                ranks[name] = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
            return hook

        named_modules = dict(model.named_modules())
        for node in module_nodes:
            if str(node.target) in named_modules:
                handles.append(named_modules[str(node.target)].register_forward_hook(make_hook(str(node.target))))
        model(inputs)
        for handle in handles:
            handle.remove()
        if not ranks:
            raise ValueError("Topology proxy captured no tensor activations")
        count = len(module_nodes)
        weights = [1.0] * count
        if mode == "pr":
            weights = [1.0 / count] * count
            for _ in range(100):
                updated = [(1 - 0.85) / count] * count
                for source in range(count):
                    destinations = outgoing[source]
                    if destinations:
                        for target in destinations:
                            updated[target] += 0.85 * weights[source] / len(destinations)
                if sum(
                    abs(left - right) for left, right in zip(updated, weights, strict=True)
                ) < 1e-8:
                    break
                weights = updated
        elif mode == "conn":
            weights = [(len(incoming[position]) + 1) * (len(outgoing[position]) + 1) for position in range(count)]
        elif mode == "deg":
            weights = [len(incoming[position]) + len(outgoing[position]) + 1 for position in range(count)]
        elif mode == "dist":
            distances = [0] * count
            for position in range(count):
                distances[position] = 1 + max((distances[parent] for parent in incoming[position]), default=0)
            maximum = max(distances)
            weights = [1.0 - abs(distance - (maximum + 1) / 2) / max(1, maximum) for distance in distances]
        weights_by_name = {str(node.target): weights[index[node]] for node in module_nodes}
        normalizer = sum(weights_by_name[name] for name in ranks)
        return sum(ranks[name] * weights_by_name[name] for name in ranks) / max(normalizer, 1e-12)

    return compute


def _unsupported(name: str) -> Callable[..., float]:
    def compute(*_: Any) -> float:
        raise NotImplementedError(f"{name} requires its specialized compatibility implementation")

    return compute


_IMPLEMENTATIONS: dict[str, tuple[Callable[..., Any], ProxyCapability]] = {
    "params": (_params, ProxyCapability("params", version="count-v2", model_families=("cnn", "transformer"), requires_data=False, resource_direction=ScoreDirection.MINIMIZE)),
    "flops": (_flops, ProxyCapability("flops", version="thop-v2", model_families=("cnn", "transformer"), dependencies=("thop",), resource_direction=ScoreDirection.MINIMIZE)),
    "gradnorm": (_gradnorm, ProxyCapability("gradnorm", requires_labels=True)),
    "synflow": (_synflow, ProxyCapability("synflow", version="double-v2", model_families=("cnn", "transformer"), requires_data=False)),
    "naswot": (_naswot, ProxyCapability("naswot", model_families=("cnn", "transformer"))),
    "er": (_effective_rank, ProxyCapability("er", model_families=("cnn", "transformer"), components=("mean", "sum"), primary_component="mean")),
    "ter": (_effective_rank, ProxyCapability("ter", model_families=("cnn",), components=("mean", "sum"), primary_component="mean")),
    "meco": (_meco, ProxyCapability("meco", version="portable-v1", model_families=("cnn", "transformer"))),
    "meco_opt": (_meco, ProxyCapability("meco_opt", version="portable-v1", model_families=("cnn", "transformer"))),
    "jacob_cov": (_jacob_cov, ProxyCapability("jacob_cov", version="portable-v1", model_families=("cnn", "transformer"))),
    "vkdnw": (_vkdnw, ProxyCapability("vkdnw", version="portable-v1", model_families=("cnn", "transformer"))),
    "near": (_near, ProxyCapability("near", version="portable-v1", model_families=("cnn", "transformer"))),
    "swap": (_swap, ProxyCapability("swap", version="portable-v1", model_families=("cnn", "transformer"))),
    "zen": (_zen, ProxyCapability("zen", version="portable-v1", model_families=("cnn", "transformer"))),
    "ntkt": (_ntkt, ProxyCapability("ntkt", version="portable-v1", model_families=("cnn", "transformer"))),
    "zico": (_zico, ProxyCapability("zico", version="portable-v1", model_families=("cnn", "transformer"), requires_labels=True)),
    "te_nas": (_te_nas, ProxyCapability("te_nas", version="portable-v2", model_families=("cnn", "transformer"), requires_labels=True, components=("synflow", "naswot", "gradnorm"), primary_component="synflow")),
    "az_nas": (_az_nas, ProxyCapability("az_nas", version="portable-v1", model_families=("cnn", "transformer"), requires_labels=True, components=("expressivity", "trainability", "complexity"), primary_component="expressivity")),
    "az_nas_autoformer": (_az_nas_autoformer, ProxyCapability("az_nas_autoformer", version="aznas-5e6683-autoformer-stable-v1", model_families=("transformer",), components=("expressivity", "trainability", "complexity"), primary_component="expressivity")),
    "er_pr": (_topology_er("pr"), ProxyCapability("er_pr", version="fx-v1", model_families=("cnn",))),
    "er_conn": (_topology_er("conn"), ProxyCapability("er_conn", version="fx-v1", model_families=("cnn",))),
    "er_deg": (_topology_er("deg"), ProxyCapability("er_deg", version="fx-v1", model_families=("cnn",))),
    "er_dist": (_topology_er("dist"), ProxyCapability("er_dist", version="fx-v1", model_families=("cnn",))),
}

_SPECIALIZED: dict[str, tuple[str, ...]] = {}

_SOURCES = {
    "gradnorm": "https://arxiv.org/abs/2101.08134",
    "synflow": "https://arxiv.org/abs/2006.05467",
    "naswot": "https://proceedings.mlr.press/v139/mellor21a.html",
    "meco": "https://papers.nips.cc/paper_files/paper/2023/hash/bfa815ac6f08f4ada34fe22be054f2b9-Abstract-Conference.html",
    "jacob_cov": "https://arxiv.org/abs/2101.08134",
    "zen": "https://arxiv.org/abs/2102.01063",
    "zico": "https://openreview.net/forum?id=rwo-ls5GqGn",
    "te_nas": "https://arxiv.org/abs/2102.11535",
    "az_nas": "https://arxiv.org/abs/2403.19232",
    "az_nas_autoformer": "https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_AutoFormer",
}
for _proxy_name, (_proxy_function, _proxy_capability) in tuple(_IMPLEMENTATIONS.items()):
    if _proxy_name in {"params", "flops"}:
        _fidelity = "structural_measure"
    elif _proxy_name in {"ter", "meco_opt"}:
        _fidelity = "alias"
    elif _proxy_name in {"te_nas", "az_nas"}:
        _fidelity = "portable_composite_approximation"
    elif _proxy_name == "az_nas_autoformer":
        _fidelity = "paper_formula_port_stabilized"
    elif _proxy_name == "er" or _proxy_name.startswith("er_"):
        _fidelity = "project_extension"
    else:
        _fidelity = "paper_formula_port_unverified"
    _IMPLEMENTATIONS[_proxy_name] = (
        _proxy_function,
        replace(
            _proxy_capability,
            implementation_fidelity=_fidelity,
            source=_SOURCES.get(_proxy_name),
            alias_of={"ter": "er", "meco_opt": "meco"}.get(_proxy_name),
        ),
    )

for _name, (_function, _capability) in _IMPLEMENTATIONS.items():
    PROXIES.register(_name, lambda function=_function, capability=_capability: FunctionProxy(capability, function))
for _name, _families in _SPECIALIZED.items():
    _capability = ProxyCapability(_name, model_families=_families)
    PROXIES.register(_name, lambda name=_name, capability=_capability: FunctionProxy(capability, _unsupported(name)))
