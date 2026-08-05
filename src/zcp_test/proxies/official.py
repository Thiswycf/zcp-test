from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from zcp_test.proxies.base import ZeroCostProxy
from zcp_test.types import ProxyCapability, ProxyContext


ProxyFunction = Callable[[Any, Any, Any | None, Any | None], float]


def _output_tensor(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


@contextmanager
def _preserve_gradients(model: Any) -> Any:
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    try:
        yield
    finally:
        for name, parameter in model.named_parameters():
            gradient = gradients[name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device)


@contextmanager
def _preserve_state(model: Any) -> Any:
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    parameter_dtypes = {name: value.dtype for name, value in model.named_parameters()}
    buffer_dtypes = {name: value.dtype for name, value in model.named_buffers()}
    training = model.training
    try:
        yield
    finally:
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(dtype=parameter_dtypes[name])
        for name, buffer in model.named_buffers():
            buffer.data = buffer.data.to(dtype=buffer_dtypes[name])
        model.load_state_dict(state)
        model.train(training)


def gradnorm(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """Official zero-cost-nas GradNorm reduction over Conv2d/Linear weights."""
    import torch

    if labels is None or loss_fn is None:
        raise ValueError("gradnorm requires labels and loss_fn")

    with _preserve_gradients(model):
        model.zero_grad()
        output = _output_tensor(model(inputs))
        loss_fn(output, labels).backward()
        score = output.new_zeros(())
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if module.weight.grad is not None:
                    score = score + module.weight.grad.norm()
        return float(score.detach().cpu())


def synflow(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """Official SynFlow formula with a one-sample all-ones input."""
    import torch

    del labels, loss_fn
    if inputs.ndim < 2:
        raise ValueError("synflow requires batched inputs")

    with _preserve_state(model), _preserve_gradients(model):
        model.double()
        with torch.no_grad():
            for value in model.state_dict().values():
                if value.is_floating_point() or value.is_complex():
                    value.abs_()
        model.zero_grad()
        shape = (1, *inputs.shape[1:])
        ones = torch.ones(shape, device=inputs.device, dtype=torch.float64)
        _output_tensor(model(ones)).sum().backward()
        score = 0.0
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                gradient = module.weight.grad
                if gradient is not None:
                    score += float((module.weight * gradient).abs().sum().detach().cpu())
        return score


def naswot(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """NASWOT log-determinant of ReLU sample-agreement kernels."""
    import numpy as np
    import torch

    del labels, loss_fn
    kernel = torch.zeros(
        (inputs.shape[0], inputs.shape[0]), device=inputs.device, dtype=torch.float64
    )
    handles = []

    def count_patterns(_module: Any, module_inputs: Any, _output: Any) -> None:
        nonlocal kernel
        activation = module_inputs[0] if isinstance(module_inputs, tuple) else module_inputs
        activation = activation.detach().reshape(activation.shape[0], -1)
        binary = (activation > 0).to(dtype=kernel.dtype)
        kernel += binary @ binary.T + (1.0 - binary) @ (1.0 - binary).T

    try:
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                handles.append(module.register_forward_hook(count_patterns))
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    if not handles:
        raise ValueError("naswot found no ReLU modules")
    _, logdet = np.linalg.slogdet(kernel.cpu().numpy())
    return float(logdet)


def jacob_cov(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """Official JacobCov eigenspectrum score with the upstream 1e-5 offset."""
    import numpy as np

    del labels, loss_fn
    with _preserve_gradients(model):
        model.zero_grad()
        sample = inputs.detach().clone().requires_grad_(True)
        output = _output_tensor(model(sample))
        output.backward(gradient=output.new_ones(output.shape))
        jacobian = sample.grad.detach().reshape(sample.shape[0], -1).cpu().numpy()

    correlations = np.corrcoef(jacobian)
    eigenvalues = np.linalg.eigvals(correlations)
    score = -np.sum(np.log(eigenvalues + 1e-5) + 1.0 / (eigenvalues + 1e-5))
    return float(np.real_if_close(score))


def effective_rank(matrix: Any) -> float:
    """NEAR effective rank, exp(Shannon entropy of normalized singular values)."""
    import torch

    singular_values = torch.linalg.svdvals(matrix.detach())
    total = singular_values.sum()
    if not bool(torch.isfinite(total)) or total.item() == 0.0:
        return 0.0
    probabilities = singular_values / total
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return float(entropy.exp().detach().cpu())


def _near_batches(inputs: Any, labels: Any | None) -> Iterable[Any]:
    import torch

    if torch.is_tensor(inputs):
        yield inputs
        return
    for batch in inputs:
        if isinstance(batch, (tuple, list)):
            yield batch[0]
        else:
            yield batch


def near(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """NEAR@4d5d7f1 sum of per-layer effective activation ranks."""
    import torch

    del loss_fn
    activation_types = tuple(
        value
        for name in torch.nn.modules.activation.__all__
        if isinstance((value := getattr(torch.nn, name, None)), type)
    )
    layers = [
        module
        for module in model.modules()
        if hasattr(module, "weight") or isinstance(module, activation_types)
    ]
    activations: list[Any | None] = [None] * len(layers)
    finished: set[int] = set()
    handles: list[Any] = []

    def capture(layer_index: int) -> Callable[..., None]:
        def hook(_module: Any, _module_inputs: Any, output: Any) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output) or output.ndim < 2:
                return
            width = output.shape[-1]
            if output.ndim > 2:
                output = output.transpose(1, 3).flatten(0, 2)
            previous = activations[layer_index]
            activations[layer_index] = output if previous is None else torch.cat((previous, output))
            activation = activations[layer_index]
            if activation.shape[0] >= activation.shape[1]:
                groups = activation.shape[0] // width
                start = 0 if groups <= 1 else int(torch.randint(groups - 1, ()).item()) * width
                activations[layer_index] = activation[start : start + activation.shape[1]]
                handles[layer_index].remove()
                finished.add(layer_index)

        return hook

    try:
        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(capture(layer_index)))
        for batch_inputs in _near_batches(inputs, labels):
            model(batch_inputs)
            called = {index for index, value in enumerate(activations) if value is not None}
            if called and finished == called:
                break
    finally:
        for handle in handles:
            handle.remove()

    return sum(effective_rank(activation) for activation in activations if activation is not None)


def swap(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """Official SWAP count of unique neuron-wise sample activation patterns."""
    import torch

    del labels, loss_fn
    features = []
    handles = []

    def capture(_module: Any, module_inputs: Any, output: Any) -> None:
        module_input = module_inputs[0] if isinstance(module_inputs, tuple) else module_inputs
        if module_input.ndim == 4:
            features.append(output.detach())

    try:
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                handles.append(module.register_forward_hook(capture))
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    if not features:
        raise ValueError("swap found no 4D ReLU activations")
    patterns = torch.cat([feature.reshape(inputs.shape[0], -1) for feature in features], dim=1)
    return float(torch.unique(torch.sign(patterns).T, dim=0).shape[0])


def _gaussian_initialize(model: Any) -> None:
    import torch

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


def zen(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
    *,
    mixup_gamma: float = 1e-2,
    repetitions: int = 1,
) -> float:
    """Official Zen score using Gaussian initialization and pre-GAP features."""
    import numpy as np
    import torch

    del labels, loss_fn
    forward_pre_gap = getattr(model, "forward_pre_GAP", None)
    if not callable(forward_pre_gap):
        raise NotImplementedError("zen requires model.forward_pre_GAP")
    if repetitions < 1:
        raise ValueError("zen repetitions must be positive")

    scores = []
    with _preserve_state(model), torch.no_grad():
        for _ in range(repetitions):
            _gaussian_initialize(model)
            first_input = torch.randn_like(inputs)
            second_input = torch.randn_like(inputs)
            first_output = forward_pre_gap(first_input)
            mixed_output = forward_pre_gap(first_input + mixup_gamma * second_input)
            dimensions = tuple(range(1, first_output.ndim))
            difference = (first_output - mixed_output).abs().sum(dim=dimensions).mean()
            score = torch.log(difference)
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    score = score + torch.log(torch.sqrt(module.running_var.mean()))
            scores.append(float(score.cpu()))
    return float(np.mean(scores))


def _zico_batches(inputs: Any, labels: Any | None) -> Iterable[tuple[Any, Any]]:
    import torch

    if torch.is_tensor(inputs):
        if labels is None:
            raise ValueError("zico requires labels")
        yield inputs, labels
        return
    yield from inputs


def zico(
    model: Any,
    inputs: Any,
    labels: Any | None = None,
    loss_fn: Any | None = None,
) -> float:
    """Official ZiCo sum of log layer-wise gradient signal/noise ratios."""
    import numpy as np
    import torch

    if loss_fn is None:
        raise ValueError("zico requires loss_fn")
    gradients: dict[str, list[Any]] = {}

    with _preserve_gradients(model):
        for batch_inputs, batch_labels in _zico_batches(inputs, labels):
            model.zero_grad()
            output = _output_tensor(model(batch_inputs))
            loss_fn(output, batch_labels).backward()
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    if module.weight.grad is None:
                        continue
                    gradients.setdefault(name, []).append(
                        module.weight.grad.detach().cpu().reshape(-1).numpy().copy()
                    )

    score = 0.0
    for layer_gradients in gradients.values():
        values = np.asarray(layer_gradients)
        standard_deviation = np.std(values, axis=0)
        nonzero = np.nonzero(standard_deviation)[0]
        mean_absolute = np.mean(np.abs(values), axis=0)
        ratio_sum = np.sum(mean_absolute[nonzero] / standard_deviation[nonzero])
        if ratio_sum != 0:
            score += math.log(float(ratio_sum))
    return score


class OfficialFormulaProxy(ZeroCostProxy):
    def __init__(self, capability: ProxyCapability, function: ProxyFunction) -> None:
        self.capability = capability
        self._function = function

    def compute(
        self,
        model: Any,
        inputs: Any,
        labels: Any | None = None,
        loss_fn: Any | None = None,
    ) -> float:
        return self._function(model, inputs, labels, loss_fn)

    def compute_context(self, model: Any, context: ProxyContext) -> float:
        batches = [(context.inputs, context.labels)]
        if context.batch_provider is not None:
            batches = []
            provider = context.batch_provider()
            for _ in range(context.proxy_batches):
                try:
                    batches.append(next(provider))
                except StopIteration:
                    break
            if not batches:
                raise ValueError("proxy batch provider produced no batches")
        if self.capability.proxy_id == "near":
            return near(model, [inputs for inputs, _labels in batches])
        if self.capability.proxy_id == "zico":
            return zico(model, batches, loss_fn=context.loss_fn)
        if self.capability.proxy_id == "zen":
            return zen(
                model,
                context.inputs,
                repetitions=context.proxy_repetitions,
            )
        values = [
            self._function(model, context.inputs, context.labels, context.loss_fn)
            for _ in range(context.proxy_repetitions)
        ]
        return sum(values) / len(values)


_SOURCES = {
    "gradnorm": "https://github.com/mohsaied/zero-cost-nas/tree/b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "synflow": "https://github.com/mohsaied/zero-cost-nas/tree/b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "naswot": "https://github.com/BayesWatch/nas-without-training/tree/b3a82a6642564df115f989ff940ec6b8ef9ca9d3",
    "jacob_cov": "https://github.com/mohsaied/zero-cost-nas/tree/b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "near": "https://github.com/ReiherGroup/NEAR/tree/4d5d7f1bf005b67b352c078190c6810ca63fbadb",
    "swap": "https://github.com/pym1024/SWAP/tree/0853fc866051dca2b3b99d068502549de3686bd1",
    "zen": "https://github.com/idstcv/ZenNAS/tree/d1d617e0352733d39890fb64ea758f9c85b28c1a",
    "zico": "https://github.com/SLDGroup/ZiCo/tree/b0fec65923a90e84501593f675b1e2f422d79e3d",
}

OFFICIAL_IMPLEMENTATIONS: Mapping[str, tuple[ProxyFunction, ProxyCapability]] = {
    "gradnorm": (
        gradnorm,
        ProxyCapability(
            "gradnorm",
            version="zero-cost-nas-b5059bc",
            requires_labels=True,
            requires_loss_fn=True,
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["gradnorm"],
        ),
    ),
    "synflow": (
        synflow,
        ProxyCapability(
            "synflow",
            version="zero-cost-nas-b5059bc",
            requires_data=False,
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["synflow"],
        ),
    ),
    "naswot": (
        naswot,
        ProxyCapability(
            "naswot",
            version="nas-without-training-b3a82a6",
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["naswot"],
        ),
    ),
    "jacob_cov": (
        jacob_cov,
        ProxyCapability(
            "jacob_cov",
            version="zero-cost-nas-b5059bc",
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["jacob_cov"],
        ),
    ),
    "near": (
        near,
        ProxyCapability(
            "near",
            version="near-4d5d7f1",
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["near"],
        ),
    ),
    "swap": (
        swap,
        ProxyCapability(
            "swap",
            version="swap-0853fc8",
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["swap"],
        ),
    ),
    "zen": (
        zen,
        ProxyCapability(
            "zen",
            version="zennas-d1d617e",
            requires_data=False,
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["zen"],
        ),
    ),
    "zico": (
        zico,
        ProxyCapability(
            "zico",
            version="zico-b0fec65",
            requires_labels=True,
            requires_loss_fn=True,
            default_batches=2,
            implementation_fidelity="fixed_source_formula_port",
            source=_SOURCES["zico"],
        ),
    ),
}

_COMMITS = {
    "gradnorm": "b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "synflow": "b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "naswot": "b3a82a6642564df115f989ff940ec6b8ef9ca9d3",
    "jacob_cov": "b5059bc42e2275534f584bc21a2d28ab8427cd8e",
    "near": "4d5d7f1bf005b67b352c078190c6810ca63fbadb",
    "swap": "0853fc866051dca2b3b99d068502549de3686bd1",
    "zen": "d1d617e0352733d39890fb64ea758f9c85b28c1a",
    "zico": "b0fec65923a90e84501593f675b1e2f422d79e3d",
}
_LICENSES = {
    "gradnorm": "Apache-2.0",
    "synflow": "Apache-2.0",
    "naswot": "NOASSERTION",
    "jacob_cov": "Apache-2.0",
    "near": "BSD-3-Clause",
    "swap": "NOASSERTION",
    "zen": "NOASSERTION",
    "zico": "Apache-2.0",
}
OFFICIAL_IMPLEMENTATIONS = {
    name: (
        function,
        replace(
            capability,
            source_commit=_COMMITS[name],
            license=_LICENSES[name],
            official_code_available=True,
            protocol_domain="relu_cnn",
        ),
    )
    for name, (function, capability) in OFFICIAL_IMPLEMENTATIONS.items()
}


def official_proxy_factories() -> dict[str, Callable[[], OfficialFormulaProxy]]:
    """Return fresh factories without mutating the process-global proxy registry."""
    return {
        name: (
            lambda function=function, capability=capability: OfficialFormulaProxy(
                capability, function
            )
        )
        for name, (function, capability) in OFFICIAL_IMPLEMENTATIONS.items()
    }


__all__ = [
    "OFFICIAL_IMPLEMENTATIONS",
    "OfficialFormulaProxy",
    "effective_rank",
    "gradnorm",
    "jacob_cov",
    "naswot",
    "near",
    "official_proxy_factories",
    "swap",
    "synflow",
    "zen",
    "zico",
]
