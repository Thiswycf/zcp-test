from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


@contextmanager
def isolated_model(model: Any) -> Iterator[Any]:
    import torch

    training = {id(module): module.training for module in model.modules()}
    state = copy.deepcopy(model.state_dict())
    requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    hooks_before = {
        id(module): (
            set(module._forward_pre_hooks),
            set(module._forward_hooks),
            set(module._backward_hooks),
        )
        for module in model.modules()
    }
    buffers_before = {id(module): set(module._buffers) for module in model.modules()}
    try:
        yield model
    finally:
        for module in model.modules():
            for name in set(module._buffers) - buffers_before[id(module)]:
                del module._buffers[name]
        model.load_state_dict(state)
        for module in model.modules():
            module.training = training[id(module)]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
            gradient = gradients[name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device)
        for module in model.modules():
            forward_pre, forward, backward = hooks_before[id(module)]
            for key in set(module._forward_pre_hooks) - forward_pre:
                module._forward_pre_hooks.pop(key, None)
            for key in set(module._forward_hooks) - forward:
                module._forward_hooks.pop(key, None)
            for key in set(module._backward_hooks) - backward:
                module._backward_hooks.pop(key, None)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
