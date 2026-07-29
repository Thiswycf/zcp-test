from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


@contextmanager
def isolated_model(model: Any) -> Iterator[Any]:
    import torch

    training = model.training
    state = copy.deepcopy(model.state_dict())
    requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    hooks_before = {id(module): (set(module._forward_hooks), set(module._backward_hooks)) for module in model.modules()}
    buffers_before = {id(module): set(module._buffers) for module in model.modules()}
    try:
        yield model
    finally:
        for module in model.modules():
            for name in set(module._buffers) - buffers_before[id(module)]:
                del module._buffers[name]
        model.load_state_dict(state)
        model.train(training)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
            parameter.grad = None
        for module in model.modules():
            forward, backward = hooks_before[id(module)]
            for key in set(module._forward_hooks) - forward:
                module._forward_hooks.pop(key, None)
            for key in set(module._backward_hooks) - backward:
                module._backward_hooks.pop(key, None)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
