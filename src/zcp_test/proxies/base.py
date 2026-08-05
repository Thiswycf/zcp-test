from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from zcp_test.types import ProxyCapability, ProxyContext, ProxyOutput


class ZeroCostProxy(ABC):
    capability: ProxyCapability

    @abstractmethod
    def compute(
        self,
        model: Any,
        inputs: Any,
        labels: Any | None = None,
        loss_fn: Any | None = None,
    ) -> float | dict[str, float] | ProxyOutput: ...

    def compute_context(
        self,
        model: Any,
        context: ProxyContext,
    ) -> float | dict[str, float] | ProxyOutput:
        return self.compute(model, context.inputs, context.labels, context.loss_fn)
