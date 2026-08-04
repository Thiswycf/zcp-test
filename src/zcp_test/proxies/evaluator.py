from __future__ import annotations

import math
import importlib.util
import time
from typing import Any

from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.isolation import isolated_model
from zcp_test.types import ProxyOutput, RecordStatus, ScoreResult


def evaluate_proxy(proxy_id: str, model: Any, inputs: Any = None, labels: Any = None, loss_fn: Any = None, model_family: str = "cnn", unsupported_reason: str | None = None) -> ScoreResult:
    load_builtin_proxies()
    proxy = PROXIES.create(proxy_id)
    result_metadata = {
        "proxy_version": proxy.capability.version,
        "direction": proxy.capability.direction,
        "primary_component": proxy.capability.primary_component,
        "implementation_fidelity": proxy.capability.implementation_fidelity,
        "source": proxy.capability.source,
        "alias_of": proxy.capability.alias_of,
        "resource_direction": proxy.capability.resource_direction,
    }
    if unsupported_reason is not None:
        return ScoreResult(
            status=RecordStatus.UNSUPPORTED,
            error_message=unsupported_reason,
            **result_metadata,
        )
    if model_family not in proxy.capability.model_families:
        return ScoreResult(
            status=RecordStatus.UNSUPPORTED,
            error_message=f"{proxy_id} does not support {model_family}",
            **result_metadata,
        )
    missing_contract = []
    if proxy.capability.requires_inputs and inputs is None:
        missing_contract.append("inputs")
    if proxy.capability.requires_labels and labels is None:
        missing_contract.append("labels")
    if proxy.capability.requires_loss_fn and loss_fn is None:
        missing_contract.append("loss_fn")
    missing_dependencies = [
        name for name in proxy.capability.dependencies if importlib.util.find_spec(name) is None
    ]
    if missing_dependencies:
        missing_contract.append("dependencies=" + ",".join(missing_dependencies))
    if missing_contract:
        return ScoreResult(
            status=RecordStatus.UNSUPPORTED,
            error_message=(
                f"{proxy_id} input contract is unavailable: " + ", ".join(missing_contract)
            ),
            **result_metadata,
        )
    started = time.perf_counter()
    peak_memory_mb = None
    try:
        try:
            import torch

            parameter = next(model.parameters(), None)
            if parameter is not None and parameter.is_cuda:
                torch.cuda.reset_peak_memory_stats(parameter.device)
        except (ImportError, RuntimeError):
            torch = None
        with isolated_model(model):
            value = proxy.compute(model, inputs, labels, loss_fn)
        if isinstance(value, ProxyOutput):
            if value.primary_component != proxy.capability.primary_component:
                raise ValueError(
                    f"{proxy_id} returned primary component {value.primary_component!r}, "
                    f"but declared {proxy.capability.primary_component!r}"
                )
            primary_component = value.primary_component
            normalized = {name: float(component) for name, component in value.components.items()}
            undeclared = sorted(set(normalized).difference(proxy.capability.components))
            if undeclared:
                raise ValueError(
                    f"{proxy_id} returned undeclared components: {', '.join(undeclared)}"
                )
            if primary_component in normalized and normalized[primary_component] != float(value.score):
                raise ValueError(
                    f"{proxy_id} returned a primary component value inconsistent with score"
                )
            normalized.setdefault(primary_component, float(value.score))
            score = float(value.score)
        elif isinstance(value, dict):
            normalized = {name: float(component) for name, component in value.items()}
            primary_component = proxy.capability.primary_component
            if primary_component not in normalized:
                raise ValueError(
                    f"{proxy_id} did not return declared primary component {primary_component!r}"
                )
            score = normalized[primary_component]
        else:
            primary_component = proxy.capability.primary_component
            score = float(value)
            normalized = {primary_component: score}
        if not all(math.isfinite(component) for component in normalized.values()):
            raise ValueError("proxy returned NaN or infinity")
        if not math.isfinite(score):
            raise ValueError("proxy primary score is NaN or infinity")
        if torch is not None and parameter is not None and parameter.is_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(parameter.device) / (1024**2)
        return ScoreResult(
            score=score,
            components=normalized,
            duration_seconds=time.perf_counter() - started,
            peak_memory_mb=peak_memory_mb,
            **{**result_metadata, "primary_component": primary_component},
        )
    except NotImplementedError as error:
        return ScoreResult(status=RecordStatus.UNSUPPORTED, error_type=type(error).__name__, error_message=str(error), duration_seconds=time.perf_counter() - started, **result_metadata)
    except Exception as error:
        return ScoreResult(status=RecordStatus.FAILED, error_type=type(error).__name__, error_message=str(error), duration_seconds=time.perf_counter() - started, **result_metadata)
