from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FORMAL_TRAINING_PROTOCOLS: dict[str, dict[str, Any]] = {
    "darts-original-cifar10": {
        "space": "darts",
        "dataset": "cifar10",
        "input_size": 32,
        "model_profile": "cifar10",
        "init_channels": 36,
        "layers": 20,
        "epochs": 600,
        "optimizer": "sgd",
        "learning_rate": 0.025,
        "weight_decay": 0.0003,
        "scheduler": "cosine",
        "batch_size": 96,
        "nesterov": False,
    },
    "darts-cifar100-adaptation": {
        "space": "darts",
        "dataset": "cifar100",
        "input_size": 32,
        "model_profile": "cifar100",
        "init_channels": 36,
        "layers": 20,
        "epochs": 600,
        "optimizer": "sgd",
        "learning_rate": 0.025,
        "weight_decay": 0.0003,
        "scheduler": "cosine",
        "batch_size": 96,
        "nesterov": False,
    },
    "darts-original-imagenet": {
        "space": "darts",
        "dataset": "imagenet1k",
        "input_size": 224,
        "model_profile": "imagenet",
        "init_channels": 48,
        "layers": 14,
        "epochs": 250,
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "weight_decay": 0.00003,
        "scheduler": "step",
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.97,
        "batch_size": 128,
        "nesterov": True,
    },
    "tenas-retrain-imagenet": {
        "space": "darts",
        "dataset": "imagenet1k",
        "input_size": 224,
        "model_profile": "imagenet",
        "init_channels": 48,
        "layers": 14,
        "epochs": 250,
        "optimizer": "sgd",
        "learning_rate": 0.5,
        "weight_decay": 0.00003,
        "scheduler": "cosine",
        "batch_size": 128,
        "nesterov": False,
    },
}


def scale_learning_rate(
    base_learning_rate: float,
    per_device_batch_size: int,
    world_size: int,
    reference_batch_size: int,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, int]:
    """Apply the explicit linear batch-size rule used by a training protocol."""
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    for name, value in {
        "per_device_batch_size": per_device_batch_size,
        "world_size": world_size,
        "reference_batch_size": reference_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    global_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps
    return base_learning_rate * global_batch_size / reference_batch_size, global_batch_size


def validate_formal_training_protocol(config: Mapping[str, Any]) -> str:
    protocol = str(config.get("protocol", ""))
    expected = FORMAL_TRAINING_PROTOCOLS.get(protocol)
    if expected is None:
        raise NotImplementedError(
            f"Formal training protocol {protocol!r} is not approved by this zcp-test version"
        )
    mismatches = [
        f"{field}={config.get(field)!r} (expected {value!r})"
        for field, value in expected.items()
        if config.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            f"Formal training protocol {protocol!r} does not match its accepted profile: "
            + "; ".join(mismatches)
        )
    return protocol


__all__ = [
    "FORMAL_TRAINING_PROTOCOLS",
    "scale_learning_rate",
    "validate_formal_training_protocol",
]
