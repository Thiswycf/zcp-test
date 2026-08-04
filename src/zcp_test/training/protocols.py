from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

ONE_PERCENT_DATA_FRACTION = 0.01


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
        "momentum": 0.9,
        "weight_decay": 0.0003,
        "scheduler": "cosine",
        "warmup_epochs": 0,
        "batch_size": 96,
        "batch_size_semantics": "global",
        "nesterov": False,
        "auxiliary": True,
        "auxiliary_weight": 0.4,
        "drop_path_prob": 0.2,
        "grad_clip": 5.0,
        "cutout_length": 16,
        "label_smoothing": 0.0,
        "deterministic": True,
        "implementation_source": "https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/train.py",
        "implementation_commit": "f276dd346a09ae3160f8e3aca5c7b193fda1da37",
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
        "momentum": 0.9,
        "weight_decay": 0.0003,
        "scheduler": "cosine",
        "warmup_epochs": 0,
        "batch_size": 96,
        "batch_size_semantics": "global",
        "nesterov": False,
        "auxiliary": True,
        "auxiliary_weight": 0.4,
        "drop_path_prob": 0.2,
        "grad_clip": 5.0,
        "cutout_length": 16,
        "label_smoothing": 0.0,
        "deterministic": True,
        "implementation_source": "https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/train.py",
        "implementation_commit": "f276dd346a09ae3160f8e3aca5c7b193fda1da37",
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
        "momentum": 0.9,
        "weight_decay": 0.00003,
        "scheduler": "step",
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.97,
        "warmup_epochs": 0,
        "batch_size": 128,
        "batch_size_semantics": "global",
        "nesterov": False,
        "auxiliary": True,
        "auxiliary_weight": 0.4,
        "drop_path_prob": 0.0,
        "grad_clip": 5.0,
        "label_smoothing": 0.1,
        "deterministic": True,
        "implementation_source": "https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/train_imagenet.py",
        "implementation_commit": "f276dd346a09ae3160f8e3aca5c7b193fda1da37",
    },
}

CANDIDATE_TRAINING_PROTOCOLS: dict[str, dict[str, Any]] = {
    "aznas-plainnet-mbv2-scratch-5e6683a2": {
        "space": "zennas_plainnet_mbv2",
        "dataset": "imagenet1k",
        "input_size": 224,
        "epochs": 150,
        "optimizer": "sgd",
        "learning_rate": 0.4,
        "learning_rate_scaling": "linear_global_batch",
        "learning_rate_reference_batch_size": 256,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.00004,
        "exclude_bias_norm_from_weight_decay": True,
        "scheduler": "cosine_warmup_step",
        "warmup_epochs": 5,
        "batch_size": 512,
        "batch_size_semantics": "global",
        "label_smoothing": 0.1,
        "resize_scale": 0.08,
        "color_distortion": "aznas_imagenet",
        "model_init": "custom_kaiming",
        "bn_momentum": 0.01,
        "use_se": True,
        "amp": True,
        "deterministic": True,
        "implementation_commit": "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47",
    },
    "project-ofa-proxyless-mbv2-scratch-v1": {
        "space": "ofa_proxyless_mbv2",
        "dataset": "imagenet1k",
        "input_size": 224,
        "candidate_input_size_policy": "architecture_resolution",
        "training_protocol_fidelity": "project_candidate_resolution_adaptation",
        "epochs": 150,
        "optimizer": "sgd",
        "learning_rate": 0.05,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.00004,
        "exclude_norm_from_weight_decay": True,
        "scheduler": "cosine_step",
        "warmup_epochs": 0,
        "batch_size": 256,
        "batch_size_semantics": "global",
        "test_batch_size": 500,
        "label_smoothing": 0.1,
        "resize_scale": 0.08,
        "color_distortion": "tf",
        "model_init": "he_fout",
        "amp": True,
        "deterministic": True,
        "implementation_commit": "b23018c9c369d22931f7422b71ca6a7eaa354c46",
    },
    "aznas-autoformer-scratch": {
        "space": "autoformer",
        "dataset": "imagenet1k",
        "input_size": 224,
        "epochs": 500,
        "optimizer": "adamw",
        "learning_rate": 0.0005,
        "learning_rate_scaling": "linear_global_batch",
        "learning_rate_reference_batch_size": 512,
        "target_global_batch_size": 2048,
        "gradient_accumulation_steps": "auto",
        "weight_decay": 0.05,
        "scheduler": "cosine",
        "warmup_epochs": 20,
        "warmup_learning_rate": 0.000001,
        "minimum_learning_rate": 0.00001,
        "batch_size": 256,
        "batch_size_semantics": "per_device",
        "label_smoothing": 0.1,
        "validation_label_smoothing": 0.0,
        "exclude_bias_norm_from_weight_decay": True,
        "amp": True,
        "deterministic": True,
        "drop_path_prob": 0.1,
        "color_jitter": 0.4,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "train_interpolation": "bicubic",
        "random_erase_probability": 0.25,
        "random_erase_mode": "pixel",
        "random_erase_count": 1,
        "mixup": 0.8,
        "cutmix": 1.0,
        "mixup_probability": 1.0,
        "mixup_switch_probability": 0.5,
        "mixup_mode": "batch",
        "repeated_augmentation": True,
        "repeated_augmentation_repeats": 3,
        "repeated_augmentation_selected_round": 256,
        "repeated_augmentation_selected_ratio": 0,
        "reference_world_size": 8,
        "reference_global_batch_size": 2048,
        "relative_position": True,
        "max_relative_position": 14,
        "change_qkv": True,
        "global_pool": True,
        "implementation_source": "https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_AutoFormer",
        "implementation_commit": "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47",
    },
}

FORMAL_TRAINING_PROTOCOLS["aznas-autoformer-scratch"] = {
    **CANDIDATE_TRAINING_PROTOCOLS["aznas-autoformer-scratch"],
    "formal_training_ready": True,
    "formal_training_acceptance": "docs/evidence/autoformer_single_candidate_dual_one_percent_completion_20260804.json",
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


def resolve_gradient_accumulation(
    requested: int | str,
    per_device_batch_size: int,
    world_size: int,
    target_global_batch_size: int | None,
    *,
    smoke: bool,
) -> int:
    """Resolve fixed or automatic micro-step accumulation without changing the protocol."""
    if smoke:
        return 1
    if str(requested).casefold() != "auto":
        steps = int(requested)
        if steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        return steps
    if target_global_batch_size is None:
        raise ValueError("automatic gradient accumulation requires target_global_batch_size")
    micro_global_batch_size = per_device_batch_size * world_size
    if target_global_batch_size <= 0 or target_global_batch_size % micro_global_batch_size:
        raise ValueError(
            "target_global_batch_size must be positive and divisible by "
            "per-device batch_size * WORLD_SIZE"
        )
    return target_global_batch_size // micro_global_batch_size


def resolve_per_device_batch_size(
    configured_batch_size: int,
    world_size: int,
    semantics: str,
) -> int:
    if configured_batch_size <= 0 or world_size <= 0:
        raise ValueError("configured_batch_size and world_size must be positive")
    if semantics == "per_device":
        return configured_batch_size
    if semantics != "global":
        raise ValueError("batch_size_semantics must be 'global' or 'per_device'")
    if configured_batch_size % world_size:
        raise ValueError(
            "global batch_size must be divisible by WORLD_SIZE for deterministic DDP semantics"
        )
    return configured_batch_size // world_size


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


def validate_candidate_training_protocol(config: Mapping[str, Any]) -> str:
    protocol = str(config.get("protocol", ""))
    expected = CANDIDATE_TRAINING_PROTOCOLS.get(protocol)
    if expected is None:
        raise NotImplementedError(
            f"Acceptance training protocol {protocol!r} is not approved by this zcp-test version"
        )
    mismatches = [
        f"{field}={config.get(field)!r} (expected {value!r})"
        for field, value in expected.items()
        if config.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            f"Acceptance training protocol {protocol!r} does not match its candidate profile: "
            + "; ".join(mismatches)
        )
    return protocol


def validate_acceptance_training_protocol(config: Mapping[str, Any]) -> str:
    protocol = str(config.get("protocol", ""))
    if protocol in FORMAL_TRAINING_PROTOCOLS:
        return validate_formal_training_protocol(config)
    return validate_candidate_training_protocol(config)


def resolve_acceptance_protocol(
    config: Mapping[str, Any],
    epochs: int,
    data_fraction: float,
) -> str:
    formal_epochs = int(config["epochs"])
    minimum_full_data_epochs = max(1, math.ceil(formal_epochs * 0.01))
    if data_fraction == 1.0 and minimum_full_data_epochs <= epochs <= formal_epochs:
        return "full_data_one_percent_epochs"
    if data_fraction == ONE_PERCENT_DATA_FRACTION and epochs == formal_epochs:
        return "one_percent_data_protocol"
    raise ValueError(
        "Acceptance training must use either full data with at least 1% epochs or "
        "exactly 1% data with the complete epoch schedule"
    )


__all__ = [
    "CANDIDATE_TRAINING_PROTOCOLS",
    "FORMAL_TRAINING_PROTOCOLS",
    "resolve_gradient_accumulation",
    "resolve_acceptance_protocol",
    "resolve_per_device_batch_size",
    "scale_learning_rate",
    "validate_acceptance_training_protocol",
    "validate_candidate_training_protocol",
    "validate_formal_training_protocol",
]
