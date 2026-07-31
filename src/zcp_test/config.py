from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


TRAIN_PROFILE_KEYS = frozenset(
    {
        "amp",
        "amp_initial_scale",
        "attention_dropout",
        "auto_augment",
        "auxiliary",
        "auxiliary_weight",
        "batch_size",
        "batch_size_semantics",
        "change_qkv",
        "color_distortion",
        "color_jitter",
        "cutmix",
        "cutout_length",
        "dataset",
        "deterministic",
        "drop_path_prob",
        "dropout",
        "epochs",
        "exclude_bias_norm_from_weight_decay",
        "exclude_norm_from_weight_decay",
        "formal_training_blockers",
        "formal_training_ready",
        "global_pool",
        "grad_clip",
        "gradient_accumulation_steps",
        "implementation_commit",
        "implementation_source",
        "init_channels",
        "input_size",
        "label_smoothing",
        "validation_label_smoothing",
        "layers",
        "learning_rate",
        "learning_rate_reference_batch_size",
        "learning_rate_scaling",
        "max_relative_position",
        "minimum_learning_rate",
        "mixup",
        "mixup_mode",
        "mixup_probability",
        "mixup_switch_probability",
        "model_profile",
        "model_init",
        "momentum",
        "nesterov",
        "optimizer",
        "patch_size",
        "protocol",
        "qkv_head_dim",
        "random_erase_count",
        "random_erase_mode",
        "random_erase_probability",
        "reference_global_batch_size",
        "reference_world_size",
        "relative_position",
        "repeated_augmentation",
        "repeated_augmentation_repeats",
        "repeated_augmentation_selected_ratio",
        "repeated_augmentation_selected_round",
        "resize_scale",
        "scheduler",
        "scheduler_gamma",
        "scheduler_step_size",
        "space",
        "target_global_batch_size",
        "test_batch_size",
        "train_interpolation",
        "warmup_epochs",
        "warmup_learning_rate",
        "weight_decay",
    }
)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def reject_unknown_config_keys(
    values: dict[str, Any], allowed: set[str] | frozenset[str], section: str
) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Config section {section!r} contains unknown keys: " + ", ".join(unknown))


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        normalized = json.loads(json.dumps(config, default=str))
        yaml.safe_dump(normalized, handle, sort_keys=True, allow_unicode=True)
    temporary.replace(destination)
