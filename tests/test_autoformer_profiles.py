from __future__ import annotations

import math

import pytest
import torch

from zcp_test.benchmarks.model_builders import model_builder
from zcp_test.models.autoformer import (
    AZNAS_SCRATCH_PROFILE,
    VITBENCH_AUTOPROX_PROFILE,
    StaticAutoFormer,
)
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.types import Architecture


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def af_zero_specification() -> dict[str, object]:
    return {
        "hidden_dim": 240,
        "depth": 13,
        "num_heads": [3, 3, 4, 4, 3, 4, 4, 4, 4, 3, 4, 4, 3],
        "mlp_ratio": [3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 3.5, 3.5, 3.5, 4.0, 4.0, 3.5],
    }


def test_autoformer_profile_is_required_and_unknown_profiles_fail_closed():
    arguments = {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": [3] * 12,
        "mlp_ratio": [3.5] * 12,
    }
    with pytest.raises(TypeError, match="profile"):
        StaticAutoFormer(**arguments)
    with pytest.raises(ValueError, match="Unknown AutoFormer profile"):
        StaticAutoFormer(profile="autoformer", **arguments)


def test_vitbench_autoprox_af_zero_matches_locked_structure_and_parameter_golden():
    specification = af_zero_specification()
    model = StaticAutoFormer(
        profile=VITBENCH_AUTOPROX_PROFILE,
        num_classes=100,
        embed_dim=int(specification["hidden_dim"]),
        depth=int(specification["depth"]),
        num_heads=specification["num_heads"],
        mlp_ratio=specification["mlp_ratio"],
        drop_path_rate=0.1,
    )

    assert parameter_count(model) == 8_763_340
    assert model.reference_metadata()["source_commit"] == (
        "90ed458eff6948a6f0d23e440a8d21bbec50d091"
    )
    assert model.reference_metadata()["cost_protocol"] is None
    assert all(block.attention.inner_dim == 240 for block in model.blocks)
    assert all(block.attention.qkv.out_features == 720 for block in model.blocks)
    assert all(block.attention.relative_key is None for block in model.blocks)
    assert all(block.attention.relative_value is None for block in model.blocks)
    assert all(block.attention_norm.eps == pytest.approx(1e-6) for block in model.blocks)
    assert model.norm.eps == pytest.approx(1e-6)
    assert [block.drop_path.probability for block in model.blocks] == pytest.approx(
        torch.linspace(0.0, 0.1, 13).tolist()
    )


def test_aznas_profile_uses_static_scale_and_super_depth_schedule():
    model = StaticAutoFormer(
        profile=AZNAS_SCRATCH_PROFILE,
        embed_dim=192,
        depth=12,
        num_heads=[4] * 12,
        mlp_ratio=[3.5] * 12,
        drop_path_rate=0.1,
        super_depth=14,
    )

    attention = model.blocks[0].attention
    assert attention.inner_dim == 4 * 64
    assert attention.scale == pytest.approx(math.pow(192 // 4, -0.5))
    assert attention.relative_key is not None
    assert attention.relative_value is not None
    assert [block.drop_path.probability for block in model.blocks] == pytest.approx(
        torch.linspace(0.0, 0.1, 14).tolist()[:12]
    )
    assert model.blocks[-1].drop_path.probability < 0.1


def test_vitbench_builder_routes_autoformer_to_autoprox_profile():
    load_builtin_spaces()
    space = SPACES.create("autoformer")
    canonical = space.canonicalize(af_zero_specification())
    architecture = Architecture("autoformer", canonical.architecture_id, canonical.spec)

    vitbench_model = model_builder(architecture, "cifar100")
    scratch_model = space.build_model(canonical, 100)

    assert vitbench_model.profile == VITBENCH_AUTOPROX_PROFILE
    assert scratch_model.profile == AZNAS_SCRATCH_PROFILE
    assert vitbench_model.blocks[0].attention.inner_dim == 240
    assert scratch_model.blocks[0].attention.inner_dim == 3 * 64


def test_profiles_reject_cross_source_attention_options():
    arguments = {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": [3] * 12,
        "mlp_ratio": [3.5] * 12,
    }
    with pytest.raises(ValueError, match="does not implement relative position"):
        StaticAutoFormer(
            profile=VITBENCH_AUTOPROX_PROFILE,
            relative_position=True,
            **arguments,
        )
    with pytest.raises(ValueError, match="requires relative position"):
        StaticAutoFormer(
            profile=AZNAS_SCRATCH_PROFILE,
            relative_position=False,
            super_depth=14,
            **arguments,
        )
    with pytest.raises(ValueError, match="requires zero dropout"):
        StaticAutoFormer(
            profile=VITBENCH_AUTOPROX_PROFILE,
            dropout=0.1,
            **arguments,
        )
