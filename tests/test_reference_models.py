import pytest
import torch

from zcp_test.models.autoformer import StaticAutoFormer
from zcp_test.models.mobile import (
    PlainNetMobileNetV2,
    StaticMobileNetV2,
    StaticMobileNetV3,
    recalibrate_batch_norm,
)
from zcp_test.models.pit import StaticPiT
from zcp_test.spaces import SPACES, load_builtin_spaces


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def autoformer(**overrides):
    configuration = {
        "image_size": 32,
        "patch_size": 8,
        "num_classes": 7,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": [2, 2],
        "mlp_ratio": [2.0, 2.0],
        "qkv_head_dim": 8,
    }
    configuration.update(overrides)
    return StaticAutoFormer(**configuration)


def mobile_configuration(**overrides):
    configuration = {
        "num_classes": 7,
        "stem_channels": 8,
        "head_channels": 24,
        "stage_channels": [8, 12],
        "stage_depths": [1, 1],
        "stage_strides": [1, 2],
        "kernel_sizes": [3, 3],
        "expand_ratios": [2, 2],
    }
    configuration.update(overrides)
    return configuration


def test_autoformer_forward_metadata_and_layer_configuration():
    model = autoformer(num_heads=[2, 4], mlp_ratio=[2.0, 3.0]).eval()

    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)
    assert [block.num_heads for block in model.blocks] == [2, 4]
    assert [block.mlp_ratio for block in model.blocks] == [2.0, 3.0]
    metadata = model.reference_metadata()
    assert metadata["model_fidelity"] == "reference_model"
    assert metadata["weight_mode"] == "independent_scratch"
    assert metadata["supports_inherited_supernet"] is False
    assert metadata["architecture"]["num_heads"] == [2, 4]


def test_autoformer_fields_change_parameter_count():
    baseline = parameter_count(autoformer())
    more_heads = parameter_count(autoformer(num_heads=[3, 3]))
    wider_mlp = parameter_count(autoformer(mlp_ratio=[3.0, 3.0]))
    deeper = parameter_count(
        autoformer(depth=3, num_heads=[2, 2, 2], mlp_ratio=[2.0, 2.0, 2.0])
    )

    assert more_heads > baseline
    assert wider_mlp > baseline
    assert deeper > baseline


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_size": 30}, "divisible"),
        ({"depth": 2, "num_heads": [2]}, "one value per layer"),
        ({"num_heads": [0, 2]}, "positive"),
        ({"mlp_ratio": [2.0, 0.0]}, "positive"),
        ({"drop_path_rate": 1.0}, "drop_path_rate"),
    ],
)
def test_autoformer_rejects_illegal_encodings(overrides, message):
    with pytest.raises(ValueError, match=message):
        autoformer(**overrides)


def test_autoformer_rejects_wrong_input_shape():
    model = autoformer()
    with pytest.raises(ValueError, match="32x32"):
        model(torch.randn(1, 3, 24, 24))


def test_pit_reference_model_matches_released_architecture_fields():
    model = StaticPiT(
        image_size=32,
        patch_size=16,
        patch_stride=8,
        num_classes=7,
        base_dim=16,
        depth=[1, 1, 1],
        num_heads=[2, 4, 4],
        mlp_ratio=2,
        drop_path_rate=0.0,
    ).eval()

    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)
    assert model.stage_dimensions == (32, 64, 64)
    assert [len(stage) for stage in model.stages] == [1, 1, 1]
    metadata = model.reference_metadata()
    assert metadata["model_fidelity"] == "reference_model"
    assert metadata["implementation_commit"] == "90ed458"
    assert metadata["architecture"]["num_heads"] == [2, 4, 4]


def test_pit_space_uses_released_three_stage_encoding():
    load_builtin_spaces()
    space = SPACES.create("pit")
    architecture = space.canonicalize(
        {"base_dim": 16, "depth": [2, 8, 4], "num_heads": [2, 4, 4], "mlp_ratio": 6}
    )

    assert space.model_fidelity == "reference_model"
    assert space.sample(5).search_space_id == "pit"
    assert space.mutate(architecture, 7) != architecture
    assert space.crossover(architecture, space.sample(8), 9).search_space_id == "pit"
    model = space.build_model(architecture, 100)
    assert model.image_size == 224
    assert parameter_count(model) == 893_828


@pytest.mark.parametrize(
    "specification",
    [
        {"base_dim": 12, "depth": [2, 8, 4], "num_heads": [2, 4, 4], "mlp_ratio": 6},
        {"base_dim": 16, "depth": [4, 8, 4], "num_heads": [2, 4, 4], "mlp_ratio": 6},
        {"base_dim": 16, "depth": [2, 8], "num_heads": [2, 4, 4], "mlp_ratio": 6},
        {"base_dim": 16, "depth": [2, 8, 4], "num_heads": [2, 3, 4], "mlp_ratio": 6},
        {"base_dim": 16, "depth": [2, 8, 4], "num_heads": [8, 4, 2], "mlp_ratio": 6},
        {"base_dim": 16, "depth": [2, 8, 4], "num_heads": [2, 4, 4], "mlp_ratio": 3},
    ],
)
def test_pit_space_rejects_values_outside_published_search_space(specification):
    load_builtin_spaces()
    with pytest.raises(ValueError, match="PiT"):
        SPACES.create("pit").canonicalize(specification)


@pytest.mark.parametrize("model_type", [PlainNetMobileNetV2, StaticMobileNetV2])
def test_mobile_models_forward_and_static_scratch_metadata(model_type):
    model = model_type(**mobile_configuration()).eval()

    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)
    metadata = model.reference_metadata()
    assert metadata["model_fidelity"] == "reference_model"
    assert metadata["weight_mode"] == "independent_scratch"
    assert metadata["supports_inherited_supernet"] is False


def test_plainnet_and_static_mbconv_have_separate_skip_semantics():
    plainnet = PlainNetMobileNetV2(**mobile_configuration())
    static = StaticMobileNetV2(**mobile_configuration())

    assert plainnet.reference_metadata()["family"] == "plainnet_mbv2"
    assert static.reference_metadata()["family"] == "proxyless_ofa_static_mbv2"
    assert plainnet.skip == (False, False)
    assert static.skip == (True, False)


def test_mobile_kernel_expansion_depth_and_stage_width_change_parameters():
    baseline = parameter_count(StaticMobileNetV2(**mobile_configuration()))
    larger_kernel = parameter_count(
        StaticMobileNetV2(**mobile_configuration(kernel_sizes=[5, 3]))
    )
    larger_expansion = parameter_count(
        StaticMobileNetV2(**mobile_configuration(expand_ratios=[4, 2]))
    )
    deeper = parameter_count(
        StaticMobileNetV2(
            **mobile_configuration(
                stage_depths=[2, 1],
                kernel_sizes=[3, 3, 3],
                expand_ratios=[2, 2, 2],
            )
        )
    )
    wider_stage = parameter_count(
        StaticMobileNetV2(**mobile_configuration(stage_channels=[8, 16]))
    )

    assert larger_kernel > baseline
    assert larger_expansion > baseline
    assert deeper > baseline
    assert wider_stage > baseline


def test_mobile_stride_and_skip_fields_change_computation():
    stride_one = StaticMobileNetV2(
        **mobile_configuration(stage_strides=[1, 1], skip=[True, False])
    ).eval()
    stride_two = StaticMobileNetV2(
        **mobile_configuration(stage_strides=[1, 2], skip=[True, False])
    ).eval()
    inputs = torch.randn(1, 3, 32, 32)
    assert stride_one.blocks(stride_one.stem(inputs)).shape[-1] == 16
    assert stride_two.blocks(stride_two.stem(inputs)).shape[-1] == 8

    with_skip = StaticMobileNetV2(
        **mobile_configuration(skip=[True, False])
    ).eval()
    without_skip = StaticMobileNetV2(
        **mobile_configuration(skip=[False, False])
    ).eval()
    without_skip.load_state_dict(with_skip.state_dict())
    assert not torch.allclose(with_skip(inputs), without_skip(inputs))


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"stage_depths": [1]}, ValueError, "equal length"),
        ({"stage_strides": [1, 3]}, ValueError, "strides"),
        ({"kernel_sizes": [3]}, ValueError, "one value per active block"),
        ({"kernel_sizes": [3, 9]}, ValueError, "kernel sizes"),
        ({"expand_ratios": [2, 0]}, ValueError, "expand ratios"),
        ({"skip": [True]}, ValueError, "one boolean"),
        ({"skip": [True, 1]}, TypeError, "booleans"),
        ({"skip": [True, True]}, ValueError, "cannot use skip"),
    ],
)
def test_static_mobile_rejects_illegal_encodings(overrides, error_type, message):
    with pytest.raises(error_type, match=message):
        StaticMobileNetV2(**mobile_configuration(**overrides))


def test_mobile_rejects_wrong_input_shape():
    model = PlainNetMobileNetV2(**mobile_configuration())
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 1, 32, 32))


def test_registered_autoformer_and_mobile_spaces_build_distinct_reference_models():
    load_builtin_spaces()
    autoformer_space = SPACES.create("autoformer")
    autoformer_architecture = autoformer_space.sample(3)
    autoformer_model = autoformer_space.build_model(autoformer_architecture, 5).eval()
    assert autoformer_space.model_fidelity == "reference_model"
    assert autoformer_model(torch.randn(1, 3, 224, 224)).shape == (1, 5)

    plain_space = SPACES.create("zennas_plainnet_mbv2")
    proxyless_space = SPACES.create("ofa_proxyless_mbv2")
    plain_architecture = plain_space.sample(4)
    proxyless_architecture = proxyless_space.canonicalize(plain_architecture.spec)
    plain = plain_space.build_model(plain_architecture, 5)
    proxyless = proxyless_space.build_model(proxyless_architecture, 5)
    assert plain_space.model_fidelity == proxyless_space.model_fidelity == "reference_model"
    assert plain.reference_metadata()["family"] != proxyless.reference_metadata()["family"]


def test_registered_mobile_space_rejects_inconsistent_block_encoding():
    load_builtin_spaces()
    space = SPACES.create("ofa_proxyless_mbv2")
    architecture = space.sample(8)
    specification = dict(architecture.spec)
    specification["kernel_size"] = list(specification["kernel_size"][:-1])
    with pytest.raises(ValueError, match="active block count"):
        space.canonicalize(specification)


def test_ofa_mbv3_static_subnet_matches_official_active_subnet_fixture():
    model = StaticMobileNetV3(
        num_classes=1000,
        width_mult=1.0,
        stage_depths=[2] * 5,
        kernel_sizes=[3] * 20,
        expand_ratios=[3] * 20,
    ).eval()

    assert parameter_count(model) == 3_410_792
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1000)
    metadata = model.reference_metadata()
    assert metadata["implementation_commit"] == "f03b2673db313b9167e2a1c2b7a5cad540cc1313"
    assert metadata["model_fidelity"] == "reference_model"
    assert metadata["supports_bn_recalibration"] is True


def test_ofa_mbv3_space_uses_official_twenty_block_encoding():
    load_builtin_spaces()
    space = SPACES.create("ofa_mbv3")
    architecture = space.sample(13)

    assert space.model_fidelity == "reference_model"
    assert len(architecture.spec["kernel_size"]) == 20
    assert len(architecture.spec["expand_ratio"]) == 20
    assert len(architecture.spec["depth"]) == 5
    assert space.mutate(architecture, 14) != architecture
    assert space.crossover(architecture, space.sample(15), 16).search_space_id == "ofa_mbv3"
    assert space.build_model(architecture, 7)(torch.randn(1, 3, 64, 64)).shape == (1, 7)


def test_ofa_mbv3_space_rejects_active_only_mbv2_encoding():
    load_builtin_spaces()
    space = SPACES.create("ofa_mbv3")
    with pytest.raises(ValueError, match="20 values"):
        space.canonicalize(
            {
                "kernel_size": [3] * 10,
                "expand_ratio": [3] * 10,
                "depth": [2] * 5,
                "width_mult": 1.0,
                "resolution": 224,
            }
        )


def test_ofa_mbv3_batch_norm_recalibration_restores_model_mode():
    model = StaticMobileNetV3(
        num_classes=7,
        width_mult=1.0,
        stage_depths=[2] * 5,
        kernel_sizes=[3] * 20,
        expand_ratios=[3] * 20,
    ).eval()
    first_batch_norm = next(
        module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)
    )

    processed = recalibrate_batch_norm(
        model,
        [torch.ones(2, 3, 32, 32), torch.full((2, 3, 32, 32), 2.0)],
        device="cpu",
        max_batches=1,
    )

    assert processed == 1
    assert model.training is False
    assert first_batch_norm.training is False
    assert first_batch_norm.momentum == 0.1
    assert first_batch_norm.num_batches_tracked.item() == 1
