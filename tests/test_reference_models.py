import hashlib

import pytest
import torch

from zcp_test.models.autoformer import StaticAutoFormer
from zcp_test.models.mobile import (
    OFAProxylessMobileNetV2,
    PlainNetMobileNetV2,
    StaticMobileNetV2,
    StaticMobileNetV3,
    load_ofa_proxyless_inherited_weights,
    recalibrate_batch_norm,
)
from zcp_test.models.pit import PitAttention, StaticPiT
from zcp_test.spaces import SPACES, load_builtin_spaces


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _store_batch_norm(state, prefix, batch_norm):
    state[f"{prefix}.weight"] = batch_norm.weight.detach().clone()
    state[f"{prefix}.bias"] = batch_norm.bias.detach().clone()
    state[f"{prefix}.running_mean"] = batch_norm.running_mean.detach().clone()
    state[f"{prefix}.running_var"] = batch_norm.running_var.detach().clone()
    state[f"{prefix}.num_batches_tracked"] = (
        batch_norm.num_batches_tracked.detach().clone()
    )


def _fake_ofa_proxyless_checkpoint(model):
    state = {"first_conv.conv.weight": model.stem[0].weight.detach().clone()}
    _store_batch_norm(state, "first_conv.bn", model.stem[1])
    first_depthwise, first_pointwise = model.first_block.layers
    state["blocks.0.mobile_inverted_conv.depth_conv.conv.weight"] = (
        first_depthwise[0].weight.detach().clone()
    )
    _store_batch_norm(
        state, "blocks.0.mobile_inverted_conv.depth_conv.bn", first_depthwise[1]
    )
    state["blocks.0.mobile_inverted_conv.point_linear.conv.weight"] = (
        first_pointwise[0].weight.detach().clone()
    )
    _store_batch_norm(
        state, "blocks.0.mobile_inverted_conv.point_linear.bn", first_pointwise[1]
    )
    for stage_index, stage in enumerate(model.stages):
        position_start, _ = model._stage_positions[stage_index]
        for block_offset, block in enumerate(stage):
            position = position_start + block_offset
            prefix = f"blocks.{position + 1}.mobile_inverted_conv"
            inverted, depthwise, pointwise = block.layers
            state[f"{prefix}.inverted_bottleneck.conv.conv.weight"] = (
                inverted[0].weight.detach().clone()
            )
            _store_batch_norm(
                state, f"{prefix}.inverted_bottleneck.bn.bn", inverted[1]
            )
            state[f"{prefix}.depth_conv.conv.conv.weight"] = (
                depthwise[0].weight.detach().clone()
            )
            _store_batch_norm(state, f"{prefix}.depth_conv.bn.bn", depthwise[1])
            state[f"{prefix}.point_linear.conv.conv.weight"] = (
                pointwise[0].weight.detach().clone()
            )
            _store_batch_norm(state, f"{prefix}.point_linear.bn.bn", pointwise[1])
    state["feature_mix_layer.conv.weight"] = model.head[0].weight.detach().clone()
    _store_batch_norm(state, "feature_mix_layer.bn", model.head[1])
    state["classifier.linear.weight"] = model.classifier.weight.detach().clone()
    state["classifier.linear.bias"] = model.classifier.bias.detach().clone()
    return {"state_dict": state}


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
    (
        "embed_dim",
        "num_heads",
        "mlp_ratio",
        "expected_parameters",
        "expected_official_complexity_ops",
    ),
    [
        pytest.param(
            192,
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3],
            [3.5, 3.5, 3.0, 3.5, 3.0, 3.0, 4.0, 4.0, 3.5, 4.0, 3.5, 4.0, 3.5],
            5_867_944,
            1_344_034_362,
            id="cream-b799630-autoformer-t",
        ),
        pytest.param(
            384,
            [6, 6, 5, 7, 5, 5, 5, 6, 6, 7, 7, 6, 7],
            [3.0, 3.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 4.0],
            22_891_432,
            4_892_144_730,
            id="cream-b799630-autoformer-s",
        ),
        pytest.param(
            576,
            [9, 9, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 9, 10],
            [3.5, 3.5, 4.0, 3.5, 4.0, 3.5, 3.5, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.5],
            53_691_688,
            11_236_929_660,
            id="cream-b799630-autoformer-b",
        ),
        pytest.param(
            192,
            [4, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4],
            [3.5, 3.5, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0],
            5_921_032,
            1_380_128_376,
            id="az-nas-5e6683-tiny",
        ),
        pytest.param(
            384,
            [7, 7, 6, 6, 5, 6, 6, 6, 7, 7, 5, 5, 6, 7],
            [3.0, 3.0, 3.0, 4.0, 3.5, 3.0, 3.0, 3.5, 4.0, 3.0, 3.0, 3.0, 3.0, 4.0],
            22_951_144,
            4_943_341_788,
            id="az-nas-5e6683-small",
        ),
        pytest.param(
            528,
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10],
            [3.0, 3.5, 3.5, 3.5, 3.0, 3.5, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 3.5, 3.0, 4.0, 4.0],
            53_710_768,
            11_386_130_712,
            id="az-nas-5e6683-base",
        ),
    ],
)
def test_autoformer_official_subnet_parameter_counts(
    embed_dim,
    num_heads,
    mlp_ratio,
    expected_parameters,
    expected_official_complexity_ops,
):
    model = StaticAutoFormer(
        embed_dim=embed_dim,
        depth=len(num_heads),
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    assert [block.num_heads for block in model.blocks] == num_heads
    assert [block.mlp_ratio for block in model.blocks] == mlp_ratio
    assert parameter_count(model) == expected_parameters
    assert model.official_complexity_ops() == expected_official_complexity_ops
    assert model.reference_metadata()["cost_protocol"] == {
        "name": "cream-autoformer-get-complexity",
        "source_commit": "b799630a29995163f282b15e2f38701160272fd1",
        "official_complexity_ops": expected_official_complexity_ops,
        "generic_flops": False,
    }


def test_autoformer_official_complexity_is_not_relabelled_as_thop_macs():
    from thop import profile

    model = StaticAutoFormer(
        embed_dim=192,
        depth=12,
        num_heads=[4, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4],
        mlp_ratio=[3.5, 3.5, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0],
    ).eval()
    thop_macs, thop_parameters = profile(
        model,
        inputs=(torch.randn(1, 3, 224, 224),),
        verbose=False,
    )

    assert 0 < thop_macs < model.official_complexity_ops()
    assert thop_parameters < parameter_count(model)
    assert model.reference_metadata()["cost_protocol"]["generic_flops"] is False


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
    assert metadata["model_fidelity"] == "reference_topology_pytorch_port"
    assert metadata["implementation_commit"] == "90ed458"
    assert metadata["architecture"]["num_heads"] == [2, 4, 4]


def test_pit_space_uses_released_three_stage_encoding():
    load_builtin_spaces()
    space = SPACES.create("pit")
    architecture = space.canonicalize(
        {"base_dim": 16, "depth": [2, 8, 4], "num_heads": [2, 4, 4], "mlp_ratio": 6}
    )

    assert space.model_fidelity == "reference_topology_pytorch_port"
    assert space.sample(5).search_space_id == "pit"
    assert space.mutate(architecture, 7) != architecture
    assert space.crossover(architecture, space.sample(8), 9).search_space_id == "pit"
    model = space.build_model(architecture, 100)
    assert model.image_size == 224
    assert parameter_count(model) == 893_828


def test_pit_port_matches_upstream_structural_and_complexity_fixture():
    from thop import profile

    model = StaticPiT(
        image_size=224,
        patch_size=16,
        patch_stride=8,
        num_classes=100,
        base_dim=16,
        depth=[2, 8, 4],
        num_heads=[2, 4, 4],
        mlp_ratio=6,
        drop_path_rate=0.1,
    ).eval()

    blocks = [block for stage in model.stages for block in stage]
    assert all(isinstance(block.attention, PitAttention) for block in blocks)
    assert [block.drop_path.probability for block in blocks] == pytest.approx(
        [0.1 * index / 14 for index in range(14)]
    )
    assert {module.eps for module in model.modules() if isinstance(module, torch.nn.LayerNorm)} == {
        1e-6
    }
    macs, _ = profile(model, inputs=(torch.zeros(1, 3, 224, 224),), verbose=False)
    assert int(macs) == 159_665_472


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


@pytest.mark.parametrize(
    ("model_type", "expected_fidelity"),
    [
        (PlainNetMobileNetV2, "proxy_approximation"),
        (StaticMobileNetV2, "reference_model"),
    ],
)
def test_mobile_models_forward_and_static_scratch_metadata(
    model_type, expected_fidelity
):
    model = model_type(**mobile_configuration()).eval()

    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)
    metadata = model.reference_metadata()
    assert metadata["model_fidelity"] == expected_fidelity
    assert metadata["weight_mode"] == "independent_scratch"
    assert metadata["supports_inherited_supernet"] is False


def test_plainnet_and_static_mbconv_have_separate_skip_semantics():
    plainnet = PlainNetMobileNetV2(**mobile_configuration())
    static = StaticMobileNetV2(**mobile_configuration())

    assert plainnet.reference_metadata()["family"] == "plainnet_mbv2"
    assert plainnet.reference_metadata()["model_fidelity"] == "proxy_approximation"
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


def test_registered_autoformer_and_mobile_spaces_expose_distinct_fidelity():
    load_builtin_spaces()
    autoformer_space = SPACES.create("autoformer")
    autoformer_architecture = autoformer_space.sample(3)
    autoformer_model = autoformer_space.build_model(autoformer_architecture, 5).eval()
    assert autoformer_space.model_fidelity == "reference_model"
    assert autoformer_space.implementation_commit == "b799630a29995163f282b15e2f38701160272fd1"
    assert autoformer_model(torch.randn(1, 3, 224, 224)).shape == (1, 5)

    plain_space = SPACES.create("zennas_plainnet_mbv2")
    proxyless_space = SPACES.create("ofa_proxyless_mbv2")
    plain_architecture = plain_space.sample(4)
    proxyless_architecture = proxyless_space.sample(4)
    plain = plain_space.build_model(plain_architecture, 5)
    proxyless = proxyless_space.build_model(proxyless_architecture, 5)
    assert plain_space.model_fidelity == "proxy_approximation"
    assert proxyless_space.model_fidelity == "reference_model"
    assert plain.reference_metadata()["family"] != proxyless.reference_metadata()["family"]


def test_registered_mobile_space_rejects_inconsistent_block_encoding():
    load_builtin_spaces()
    space = SPACES.create("ofa_proxyless_mbv2")
    architecture = space.sample(8)
    specification = dict(architecture.spec)
    specification["kernel_size"] = list(specification["kernel_size"][:-1])
    with pytest.raises(ValueError, match="21 positional"):
        space.canonicalize(specification)


@pytest.mark.parametrize(
    ("width_mult", "expected_parameters"),
    [(1.0, 2_500_632), (1.3, 3_718_832)],
)
def test_ofa_proxyless_static_subnet_matches_official_active_subnet_fixture(
    width_mult, expected_parameters
):
    model = OFAProxylessMobileNetV2(
        num_classes=1000,
        width_mult=width_mult,
        stage_depths=[2] * 5,
        kernel_sizes=[3] * 21,
        expand_ratios=[3] * 21,
        image_size=224,
    ).eval()

    assert parameter_count(model) == expected_parameters
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1000)
    metadata = model.reference_metadata()
    assert metadata["implementation_commit"] == "f03b2673db313b9167e2a1c2b7a5cad540cc1313"
    assert metadata["positional_encoding"] == "21_dynamic_blocks_5x4_plus_fixed_final"


def test_ofa_proxyless_space_uses_official_supernet_width_and_positions():
    load_builtin_spaces()
    space = SPACES.create("ofa_proxyless_mbv2")
    architecture = space.sample(19)

    assert architecture.spec["width_mult"] == 1.3
    assert len(architecture.spec["kernel_size"]) == 21
    assert len(architecture.spec["expand_ratio"]) == 21
    assert 128 <= architecture.spec["resolution"] <= 224
    assert architecture.spec["resolution"] % 4 == 0
    with pytest.raises(ValueError, match="width multiplier 1.3"):
        space.canonicalize({**architecture.spec, "width_mult": 1.0})


def test_ofa_proxyless_inherited_loader_requires_trust_and_records_provenance(tmp_path):
    configuration = {
        "num_classes": 1000,
        "width_mult": 0.1,
        "stage_depths": [2] * 5,
        "kernel_sizes": [7] * 21,
        "expand_ratios": [3] * 21,
        "image_size": 128,
    }
    source = OFAProxylessMobileNetV2(**configuration).eval()
    checkpoint = tmp_path / "official-ofa.pt"
    torch.save(_fake_ofa_proxyless_checkpoint(source), checkpoint)
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    target = OFAProxylessMobileNetV2(**configuration).eval()
    for parameter in target.parameters():
        parameter.data.zero_()

    with pytest.raises(PermissionError, match="trusted=True"):
        load_ofa_proxyless_inherited_weights(target, checkpoint)
    with pytest.raises(ValueError, match="SHA-256"):
        load_ofa_proxyless_inherited_weights(
            target, checkpoint, trusted=True, expected_sha256="0" * 64
        )
    provenance = load_ofa_proxyless_inherited_weights(
        target, checkpoint, trusted=True, expected_sha256=checksum
    )

    assert provenance["protocol"] == "ofa_inherited_supernet"
    assert provenance["bn_recalibration_required"] is True
    assert target.reference_metadata()["weight_mode"] == "inherited_supernet"
    assert target.reference_metadata()["checkpoint_provenance"] == provenance
    source_state = source.state_dict()
    target_state = target.state_dict()
    assert source_state.keys() == target_state.keys()
    assert all(torch.equal(source_state[key], target_state[key]) for key in source_state)


def test_ofa_proxyless_bn_recalibration_is_recorded():
    model = OFAProxylessMobileNetV2(
        num_classes=1000,
        width_mult=0.1,
        stage_depths=[2] * 5,
        kernel_sizes=[7] * 21,
        expand_ratios=[3] * 21,
        image_size=128,
    ).eval()

    processed = recalibrate_batch_norm(
        model, [torch.randn(2, 3, 32, 32)], device="cpu"
    )

    assert processed == 1
    assert model.reference_metadata()["bn_recalibrated_batches"] == 1


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
