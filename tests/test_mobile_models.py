import math

import pytest
import torch
from torch import nn

from zcp_test.models import mobile
from zcp_test.models.mobile import OFAProxylessMobileNetV2, recalibrate_batch_norm
from zcp_test.models.plainnet import (
    AZNAS_COMMIT,
    AZPlainNetMobileNetV2,
    INITIAL_STRUCTURE,
    canonical_plainnet_structure,
    parse_plainnet_structure,
)
from zcp_test.spaces import SPACES, load_builtin_spaces


OFFICIAL_OFA_COMMIT = "f03b2673db313b9167e2a1c2b7a5cad540cc1313"
OFFICIAL_PROXYLESS_COMMIT = "b23018c9c369d22931f7422b71ca6a7eaa354c46"
OFFICIAL_PROXYLESS_EXACT_MACS = 265_526_256
OFFICIAL_OFA_FLOAT32_REPORTED_OPS = 265_526_240
OFFICIAL_PROXYLESS_CONV_MACS = 263_862_256
OFFICIAL_PROXYLESS_LINEAR_MACS = 1_664_000
OFFICIAL_AZNAS_PLAINNET_PARAMETERS = {False: 2_824_264, True: 3_579_232}
OFFICIAL_AZNAS_PLAINNET_MACS = {False: 159_334_080, True: 160_081_728}
OFFICIAL_PROXYLESS_1_3_DEPTH_TWO_BLOCKS = (
    (40, None, 24, 3, 1, False),
    (24, 72, 32, 3, 2, False),
    (32, 96, 32, 3, 1, True),
    (32, 96, 56, 3, 2, False),
    (56, 168, 56, 3, 1, True),
    (56, 168, 104, 3, 2, False),
    (104, 312, 104, 3, 1, True),
    (104, 312, 128, 3, 1, False),
    (128, 384, 128, 3, 1, True),
    (128, 384, 248, 3, 2, False),
    (248, 744, 248, 3, 1, True),
    (248, 744, 416, 3, 1, False),
)


def _ofa_proxyless_model() -> OFAProxylessMobileNetV2:
    return OFAProxylessMobileNetV2(
        num_classes=1000,
        width_mult=1.3,
        stage_depths=[2] * 5,
        kernel_sizes=[3] * 21,
        expand_ratios=[3] * 21,
        image_size=224,
    )


def _block_fixture(model: OFAProxylessMobileNetV2):
    first_depthwise, first_pointwise = model.first_block.layers
    blocks = [
        (
            first_depthwise[0].in_channels,
            None,
            first_pointwise[0].out_channels,
            first_depthwise[0].kernel_size[0],
            first_depthwise[0].stride[0],
            model.first_block.use_residual,
        )
    ]
    for stage in model.stages:
        for block in stage:
            inverted, depthwise, pointwise = block.layers
            blocks.append(
                (
                    inverted[0].in_channels,
                    inverted[0].out_channels,
                    pointwise[0].out_channels,
                    depthwise[0].kernel_size[0],
                    depthwise[0].stride[0],
                    block.use_residual,
                )
            )
    return tuple(blocks)


def _official_conv_linear_macs(model: nn.Module):
    layer_macs = {}
    handles = []

    def count_layer(module, _, output):
        if isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            layer_macs[module] = (
                module.in_channels * output.numel() * kernel_ops // module.groups
            )
        elif isinstance(module, nn.Linear):
            layer_macs[module] = module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count_layer))
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, 3, 224, 224))
    finally:
        for handle in handles:
            handle.remove()

    conv_macs = sum(
        macs for module, macs in layer_macs.items() if isinstance(module, nn.Conv2d)
    )
    linear_macs = sum(
        macs for module, macs in layer_macs.items() if isinstance(module, nn.Linear)
    )
    ofa_float32_total = torch.zeros(1)
    for module in model.modules():
        if module in layer_macs:
            ofa_float32_total += torch.zeros(1).fill_(layer_macs[module])
    return conv_macs, linear_macs, int(ofa_float32_total.item())


def test_ofa_proxyless_structure_matches_official_export_fixture():
    model = _ofa_proxyless_model()

    assert model.implementation_commit == OFFICIAL_OFA_COMMIT
    assert (model.stem[0].in_channels, model.stem[0].out_channels) == (3, 40)
    assert _block_fixture(model) == OFFICIAL_PROXYLESS_1_3_DEPTH_TWO_BLOCKS
    assert (model.head[0].in_channels, model.head[0].out_channels) == (416, 1664)
    assert (model.classifier.in_features, model.classifier.out_features) == (1664, 1000)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_718_832


def test_ofa_proxyless_224_official_mac_golden():
    model = _ofa_proxyless_model()

    conv_macs, linear_macs, ofa_reported_ops = _official_conv_linear_macs(model)

    assert model.implementation_commit == OFFICIAL_OFA_COMMIT
    assert OFFICIAL_PROXYLESS_COMMIT == "b23018c9c369d22931f7422b71ca6a7eaa354c46"
    assert conv_macs == OFFICIAL_PROXYLESS_CONV_MACS
    assert linear_macs == OFFICIAL_PROXYLESS_LINEAR_MACS
    assert conv_macs + linear_macs == OFFICIAL_PROXYLESS_EXACT_MACS
    assert ofa_reported_ops == OFFICIAL_OFA_FLOAT32_REPORTED_OPS
    assert 2 * OFFICIAL_PROXYLESS_EXACT_MACS == 531_052_512


@pytest.mark.parametrize("use_se", [False, True])
def test_aznas_plainnet_initial_structure_golden(use_se):
    model = AZPlainNetMobileNetV2(INITIAL_STRUCTURE, use_se=use_se).eval()

    conv_macs, linear_macs, _ = _official_conv_linear_macs(model)

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        OFFICIAL_AZNAS_PLAINNET_PARAMETERS[use_se]
    )
    assert conv_macs + linear_macs == OFFICIAL_AZNAS_PLAINNET_MACS[use_se]
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 1000)
    assert model.reference_metadata()["implementation_commit"] == AZNAS_COMMIT


def test_plainnet_structure_parser_is_canonical_and_rejects_code():
    blocks = parse_plainnet_structure("\n" + INITIAL_STRUCTURE + "\n")

    assert canonical_plainnet_structure(blocks) == INITIAL_STRUCTURE
    with pytest.raises(ValueError, match="Unsupported PlainNet block type"):
        parse_plainnet_structure("eval(1,2,1,1)")
    with pytest.raises(ValueError, match="adjacent block channels"):
        parse_plainnet_structure(
            "SuperConvK3BNRELU(3,8,2,1)SuperResIDWE6K3(16,32,2,8,1)"
        )


def test_plainnet_space_sample_mutate_crossover_and_training_model():
    load_builtin_spaces()
    space = SPACES.create("zennas_plainnet_mbv2")
    left = space.sample(10)
    right = space.mutate(left, 11)
    child = space.crossover(left, right, 12)

    for architecture in (left, right, child):
        assert space.canonicalize(architecture.spec) == architecture
        assert parse_plainnet_structure(architecture.spec["structure"])[-1].out_channels == 2048
    assert left.architecture_id != right.architecture_id
    search_model = space.build_model(child, 7)
    training_model = space.build_training_model(
        child,
        7,
        {"model_init": "custom_kaiming", "use_se": True, "bn_momentum": 0.01},
    )
    assert search_model.reference_metadata()["use_se"] is False
    assert training_model.reference_metadata()["use_se"] is True
    assert training_model.reference_metadata()["bn_momentum"] == pytest.approx(0.01)


def test_ofa_proxyless_constructor_applies_official_he_fout(monkeypatch):
    calls = []
    initialize = mobile._initialize_he_fout

    def tracked_initialize(model):
        calls.append(model)
        initialize(model)

    monkeypatch.setattr(mobile, "_initialize_he_fout", tracked_initialize)
    model = _ofa_proxyless_model()

    assert calls == [model]
    assert torch.count_nonzero(model.classifier.bias) == 0


def test_official_he_fout_matches_reference_initialization_exactly():
    model = _ofa_proxyless_model()
    random_state = torch.random.get_rng_state()

    mobile._initialize_he_fout(model)
    torch.random.set_rng_state(random_state)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            expected = torch.empty_like(module.weight).normal_(0, math.sqrt(2.0 / fan_out))
            assert torch.equal(module.weight, expected)
            assert module.bias is None or torch.count_nonzero(module.bias) == 0
        elif isinstance(module, nn.BatchNorm2d):
            assert torch.count_nonzero(module.weight - 1) == 0
            assert torch.count_nonzero(module.bias) == 0
        elif isinstance(module, nn.Linear):
            bound = 1.0 / math.sqrt(module.weight.shape[1])
            expected = torch.empty_like(module.weight).uniform_(-bound, bound)
            assert torch.equal(module.weight, expected)
            assert module.bias is None or torch.count_nonzero(module.bias) == 0


def test_batch_norm_recalibration_uses_official_weighted_biased_variance():
    model = nn.Sequential(nn.BatchNorm2d(1, momentum=0.3)).eval()
    batch_norm = model[0]
    batch_norm.running_mean.fill_(100)
    batch_norm.running_var.fill_(200)
    first = torch.stack((torch.zeros(1, 2, 2), torch.full((1, 2, 2), 2.0)))
    second = torch.full((1, 1, 2, 2), 5.0)

    processed = recalibrate_batch_norm(model, [first, second], device="cpu")

    assert processed == 2
    assert batch_norm.running_mean.item() == pytest.approx(7 / 3)
    assert batch_norm.running_var.item() == pytest.approx(2 / 3)
    assert batch_norm.num_batches_tracked.item() == 2
    assert batch_norm.momentum == 0.3
    assert model.training is False
