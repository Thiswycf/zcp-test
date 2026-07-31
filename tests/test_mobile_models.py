import math

import pytest
import torch
from torch import nn

from zcp_test.models import mobile
from zcp_test.models.mobile import OFAProxylessMobileNetV2, recalibrate_batch_norm


OFFICIAL_OFA_COMMIT = "f03b2673db313b9167e2a1c2b7a5cad540cc1313"
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


def test_ofa_proxyless_structure_matches_official_export_fixture():
    model = _ofa_proxyless_model()

    assert model.implementation_commit == OFFICIAL_OFA_COMMIT
    assert (model.stem[0].in_channels, model.stem[0].out_channels) == (3, 40)
    assert _block_fixture(model) == OFFICIAL_PROXYLESS_1_3_DEPTH_TWO_BLOCKS
    assert (model.head[0].in_channels, model.head[0].out_channels) == (416, 1664)
    assert (model.classifier.in_features, model.classifier.out_features) == (1664, 1000)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_718_832


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
