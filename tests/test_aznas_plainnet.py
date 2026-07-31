import math

import pytest
import torch

from zcp_test.models.plainnet import (
    AZNAS_COMMIT,
    INITIAL_STRUCTURE,
    OFFICIAL_COMPLEXITY_VERSION,
    AZPlainNetMobileNetV2,
)
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.az_nas import PLAINNET_COMPONENTS, plainnet_components
from zcp_test.proxies.evaluator import evaluate_proxy


TINY_STRUCTURE = (
    "SuperConvK3BNRELU(3,8,2,1)"
    "SuperResIDWE1K3(8,16,2,8,1)"
    "SuperConvK1BNRELU(16,16,1,1)"
)


def _hook_count(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


@pytest.mark.parametrize(
    ("use_se", "expected"),
    [(False, 162_396_776.0), (True, 164_511_512.0)],
)
def test_initial_structure_official_complexity_golden(use_se, expected):
    model = AZPlainNetMobileNetV2(INITIAL_STRUCTURE, use_se=use_se)
    metadata = model.reference_metadata()

    assert model.official_complexity_ops(224) == expected
    assert metadata["implementation_commit"] == AZNAS_COMMIT
    assert metadata["official_complexity_version"] == OFFICIAL_COMPLEXITY_VERSION


def test_relu_feature_shapes_and_temporary_hooks_are_cleaned():
    model = AZPlainNetMobileNetV2(TINY_STRUCTURE, num_classes=3).eval()
    inputs = torch.randn(2, 3, 16, 16)
    hooks_before = _hook_count(model)

    features, logits = model.extract_layer_features_and_logit(inputs)

    assert [tuple(feature.shape) for feature in features] == [
        (2, 8, 8, 8),
        (2, 8, 8, 8),
        (2, 8, 4, 4),
        (2, 8, 4, 4),
        (2, 16, 4, 4),
        (2, 16, 4, 4),
        (2, 16, 4, 4),
        (2, 16, 4, 4),
    ]
    assert logits.shape == (2, 3)
    assert all(feature.requires_grad for feature in features)
    assert _hook_count(model) == hooks_before

    with pytest.raises(ValueError, match="shape"):
        model.extract_layer_features_and_logit(torch.randn(2, 1, 16, 16))
    assert _hook_count(model) == hooks_before


@pytest.mark.parametrize("use_se", [False, True])
def test_initial_structure_extracts_26_upstream_relu_features(use_se):
    model = AZPlainNetMobileNetV2(INITIAL_STRUCTURE, use_se=use_se).eval()

    with torch.no_grad():
        features, logits = model.extract_layer_features_and_logit(
            torch.randn(1, 3, 32, 32)
        )

    assert len(features) == 26
    assert logits.shape == (1, 1000)


def test_plainnet_formula_components_are_finite_and_registered():
    torch.manual_seed(31)
    model = AZPlainNetMobileNetV2(TINY_STRUCTURE, num_classes=3).eval()
    inputs = torch.randn(2, 3, 16, 16)

    torch.manual_seed(37)
    components = plainnet_components(model, inputs)
    torch.manual_seed(37)
    result = evaluate_proxy("az_nas_plainnet", model, inputs, model_family="cnn")

    load_builtin_proxies()
    capability = PROXIES.create("az_nas_plainnet").capability
    assert tuple(components) == PLAINNET_COMPONENTS
    assert all(math.isfinite(value) for value in components.values())
    assert components["complexity"] == model.official_complexity_ops(16)
    assert result.status.value == "ok"
    assert result.components == pytest.approx(components)
    assert result.score == pytest.approx(components["expressivity"])
    assert capability.model_families == ("cnn",)
    assert capability.requires_labels is False
    assert capability.components == PLAINNET_COMPONENTS
    assert capability.primary_component == "expressivity"
    assert capability.implementation_fidelity == "paper_formula_port_stabilized"


def test_plainnet_proxy_rejects_other_cnn_models():
    result = evaluate_proxy(
        "az_nas_plainnet",
        torch.nn.Linear(3, 2),
        torch.randn(2, 3),
        model_family="cnn",
    )

    assert result.status.value == "unsupported"
    assert result.primary_component == "expressivity"
    assert result.error_message == "AZ-NAS PlainNet requires AZPlainNetMobileNetV2"
