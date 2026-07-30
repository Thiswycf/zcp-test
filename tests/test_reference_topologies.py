import torch

from zcp_test.benchmarks.model_builders import model_builder
from zcp_test.models.nb201 import TinyNetwork, build_nats_sss, parse_architecture
from zcp_test.models.transnas import TransNasNetwork, macro_codes
from zcp_test.types import Architecture


NB201_SPEC = (
    "|nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+"
    "|avg_pool_3x3~0|none~1|nor_conv_3x3~2|"
)


def test_nb201_reference_macro_has_reductions_and_logits():
    architecture = Architecture("nb201_topology", "a", {"architecture": NB201_SPEC})
    model = model_builder(architecture, "cifar10")
    assert isinstance(model, TinyNetwork)
    assert len(model.cells) == 17
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_nats_sss_stage_channels_change_model():
    narrow = build_nats_sss([8, 8, 8, 8, 8], 10)
    wide = build_nats_sss([16, 24, 32, 40, 48], 10)
    narrow_parameters = sum(parameter.numel() for parameter in narrow.parameters())
    wide_parameters = sum(parameter.numel() for parameter in wide.parameters())
    assert len(narrow.cells) == len(wide.cells) == 5
    assert wide_parameters > narrow_parameters
    assert wide(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_nb201_parser_rejects_invalid_sources():
    invalid = "|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|+|none~0|none~1|none~2|"
    try:
        parse_architecture(invalid)
    except ValueError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("invalid NB201 source was accepted")


def test_transnas_spaces_match_published_cardinalities_and_forward():
    assert len(macro_codes()) == 3256
    micro = TransNasNetwork("64-41414-3_33_333", 47)
    macro = TransNasNetwork("64-4111-basic", 75)
    inputs = torch.randn(2, 3, 32, 32)
    assert micro(inputs).shape == (2, 47)
    assert macro(inputs).shape == (2, 75)
