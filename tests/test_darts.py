import pytest
import torch

from zcp_test.models import (
    OPS,
    AuxiliaryHeadCIFAR,
    AuxiliaryHeadImageNet,
    NetworkCIFAR,
    NetworkImageNet,
    drop_path,
)
from zcp_test.spaces.darts import (
    DARTS_CONCAT,
    DARTS_PRIMITIVES,
    DartsSpace,
    canonicalize_genotype,
    genotype_from_spec,
    genotype_to_spec,
    get_darts_profile,
)


def _assert_valid_genotype(specification):
    for cell in ("normal", "reduce"):
        assert len(specification[cell]) == 8
        for node in range(4):
            edges = specification[cell][2 * node : 2 * node + 2]
            assert len({edge[1] for edge in edges}) == 2
            assert all(edge[0] in DARTS_PRIMITIVES for edge in edges)
            assert all(0 <= edge[1] < node + 2 for edge in edges)
        assert specification[f"{cell}_concat"] == list(DARTS_CONCAT)


def test_genotype_canonicalization_is_stable():
    specification = DartsSpace().sample(4).spec
    reordered = {key: list(value) for key, value in specification.items()}
    for cell in ("normal", "reduce"):
        reordered[cell] = [
            edge
            for node in range(4)
            for edge in reversed(reordered[cell][2 * node : 2 * node + 2])
        ]
    first = DartsSpace().canonicalize(specification)
    second = DartsSpace().canonicalize(reordered)
    assert first.architecture_id == second.architecture_id
    assert first.spec == second.spec
    assert genotype_to_spec(genotype_from_spec(first.spec)) == first.spec


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("normal", [["skip_connect", 0]], ValueError),
        ("normal_concat", [2, 3], ValueError),
    ],
)
def test_genotype_rejects_nonstandard_shape(field, value, error):
    specification = dict(DartsSpace().sample(1).spec)
    specification[field] = value
    with pytest.raises(error):
        canonicalize_genotype(specification)


def test_genotype_rejects_duplicate_inputs_and_unknown_operations():
    specification = dict(DartsSpace().sample(2).spec)
    specification["normal"] = [list(edge) for edge in specification["normal"]]
    specification["normal"][1][1] = specification["normal"][0][1]
    with pytest.raises(ValueError, match="distinct inputs"):
        canonicalize_genotype(specification)
    specification = dict(DartsSpace().sample(2).spec)
    specification["reduce"] = [list(edge) for edge in specification["reduce"]]
    specification["reduce"][0][0] = "none"
    with pytest.raises(ValueError, match="operation"):
        canonicalize_genotype(specification)


def test_sample_mutation_and_crossover_are_seeded_and_valid():
    space = DartsSpace()
    left = space.sample(10)
    right = space.sample(11)
    assert left == space.sample(10)
    mutated = space.mutate(left, 12)
    assert mutated == space.mutate(left, 12)
    assert mutated.architecture_id != left.architecture_id
    child = space.crossover(left, right, 13)
    assert child == space.crossover(left, right, 13)
    _assert_valid_genotype(left.spec)
    _assert_valid_genotype(mutated.spec)
    _assert_valid_genotype(child.spec)


@pytest.mark.parametrize("operation_name", sorted(OPS))
def test_operations_preserve_expected_reduction_shape(operation_name):
    operation = OPS[operation_name](8, 2, True)
    output = operation(torch.randn(2, 8, 16, 16))
    assert output.shape == (2, 8, 8, 8)


def test_drop_path_is_disabled_for_evaluation_and_scales_training_paths():
    inputs = torch.ones(64, 3, 2, 2)
    assert drop_path(inputs, 0.5, training=False) is inputs
    torch.manual_seed(3)
    output = drop_path(inputs, 0.5, training=True)
    assert set(output.unique().tolist()) == {0.0, 2.0}
    with pytest.raises(ValueError):
        drop_path(inputs, 1.0, training=True)


def test_auxiliary_heads_produce_class_logits():
    cifar_head = AuxiliaryHeadCIFAR(16, 10).eval()
    imagenet_head = AuxiliaryHeadImageNet(16, 1000).eval()
    assert cifar_head(torch.randn(2, 16, 8, 8)).shape == (2, 10)
    assert imagenet_head(torch.randn(2, 16, 7, 7)).shape == (2, 1000)


def test_cifar_network_supports_auxiliary_logits_and_drop_path():
    genotype = genotype_from_spec(DartsSpace().sample(20).spec)
    model = NetworkCIFAR(4, 10, 8, True, genotype, drop_path_prob=0.2).train()
    logits, auxiliary_logits = model(torch.randn(2, 3, 32, 32), return_auxiliary=True)
    assert logits.shape == (2, 10)
    assert auxiliary_logits is not None
    assert auxiliary_logits.shape == (2, 10)
    assert model.auxiliary_logits is auxiliary_logits


def test_imagenet_network_forward_and_profile_building():
    space = DartsSpace()
    genotype = genotype_from_spec(space.sample(21).spec)
    model = NetworkImageNet(4, 1000, 2, False, genotype).eval()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1000)
    assert get_darts_profile("imagenet1k").network == "imagenet"
    assert space.resolve_profile(10).name == "cifar10"
    assert space.resolve_profile(100).name == "cifar100"
    assert space.resolve_profile(1000).name == "imagenet"
    built = space.build_model(space.sample(22), 10)
    assert isinstance(built, NetworkCIFAR)
    assert built.drop_path_prob == pytest.approx(0.2)
