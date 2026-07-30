from pathlib import Path

import pytest
import torch

from zcp_test.artifacts import read_jsonl
from zcp_test.config import load_config
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
from zcp_test.training.protocols import (
    resolve_per_device_batch_size,
    validate_formal_training_protocol,
)
from zcp_test.training.trainer import TrainingConfig, train_model


ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize(
    "name",
    ["darts_cifar10.yaml", "darts_cifar100.yaml", "darts_imagenet.yaml"],
)
def test_darts_formal_profiles_match_pinned_recipes(name):
    config = load_config(ROOT / "configs" / "training" / name)
    assert validate_formal_training_protocol(config) == config["protocol"]
    assert config["implementation_commit"] == "f276dd346a09ae3160f8e3aca5c7b193fda1da37"


def test_original_darts_optimizer_and_regularization_contracts():
    cifar = load_config(ROOT / "configs" / "training" / "darts_cifar10.yaml")
    imagenet = load_config(ROOT / "configs" / "training" / "darts_imagenet.yaml")
    assert cifar["nesterov"] is False
    assert imagenet["nesterov"] is False
    assert (cifar["auxiliary_weight"], cifar["drop_path_prob"]) == (0.4, 0.2)
    assert (imagenet["auxiliary_weight"], imagenet["drop_path_prob"]) == (0.4, 0.0)

    changed = dict(cifar, auxiliary=False)
    with pytest.raises(ValueError, match="auxiliary=False"):
        validate_formal_training_protocol(changed)


def test_tenas_recipe_is_not_accepted_as_original_darts_protocol():
    config = load_config(ROOT / "configs" / "training" / "tenas_imagenet.yaml")
    with pytest.raises(NotImplementedError, match="tenas-retrain-imagenet"):
        validate_formal_training_protocol(config)


def test_darts_global_batch_is_split_without_changing_published_global_batch():
    assert resolve_per_device_batch_size(96, 1, "global") == 96
    assert resolve_per_device_batch_size(96, 4, "global") == 24
    assert resolve_per_device_batch_size(128, 8, "global") == 16
    with pytest.raises(ValueError, match="divisible"):
        resolve_per_device_batch_size(96, 5, "global")


def test_darts_drop_path_uses_upstream_epoch_over_epochs_schedule(tmp_path):
    class DropPathModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(3 * 4 * 4, 2)
            self.drop_path_prob = -1.0

        def forward(self, inputs):
            return self.classifier(inputs.flatten(1))

    data = torch.utils.data.TensorDataset(
        torch.randn(4, 3, 4, 4), torch.randint(2, (4,))
    )
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        DropPathModel(),
        loader,
        loader,
        TrainingConfig(
            epochs=2,
            optimizer="sgd",
            learning_rate=0.025,
            weight_decay=3e-4,
            nesterov=False,
            drop_path_prob=0.2,
            amp=False,
        ),
        tmp_path,
        torch.device("cpu"),
        run_identity={"architecture_id": "darts-test", "protocol": "darts-test"},
    )
    assert [row["drop_path_prob"] for row in read_jsonl(tmp_path / "training.jsonl")] == pytest.approx(
        [0.0, 0.1]
    )
