from __future__ import annotations

import pytest
import torch

from zcp_test.spaces.nb101 import Nb101Space


def _spec() -> dict[str, object]:
    return {
        "matrix": [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        "operations": ["input", "conv1x1-bn-relu", "conv3x3-bn-relu", "output"],
    }


def test_nb101_canonicalization_prunes_and_hashes_stably() -> None:
    space = Nb101Space()
    first = space.canonicalize(_spec())
    second = space.canonicalize(_spec())

    assert len(first.architecture_id) == 32
    assert first == second
    dangling = {
        "matrix": [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "operations": [
            "input",
            "conv1x1-bn-relu",
            "maxpool3x3",
            "conv3x3-bn-relu",
            "output",
        ],
    }
    pruned = space.canonicalize(dangling)
    assert len(pruned.spec["matrix"]) == 3


def test_nb101_rejects_invalid_graphs() -> None:
    space = Nb101Space()
    invalid = _spec()
    invalid["matrix"][2][1] = 1
    with pytest.raises(ValueError, match="upper triangular"):
        space.canonicalize(invalid)


def test_nb101_matches_known_official_module_hash() -> None:
    specification = {
        "matrix": [
            [0, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        "operations": [
            "input",
            "conv3x3-bn-relu",
            "maxpool3x3",
            "conv3x3-bn-relu",
            "conv3x3-bn-relu",
            "conv1x1-bn-relu",
            "output",
        ],
    }

    architecture = Nb101Space().canonicalize(specification)

    assert architecture.architecture_id == "00005c142e6f48ac74fdcf73e3439874"


def test_nb101_sample_mutate_and_reference_model_forward() -> None:
    space = Nb101Space()
    architecture = space.canonicalize(_spec())
    mutated = space.mutate(architecture, seed=7)
    sampled = space.sample(seed=9)

    assert mutated.architecture_id != architecture.architecture_id
    assert sampled.search_space_id == "nb101_dag"
    model = space.build_model(architecture, 10).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 10)
