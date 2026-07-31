import json
from pathlib import Path

import pytest
import yaml

from zcp_test.acceptance import (
    ResourceMeasurement,
    freeze_training_candidates,
    measure_architecture_resources,
)
from zcp_test.spaces import SPACES, load_builtin_spaces


def _search_run(
    tmp_path: Path,
    status: str = "completed",
    include_identity: bool = True,
    seed: int = 91,
    complete_identity: bool = False,
):
    load_builtin_spaces()
    space = SPACES.create("autoformer")
    selected = space.sample(seed)
    run = tmp_path / f"search-run-{seed}"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"status": status, "run_id": "search-fixture"}), encoding="utf-8"
    )
    config = {}
    if include_identity:
        identity = {
            "search_space_id": "autoformer",
            "proxy_id": "naswot",
            "proxy_version": "1",
            "input_fingerprint": f"fixture-batch-{seed}",
            "dataset": "imagenet1k",
            "seed": seed,
        }
        if complete_identity:
            identity.update(
                {
                    "model_fidelity": "reference_model",
                    "model_profile": "fixture",
                    "implementation_commit": "fixture-commit",
                    "proxy_direction": "maximize",
                    "aggregator": "primary",
                    "model_initialization_protocol": "architecture-hash-v1",
                    "input_source": "random",
                    "population_size": 1,
                    "elite_ratio": 0.2,
                    "batch_size": 2,
                    "input_size": 224,
                    "classes": 1000,
                    "weight_mode": "independent_scratch",
                }
            )
        config["search_identity"] = identity
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "best_architecture.json").write_text(
        json.dumps(selected.to_dict()), encoding="utf-8"
    )
    (run / "search.jsonl").write_text(
        json.dumps(
            {
                "record_kind": "candidate",
                "architecture_id": selected.architecture_id,
                "architecture": selected.spec,
                "score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "search-state.json").write_text(
        json.dumps(
            {
                "completed_generation": 0,
                "identity": config.get("search_identity"),
                "population": [
                    {
                        "architecture": selected.to_dict(),
                        "score": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    training_config = tmp_path / "training.yaml"
    training_config.write_text(
        yaml.safe_dump({"space": "autoformer", "input_size": 224}), encoding="utf-8"
    )
    return run, training_config, selected


def _fake_measure(space, architecture, training_config, classes):
    del space, training_config, classes
    depth = int(architecture.spec["depth"])
    width = int(architecture.spec["hidden_dim"])
    return ResourceMeasurement(
        parameters=depth * width,
        compute_value=depth * width * width,
        compute_metric="fixture_ops",
        generic_flops=False,
        input_size=224,
    )


def test_freeze_candidates_requires_search_provenance_and_writes_three_roles(tmp_path):
    search_run, config, selected = _search_run(tmp_path)
    output = tmp_path / "candidates"
    result = freeze_training_candidates(
        search_run=search_run,
        training_config_path=config,
        output=output,
        seed=2026,
        pool_size=4,
        measure=_fake_measure,
    )

    expected = {
        "zcp_selected.json": "zcp_selected",
        "fixed_random.json": "fixed_random",
        "params_flops_matched.json": "params_flops_matched",
    }
    payloads = {
        name: json.loads((output / name).read_text(encoding="utf-8"))
        for name in expected
    }
    assert {payload["candidate_role"] for payload in payloads.values()} == set(
        expected.values()
    )
    assert payloads["zcp_selected.json"]["architecture_id"] == selected.architecture_id
    assert len({payload["architecture_id"] for payload in payloads.values()}) == 3
    assert result["resource_protocol"]["compute_metric"] == "fixture_ops"
    assert result["search_provenance"]["search_identity"]["proxy_version"] == "1"
    manifest = json.loads(
        (output / "candidates-manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["candidates"]) == set(expected)
    assert all(len(entry["sha256"]) == 64 for entry in manifest["candidates"].values())


@pytest.mark.parametrize(
    ("status", "include_identity", "message"),
    [
        ("running", True, "completed search run"),
        ("completed", False, "versioned search_identity"),
    ],
)
def test_freeze_candidates_rejects_unaccepted_search_run(
    tmp_path, status, include_identity, message
):
    search_run, config, _ = _search_run(
        tmp_path, status=status, include_identity=include_identity
    )
    with pytest.raises(ValueError, match=message):
        freeze_training_candidates(
            search_run=search_run,
            training_config_path=config,
            output=tmp_path / "output",
            seed=1,
            pool_size=2,
            measure=_fake_measure,
        )


def test_freeze_candidates_records_supporting_runs_without_averaging(tmp_path):
    primary, config, selected = _search_run(
        tmp_path / "primary", seed=91, complete_identity=True
    )
    supporting, _unused_config, support_selected = _search_run(
        tmp_path / "support", seed=92, complete_identity=True
    )
    result = freeze_training_candidates(
        search_run=primary,
        supporting_search_runs=[supporting],
        training_config_path=config,
        output=tmp_path / "candidates",
        seed=2026,
        pool_size=2,
        measure=_fake_measure,
    )

    cohort = result["search_cohort"]
    assert cohort["primary_seed"] == 91
    assert cohort["supporting_seeds"] == [92]
    assert cohort["top_candidates"] == [
        {
            "role": "primary_selection",
            "seed": 91,
            "architecture_id": selected.architecture_id,
        },
        {
            "role": "supporting_robustness_only",
            "seed": 92,
            "architecture_id": support_selected.architecture_id,
        },
    ]
    frozen = json.loads(
        (tmp_path / "candidates" / "zcp_selected.json").read_text(encoding="utf-8")
    )
    assert frozen["architecture_id"] == selected.architecture_id
    assert "not averaged or cherry-picked" in frozen["provenance"]["search_cohort"][
        "selection_rule"
    ]


def test_freeze_candidates_rejects_supporting_protocol_mismatch(tmp_path):
    primary, config, _selected = _search_run(
        tmp_path / "primary", seed=91, complete_identity=True
    )
    supporting, _unused_config, _support_selected = _search_run(
        tmp_path / "support", seed=92, complete_identity=True
    )
    support_config_path = supporting / "config.yaml"
    support_config = yaml.safe_load(support_config_path.read_text(encoding="utf-8"))
    support_config["search_identity"]["proxy_version"] = "different"
    support_config_path.write_text(yaml.safe_dump(support_config), encoding="utf-8")
    state_path = supporting / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["identity"] = support_config["search_identity"]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="Supporting search protocol mismatch: proxy_version"):
        freeze_training_candidates(
            search_run=primary,
            supporting_search_runs=[supporting],
            training_config_path=config,
            output=tmp_path / "candidates",
            seed=2026,
            pool_size=2,
            measure=_fake_measure,
        )


def test_freeze_candidates_resolves_best_score_ties_by_architecture_id(tmp_path):
    run, config, selected = _search_run(tmp_path, seed=91)
    space = SPACES.create("autoformer")
    other = space.sample(92)
    stable, unstable = sorted((selected, other), key=lambda item: item.architecture_id)
    (run / "best_architecture.json").write_text(
        json.dumps(unstable.to_dict()), encoding="utf-8"
    )
    rows = [
        {
            "record_kind": "candidate",
            "architecture_id": architecture.architecture_id,
            "architecture": architecture.spec,
            "score": 1.0,
        }
        for architecture in (unstable, stable)
    ]
    (run / "search.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    state = json.loads((run / "search-state.json").read_text(encoding="utf-8"))
    state["population"] = [
        {"architecture": unstable.to_dict(), "score": 1.0},
        {"architecture": stable.to_dict(), "score": 1.0},
    ]
    (run / "search-state.json").write_text(json.dumps(state), encoding="utf-8")

    result = freeze_training_candidates(
        search_run=run,
        training_config_path=config,
        output=tmp_path / "candidates",
        seed=2026,
        pool_size=2,
        measure=_fake_measure,
    )

    assert result["search_provenance"]["best_selection"] == {
        "strategy": "maximum_score_then_architecture_id_ascending_v1",
        "maximum_score": 1.0,
        "maximum_score_tie_count": 2,
        "best_file_architecture_id": unstable.architecture_id,
        "selected_architecture_id": stable.architecture_id,
    }


def test_autoformer_resource_measurement_preserves_official_non_flops_name():
    load_builtin_spaces()
    space = SPACES.create("autoformer")
    architecture = space.sample(3)
    config = yaml.safe_load(
        Path("configs/training/autoformer_imagenet.yaml").read_text(encoding="utf-8")
    )
    resources = measure_architecture_resources(space, architecture, config, 1000)

    assert resources.parameters > 0
    assert resources.compute_value > 0
    assert resources.compute_metric == "cream_autoformer_official_complexity_ops"
    assert resources.generic_flops is False
    assert resources.input_size == 224
