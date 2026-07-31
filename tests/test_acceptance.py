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


def _search_run(tmp_path: Path, status: str = "completed", include_identity: bool = True):
    load_builtin_spaces()
    space = SPACES.create("autoformer")
    selected = space.sample(91)
    run = tmp_path / "search-run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"status": status, "run_id": "search-fixture"}), encoding="utf-8"
    )
    config = {}
    if include_identity:
        config["search_identity"] = {
            "search_space_id": "autoformer",
            "proxy_id": "naswot",
            "proxy_version": "1",
            "input_fingerprint": "fixture-batch",
            "dataset": "imagenet1k",
            "seed": 91,
        }
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
