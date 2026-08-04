import hashlib
import json
from pathlib import Path

import pytest
import yaml

from zcp_test import cli
from zcp_test.acceptance import (
    ResourceMeasurement,
    freeze_training_candidates,
    measure_architecture_resources,
    reconcile_search_cohort,
    validate_plainnet_search,
)
from zcp_test.spaces import SPACES, load_builtin_spaces


def _plainnet_acceptance_run(tmp_path: Path) -> Path:
    run = tmp_path / "450m" / "plainnet-run"
    run.mkdir(parents=True)
    identity = {
        "search_space_id": "zennas_plainnet_mbv2",
        "proxy_id": "az_nas_plainnet",
        "aggregator": "az_nas_log_rank",
        "flops_target": "450m",
        "valid_candidates": 3,
        "search_budget_protocol": "one_percent_acceptance",
        "search_budget_fraction": 0.01,
        "controller_fidelity": (
            "source_aligned_control_flow_port_truncated_one_percent_budget"
        ),
    }
    state_identity = {
        **identity,
        "component_names": [
            "expressivity",
            "progressivity",
            "trainability",
            "complexity",
        ],
    }
    components_a = {
        "expressivity": 1.0,
        "progressivity": 2.0,
        "trainability": 3.0,
        "complexity": 4.0,
    }
    components_b = {
        "expressivity": 2.0,
        "progressivity": 3.0,
        "trainability": 4.0,
        "complexity": 5.0,
    }
    candidates = [
        {
            **identity,
            "record_kind": "candidate",
            "accepted_index": 0,
            "architecture_id": "architecture-a",
            "components": components_a,
            "score": 0.5,
            "score_at_acceptance": 0.5,
            "cache_hit": False,
            "cumulative_evaluations": 1,
            "cumulative_cache_hits": 0,
        },
        {
            **identity,
            "record_kind": "candidate",
            "accepted_index": 1,
            "architecture_id": "architecture-b",
            "components": components_b,
            "score": 1.5,
            "score_at_acceptance": 1.5,
            "cache_hit": False,
            "cumulative_evaluations": 2,
            "cumulative_cache_hits": 0,
        },
        {
            **identity,
            "record_kind": "candidate",
            "accepted_index": 2,
            "architecture_id": "architecture-a",
            "components": components_a,
            "score": 0.5,
            "score_at_acceptance": 0.5,
            "cache_hit": True,
            "cumulative_evaluations": 2,
            "cumulative_cache_hits": 1,
        },
    ]
    summary = {
        **identity,
        "record_kind": "search_summary",
        "evaluations": 2,
        "cache_hits": 1,
        "best_architecture_id": "architecture-b",
        "best_score": 1.5,
    }
    journal = "".join(json.dumps(row) + "\n" for row in [*candidates, summary])
    (run / "manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": "plainnet-fixture"}),
        encoding="utf-8",
    )
    (run / "config.yaml").write_text(
        yaml.safe_dump({"search_identity": identity}), encoding="utf-8"
    )
    (run / "search.jsonl").write_text(journal, encoding="utf-8")
    (run / "search-state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "identity": state_identity,
                "accepted_count": 3,
                "evaluations": 2,
                "cache_hits": 1,
                "summary_written": True,
                "journal_rows": 4,
                "journal_sha256": hashlib.sha256(journal.encode()).hexdigest(),
                "best_architecture_id": "architecture-b",
                "best_score": 1.5,
            }
        ),
        encoding="utf-8",
    )
    (run.parent / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "flops_target": "450m",
                "valid_candidates": 3,
                "search_budget_protocol": "one_percent_acceptance",
                "search_budget_fraction": 0.01,
                "formal_search_completed": False,
                "one_percent_search_completed": True,
            }
        ),
        encoding="utf-8",
    )
    return run


def _rewrite_plainnet_journal(run: Path, rows: list[dict]) -> None:
    journal = "".join(json.dumps(row) + "\n" for row in rows)
    (run / "search.jsonl").write_text(journal, encoding="utf-8")
    state_path = run / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["journal_sha256"] = hashlib.sha256(journal.encode()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")


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


def test_freeze_candidates_selected_only_avoids_comparison_sampling(tmp_path):
    search_run, config, selected = _search_run(tmp_path)
    output = tmp_path / "selected-only"
    result = freeze_training_candidates(
        search_run=search_run,
        training_config_path=config,
        output=output,
        seed=2026,
        pool_size=0,
        selected_only=True,
        measure=_fake_measure,
    )

    assert result["candidate_policy"] == "selected_only"
    assert result["comparison_candidates_included"] is False
    assert set(result["candidates"]) == {"zcp_selected.json"}
    assert {path.name for path in output.iterdir()} == {
        "zcp_selected.json",
        "candidates-manifest.json",
    }
    payload = json.loads((output / "zcp_selected.json").read_text(encoding="utf-8"))
    assert payload["architecture_id"] == selected.architecture_id


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


def _cohort_search_run(tmp_path: Path, seed: int):
    run, _config, _selected = _search_run(
        tmp_path, seed=seed, complete_identity=True
    )
    row = json.loads((run / "search.jsonl").read_text(encoding="utf-8"))
    row["components"] = {"a": 1.0, "b": 2.0}
    row["cache_hit"] = False
    summary = {
        "record_kind": "generation_summary",
        "generation": 0,
        "cumulative_evaluations": 1,
        "cumulative_cache_hits": 0,
    }
    (run / "search.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(summary) + "\n", encoding="utf-8"
    )
    state_path = run / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"evaluations": 1, "cache_hits": 0})
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ended_at"] = f"2026-07-31T18:00:{seed % 60:02d}+08:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_reconcile_search_cohort_validates_artifacts_and_preserves_supervisor_failure(
    tmp_path,
):
    root = tmp_path / "cohort"
    root.mkdir()
    runs = [_cohort_search_run(tmp_path / str(seed), seed) for seed in (91, 92, 93)]
    (root / "status.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "detail": "supervisor wait failed",
                "primary_selection_seed": 92,
                "supporting_robustness_seeds": [91, 93],
            }
        ),
        encoding="utf-8",
    )

    result = reconcile_search_cohort(
        cohort_root=root,
        search_runs=runs,
        expected_space="autoformer",
        expected_population=1,
        expected_seeds=[91, 92, 93],
        expected_components=["a", "b"],
    )

    assert result["candidate_rows_total"] == 3
    assert result["unique_evaluations_total"] == 3
    assert result["primary_selection_seed"] == 92
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["supervisor_terminal_status"] == "failed"
    assert status["supervisor_terminal_detail"] == "supervisor wait failed"
    assert len(status["cohort_validation_sha256"]) == 64

    rerun = reconcile_search_cohort(
        cohort_root=root,
        search_runs=runs,
        expected_space="autoformer",
        expected_population=1,
        expected_seeds=[91, 92, 93],
        expected_components=["a", "b"],
    )
    assert rerun["supervisor_status_before_reconciliation"] == "failed"
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["supervisor_terminal_status"] == "failed"


def test_reconcile_search_cohort_rejects_nonfinite_components(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    run = _cohort_search_run(tmp_path / "run", 91)
    rows = (run / "search.jsonl").read_text(encoding="utf-8").splitlines()
    candidate = json.loads(rows[0])
    candidate["components"]["a"] = float("nan")
    (run / "search.jsonl").write_text(
        json.dumps(candidate) + "\n" + rows[1] + "\n", encoding="utf-8"
    )
    (root / "status.json").write_text(
        json.dumps({"status": "failed", "primary_selection_seed": 91}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="non-finite component"):
        reconcile_search_cohort(
            cohort_root=root,
            search_runs=[run],
            expected_space="autoformer",
            expected_population=1,
            expected_seeds=[91],
            expected_components=["a", "b"],
        )


def test_validate_plainnet_search_cli_returns_structured_summary(tmp_path, capsys):
    run = _plainnet_acceptance_run(tmp_path)

    cli.main(
        [
            "acceptance",
            "validate-plainnet-search",
            "--run",
            str(run),
            "--expected-target",
            "450m",
            "--expected-candidates",
            "3",
            "--expected-budget-protocol",
            "one_percent_acceptance",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"
    assert summary["formal_search_completed"] is False
    assert summary["one_percent_search_completed"] is True
    assert summary["candidate_rows"] == 3
    assert summary["summary_rows"] == 1
    assert summary["unique_architectures"] == 2
    assert summary["evaluations"] == 2
    assert summary["cache_hits"] == 1
    assert summary["best_architecture_id"] == "architecture-b"
    assert summary["best_score"] == 1.5
    assert len(summary["journal_sha256"]) == 64


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("manifest_status", "manifest status must be completed"),
        ("launcher_status", "launcher status mismatch"),
        ("identity", "search identity mismatch"),
        ("accepted_index", "accepted_index values must be complete"),
        ("nonfinite_score", "must be a finite number"),
        ("cache_hit", "cache_hit contradicts architecture history"),
        ("journal_sha256", "journal SHA256 does not match"),
        ("summary_written", "summary_written=true"),
        ("best_candidate", "best candidate does not match maximum score"),
        ("state_counts", "state evaluation/cache counters"),
    ],
)
def test_validate_plainnet_search_rejects_corrupt_runs(tmp_path, damage, message):
    run = _plainnet_acceptance_run(tmp_path)
    if damage == "manifest_status":
        path = run / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "running"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "launcher_status":
        path = run.parent / "status.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["formal_search_completed"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "identity":
        path = run / "config.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["search_identity"]["aggregator"] = "wrong"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    elif damage in {"accepted_index", "nonfinite_score", "cache_hit", "best_candidate"}:
        rows = [
            json.loads(line)
            for line in (run / "search.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if damage == "accepted_index":
            rows[1]["accepted_index"] = 2
        elif damage == "nonfinite_score":
            rows[0]["score"] = float("nan")
        elif damage == "cache_hit":
            rows[2]["cache_hit"] = False
        else:
            rows[-1]["best_architecture_id"] = "architecture-a"
        _rewrite_plainnet_journal(run, rows)
    else:
        path = run / "search-state.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if damage == "journal_sha256":
            payload["journal_sha256"] = "0" * 64
        elif damage == "summary_written":
            payload["summary_written"] = False
        else:
            payload["evaluations"] = 3
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_plainnet_search(
            run=run,
            expected_target="450m",
            expected_candidates=3,
            expected_budget_protocol="one_percent_acceptance",
        )
