import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _module():
    path = Path(__file__).resolve().parents[1] / "panel" / "live-status.py"
    specification = importlib.util.spec_from_file_location("zcp_test_panel_live_status", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_search_progress_counts_candidates_separately_from_unique_evaluations(tmp_path):
    module = _module()
    run = tmp_path / "run"
    run.mkdir()
    identity = {"population_size": 3}
    (run / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (run / "search-state.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "population_size": 3,
                "completed_generation": 0,
                "population": [{}, {}, {}],
                "evaluations": 2,
                "cache_hits": 1,
                "elapsed_seconds": 2.0,
            }
        ),
        encoding="utf-8",
    )

    result = module.search_progress(
        "20260732",
        run / "search-state.json",
        [],
        datetime.now(ZoneInfo("Asia/Shanghai")),
    )

    assert result["status"] == "completed"
    assert result["candidates"] == result["evaluations"] == 3
    assert result["unique_evaluations"] == 2
    assert result["cache_hits"] == 1
    assert result["stale"] is False


def test_build_live_status_prefers_completed_seed_manifests_but_preserves_supervisor(
    monkeypatch,
):
    module = _module()
    monkeypatch.setattr(module, "latest_search_states", lambda: [("1", Path("one"))])
    monkeypatch.setattr(
        module,
        "search_progress",
        lambda *_args: {
            "seed": 1,
            "status": "completed",
            "candidates": 3,
            "evaluations": 3,
            "unique_evaluations": 2,
            "cache_hits": 1,
            "total": 3,
        },
    )
    monkeypatch.setattr(
        module,
        "read_json",
        lambda path, _warnings, _label: {
            "status": "completed",
            "supervisor_terminal_status": "failed",
            "supervisor_terminal_detail": "wait failed",
            "cohort_validation_sha256": "a" * 64,
        }
        if path == module.AUTOFORMER_STATUS
        else {"status": "completed"},
    )
    monkeypatch.setattr(module, "gpu_snapshot", lambda _warnings: [])

    result = module.build_live_status()["autoformer"]

    assert result["status"] == "completed"
    assert result["supervisor_status"] == "failed"
    assert result["supervisor_detail"] == "wait failed"
    assert result["reconciled"] is True
    assert result["cohort_status_source"] == "seed_manifests"
    assert result["evaluations_total"] == 3
    assert result["unique_evaluations_total"] == 2
    assert result["cache_hits_total"] == 1


def test_latest_search_states_prefers_completed_over_newer_partial(tmp_path, monkeypatch):
    module = _module()
    root = tmp_path / "autoformer"
    seed = root / "seed-20260731"
    completed = seed / "completed"
    partial = seed / "partial"
    completed.mkdir(parents=True)
    partial.mkdir()
    (completed / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (partial / "manifest.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    (completed / "search-state.json").write_text("{}", encoding="utf-8")
    (partial / "search-state.json").write_text("{}", encoding="utf-8")
    partial_time = (completed / "search-state.json").stat().st_mtime + 10
    import os

    os.utime(partial / "search-state.json", (partial_time, partial_time))
    monkeypatch.setattr(module, "AUTOFORMER_ROOT", root)

    assert module.latest_search_states() == [
        ("20260731", completed / "search-state.json")
    ]


def test_declared_missing_seed_keeps_full_target_denominator(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "latest_search_states", lambda: [("1", Path("one"))])
    monkeypatch.setattr(
        module,
        "search_progress",
        lambda *_args: {
            "seed": 1,
            "status": "completed",
            "candidates": 8,
            "evaluations": 8,
            "unique_evaluations": 8,
            "cache_hits": 0,
            "total": 8,
        },
    )
    monkeypatch.setattr(
        module,
        "read_json",
        lambda path, _warnings, _label: {
            "status": "running",
            "seeds": [1, 2, 3],
            "population_per_seed": 8,
        }
        if path == module.AUTOFORMER_STATUS
        else {"status": "completed"},
    )
    monkeypatch.setattr(module, "gpu_snapshot", lambda _warnings: [])

    result = module.build_live_status()["autoformer"]

    assert result["status"] == "running"
    assert result["candidates_total"] == 8
    assert result["target_total"] == 24
    assert [seed["status"] for seed in result["seeds"]] == [
        "completed",
        "missing",
        "missing",
    ]
