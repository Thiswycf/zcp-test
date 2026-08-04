from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from zcp_test import cli
from zcp_test.artifacts import JsonlWriter, read_jsonl
from zcp_test.cli import build_parser
from zcp_test.config import load_config
from zcp_test.models.plainnet import (
    INITIAL_STRUCTURE,
    AZPlainNetMobileNetV2,
    parse_plainnet_structure,
)
from zcp_test.proxies.az_nas import PLAINNET_COMPONENTS, log_rank_aggregate
from zcp_test.search.plainnet_source_aligned import (
    CONTROLLER_FIDELITY,
    PlainNetSourceAlignedSearch,
    PlainNetTargetProfile,
    SOURCE_PARENT_POOL,
    SOURCE_VALID_CANDIDATES,
    load_plainnet_search_state,
    plainnet_num_layers,
    plainnet_official_flops,
    resolve_target_profile,
    source_block_replacement,
)
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.types import Architecture


def _space() -> Any:
    load_builtin_spaces()
    return SPACES.create("zennas_plainnet_mbv2")


def _components(architecture: Architecture) -> dict[str, float]:
    value = int(architecture.architecture_id[:12], 16) / float(16**12)
    return {
        "expressivity": value,
        "progressivity": value * 0.5 + 0.1,
        "trainability": value * 0.25 + 0.2,
        "complexity": value * 1000 + 1,
    }


def _target(max_layers: int = 100) -> PlainNetTargetProfile:
    return PlainNetTargetProfile("test", 10**15, max_layers)


def _search(
    root: Path,
    *,
    valid_candidates: int,
    parent_pool: int = SOURCE_PARENT_POOL,
    evaluator: Any = _components,
    mutator: Any = source_block_replacement,
    resume_state: dict[str, Any] | None = None,
    resume_journal_path: Path | None = None,
    state_identity: dict[str, Any] | None = None,
) -> PlainNetSourceAlignedSearch:
    return PlainNetSourceAlignedSearch(
        space=_space(),
        evaluator=evaluator,
        writer=JsonlWriter(root / "search.jsonl", 1),
        state_path=root / "search-state.json",
        seed=123,
        target=_target(),
        valid_candidates=valid_candidates,
        parent_pool=parent_pool,
        classes=1000,
        state_identity=state_identity,
        resume_state=resume_state,
        resume_journal_path=resume_journal_path,
        resource_evaluator=lambda _architecture: 1,
        mutator=mutator,
    )


def test_source_profiles_and_default_protocol_config() -> None:
    assert SOURCE_VALID_CANDIDATES == 100_000
    assert SOURCE_PARENT_POOL == 1_024
    assert resolve_target_profile("450M") == PlainNetTargetProfile(
        "450m", 450_000_000, 14
    )
    assert resolve_target_profile("600m") == PlainNetTargetProfile(
        "600m", 600_000_000, 14
    )
    assert resolve_target_profile("1G") == PlainNetTargetProfile(
        "1g", 1_000_000_000, 16
    )
    with pytest.raises(ValueError, match="Unknown PlainNet FLOPs target"):
        resolve_target_profile("900m")

    config = load_config("configs/search/plainnet_mbv2_source_aligned.yaml")["search"]
    assert config["controller"] == "plainnet_source_aligned"
    assert config["valid_candidates"] == SOURCE_VALID_CANDIDATES
    assert config["population"] == SOURCE_PARENT_POOL
    assert config["batch_size"] == 64
    assert config["input_size"] == 224
    assert config["flops_target"] == "450m"


def test_source_replacement_is_deterministic_valid_and_has_no_crossover() -> None:
    space = _space()
    initial = space.canonicalize({"structure": INITIAL_STRUCTURE, "resolution": 224})

    left, left_ids = source_block_replacement(space, initial, random.Random(31), 2)
    right, right_ids = source_block_replacement(space, initial, random.Random(31), 2)

    assert left == right
    assert left_ids == right_ids
    assert 1 <= len(left_ids) <= 2
    blocks = parse_plainnet_structure(str(left.spec["structure"]))
    assert all(block.sub_layers < 6 for block in blocks if block.kind == "residual")
    assert blocks[-1].out_channels == 2048


def test_official_flops_port_matches_reference_model_golden() -> None:
    space = _space()
    architecture = space.canonicalize(
        {"structure": INITIAL_STRUCTURE, "resolution": 224}
    )
    model = AZPlainNetMobileNetV2(INITIAL_STRUCTURE, num_classes=1000, use_se=False)

    assert plainnet_num_layers(architecture) == 6
    assert plainnet_official_flops(architecture) == int(
        model.official_complexity_ops(224)
    )
    assert plainnet_official_flops(architecture) == 162_396_776


def test_parent_selection_matches_initial_history_and_top_pool_rules(tmp_path: Path) -> None:
    history_search = _search(tmp_path / "history", valid_candidates=13)
    history_search.run()
    history_rows = list(read_jsonl(tmp_path / "history/search.jsonl"))[:-1]

    assert [row["parent_selection"] for row in history_rows[:11]] == [
        "initial_structure"
    ] * 11
    assert [row["replacements_requested"] for row in history_rows[:11]] == [1] * 11
    assert [row["parent_selection"] for row in history_rows[11:]] == [
        "all_history_random"
    ] * 2
    assert [row["replacements_requested"] for row in history_rows[11:]] == [2] * 2
    assert all(row["crossover"] is False for row in history_rows)
    final_scores = log_rank_aggregate(
        [row["components"] for row in history_rows], PLAINNET_COMPONENTS
    )
    assert [row["score"] for row in history_rows] == pytest.approx(final_scores)
    assert all(
        row["score_semantics"] == "final_full_history_log_rank"
        for row in history_rows
    )

    top_search = _search(tmp_path / "top", valid_candidates=13, parent_pool=4)
    top_search.run()
    top_rows = list(read_jsonl(tmp_path / "top/search.jsonl"))[:-1]
    first_scores = log_rank_aggregate(
        [row["components"] for row in top_rows[:11]], PLAINNET_COMPONENTS
    )
    expected_top = set(sorted(range(11), key=lambda index: first_scores[index])[-3:])

    assert top_rows[11]["parent_selection"] == "top_pool_random"
    assert top_rows[11]["parent_index"] in expected_top
    assert top_rows[12]["parent_selection"] == "top_pool_random"


def test_duplicate_architectures_use_cache_but_remain_valid_candidates(
    tmp_path: Path,
) -> None:
    calls = 0

    def evaluator(architecture: Architecture) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return _components(architecture)

    def unchanged(
        _space: Any,
        architecture: Architecture,
        _rng: random.Random,
        _replacements: int,
    ) -> tuple[Architecture, tuple[int, ...]]:
        return architecture, (0,)

    search = _search(
        tmp_path,
        valid_candidates=4,
        evaluator=evaluator,
        mutator=unchanged,
    )
    best = search.run()
    rows = list(read_jsonl(tmp_path / "search.jsonl"))
    state = load_plainnet_search_state(tmp_path / "search-state.json")

    assert best is not None
    assert calls == 1
    assert [row["cache_hit"] for row in rows[:-1]] == [False, True, True, True]
    assert state["evaluations"] == 1
    assert state["cache_hits"] == 3
    assert rows[-1]["diversity"] == 0.25


def test_resume_is_deterministic_and_trims_uncommitted_tail(tmp_path: Path) -> None:
    full_root = tmp_path / "full"
    resumed_root = tmp_path / "resumed"
    full = _search(full_root, valid_candidates=15)
    full_best = full.run()

    partial = _search(resumed_root, valid_candidates=15)
    assert partial.run(stop_after_accepted=7) is None
    with (resumed_root / "search.jsonl").open("ab") as handle:
        handle.write(b'{"uncommitted"')
    state = load_plainnet_search_state(resumed_root / "search-state.json")
    resumed = _search(resumed_root, valid_candidates=15, resume_state=state)
    assert len(list(read_jsonl(resumed_root / "search.jsonl"))) == 7
    resumed_best = resumed.run()

    assert full_best is not None and resumed_best is not None
    assert resumed_best.architecture == full_best.architecture
    assert resumed_best.score == full_best.score
    assert (resumed_root / "search.jsonl").read_bytes() == (
        full_root / "search.jsonl"
    ).read_bytes()
    completed = load_plainnet_search_state(resumed_root / "search-state.json")
    assert completed["status"] == "completed"
    assert completed["accepted_count"] == 15
    assert completed["summary_written"] is True


def test_resume_can_continue_into_a_new_run_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source = _search(source_root, valid_candidates=9)
    assert source.run(stop_after_accepted=4) is None
    state = load_plainnet_search_state(source_root / "search-state.json")

    resumed = _search(
        destination_root,
        valid_candidates=9,
        resume_state=state,
        resume_journal_path=source_root / "search.jsonl",
    )
    best = resumed.run()

    assert best is not None
    assert len(list(read_jsonl(source_root / "search.jsonl"))) == 4
    assert len(list(read_jsonl(destination_root / "search.jsonl"))) == 10
    assert load_plainnet_search_state(destination_root / "search-state.json")[
        "status"
    ] == "completed"


def test_resume_rejects_protocol_identity_mismatch(tmp_path: Path) -> None:
    search = _search(tmp_path, valid_candidates=3)
    search.run(stop_after_accepted=2)
    state = load_plainnet_search_state(tmp_path / "search-state.json")

    with pytest.raises(ValueError, match="identity mismatch"):
        _search(
            tmp_path,
            valid_candidates=3,
            resume_state=state,
            state_identity={"dataset": "different"},
        )


def test_resume_rejects_committed_journal_checksum_mismatch(tmp_path: Path) -> None:
    search = _search(tmp_path, valid_candidates=3)
    search.run(stop_after_accepted=2)
    state = load_plainnet_search_state(tmp_path / "search-state.json")
    journal = tmp_path / "search.jsonl"
    payload = journal.read_text(encoding="utf-8")
    corrupted = payload.replace('"attempt": 1', '"attempt": 9', 1)
    assert corrupted != payload
    journal.write_text(corrupted, encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        _search(tmp_path, valid_candidates=3, resume_state=state)


def test_constraints_are_applied_before_proxy_evaluation(tmp_path: Path) -> None:
    calls = 0
    attempts = 0

    def evaluator(architecture: Architecture) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return _components(architecture)

    def mutator(
        space: Any,
        architecture: Architecture,
        _rng: random.Random,
        _replacements: int,
    ) -> tuple[Architecture, tuple[int, ...]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            high_layers = INITIAL_STRUCTURE.replace(
                "SuperResIDWE6K3(8,32,2,8,1)",
                "SuperResIDWE6K3(8,32,2,8,2)",
            )
            return space.canonicalize(
                {"structure": high_layers, "resolution": 224}
            ), (1,)
        return architecture, (0,)

    resources = iter([101, 99])
    search = PlainNetSourceAlignedSearch(
        space=_space(),
        evaluator=evaluator,
        writer=JsonlWriter(tmp_path / "search.jsonl", 1),
        state_path=tmp_path / "search-state.json",
        seed=1,
        target=PlainNetTargetProfile("test", 100, 6),
        valid_candidates=1,
        parent_pool=1024,
        resource_evaluator=lambda _architecture: next(resources),
        mutator=mutator,
    )
    search.run()
    state = load_plainnet_search_state(tmp_path / "search-state.json")

    assert attempts == 3
    assert calls == 1
    assert state["attempts"] == 3
    assert state["rejected_layers"] == 1
    assert state["rejected_flops"] == 1


def test_cli_exposes_only_explicit_source_controller_arguments() -> None:
    args = build_parser().parse_args(
        [
            "search",
            "--controller",
            "plainnet_source_aligned",
            "--space",
            "zennas_plainnet_mbv2",
            "--proxy",
            "az_nas_plainnet",
            "--aggregator",
            "az_nas_log_rank",
            "--valid-candidates",
            "100000",
            "--population",
            "1024",
            "--generations",
            "0",
            "--flops-target",
            "1g",
        ]
    )

    assert args.controller == "plainnet_source_aligned"
    assert args.valid_candidates == 100_000
    assert args.population == 1_024
    assert args.flops_target == "1g"
    assert CONTROLLER_FIDELITY == "source_aligned_control_flow_port"


def test_search_config_can_supply_space_without_duplicate_cli_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture(args: Any) -> None:
        captured.update(vars(args))

    monkeypatch.setattr(cli, "command_search", capture)
    cli.main(
        [
            "search",
            "--config",
            "configs/search/plainnet_mbv2_source_aligned.yaml",
        ]
    )

    assert captured["space"] == "zennas_plainnet_mbv2"
    assert captured["controller"] == "plainnet_source_aligned"
    assert captured["valid_candidates"] == 100_000
