import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import zcp_test.cli as cli
from zcp_test.inputs import CandidateInputResolver
from zcp_test.search import cache_key
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.types import Architecture, RecordStatus, ScoreDirection, ScoreResult


def _ofa_pair():
    load_builtin_spaces()
    space = SPACES.create("ofa_proxyless_mbv2")
    first = space.sample(17)
    second_spec = {
        key: list(value) if isinstance(value, list) else value for key, value in first.spec.items()
    }
    second_spec["resolution"] = 132 if first.spec["resolution"] != 132 else 136
    return space, first, space.canonicalize(second_spec)


def _imagenet_fixture(root: Path) -> None:
    directory = root / "train" / "class-a"
    directory.mkdir(parents=True)
    horizontal = np.arange(256, dtype=np.uint8)[None, :]
    image = np.stack(
        [
            np.broadcast_to(horizontal, (256, 256)),
            np.broadcast_to(horizontal.T, (256, 256)),
            np.full((256, 256), 127, dtype=np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(image).save(directory / "sample.png")


def test_ofa_resolution_controls_dataset_tensor_fingerprint_cache_and_forward(tmp_path):
    space, first, second = _ofa_pair()
    _imagenet_fixture(tmp_path)
    resolver = CandidateInputResolver(
        source="dataset",
        dataset="imagenet1k",
        batch_size=1,
        requested_input_size=32,
        classes=1000,
        seed=9,
        device=torch.device("cpu"),
        data_root=str(tmp_path),
    )

    first_batch = resolver.resolve(first)
    second_batch = resolver.resolve(second)

    assert first_batch.inputs.shape[-2:] == (first.spec["resolution"],) * 2
    assert second_batch.inputs.shape[-2:] == (second.spec["resolution"],) * 2
    assert first_batch.protocol["sample_ids"] == second_batch.protocol["sample_ids"] == [0]
    assert first_batch.fingerprint != second_batch.fingerprint
    assert resolver.resolve(first) is first_batch
    assert cache_key(first, "naswot", "imagenet1k", 9, first_batch.fingerprint) != cache_key(
        second, "naswot", "imagenet1k", 9, second_batch.fingerprint
    )

    first_model = space.build_model(first, 3).eval()
    second_model = space.build_model(second, 3).eval()
    second_model.load_state_dict(first_model.state_dict())
    resolver.validate_model(first, first_model, first_batch)
    resolver.validate_model(second, second_model, second_batch)
    with torch.no_grad():
        first_output = first_model(first_batch.inputs)
        second_output = second_model(second_batch.inputs)
    assert first_output.shape == second_output.shape == (1, 3)
    assert not torch.equal(first_output, second_output)


def test_ofa_explicit_input_size_conflict_fails_closed():
    _, architecture, _ = _ofa_pair()
    resolver = CandidateInputResolver(
        source="random",
        dataset="imagenet1k",
        batch_size=1,
        requested_input_size=224,
        classes=1000,
        seed=1,
        device=torch.device("cpu"),
        explicit_input_size=True,
    )

    with pytest.raises(ValueError, match="Explicit input_size conflicts"):
        resolver.resolve(architecture)


class _TinyOFAResolutionSpace:
    search_space_id = "ofa_proxyless_mbv2"
    model_family = "cnn"
    model_fidelity = "reference_model"
    implementation_source = "test-fixture"
    implementation_commit = "fixture"

    def __init__(self):
        self.counter = 0

    def _architecture(self):
        resolution = (128, 132)[self.counter % 2]
        self.counter += 1
        return Architecture(
            self.search_space_id,
            f"candidate-{resolution}",
            {"resolution": resolution},
        )

    def sample(self, seed=None):
        return self._architecture()

    def canonicalize(self, specification):
        resolution = int(specification["resolution"])
        return Architecture(
            self.search_space_id,
            f"candidate-{resolution}",
            {"resolution": resolution},
        )

    def mutate(self, architecture, seed=None):
        return self._architecture()

    def crossover(self, left, right, seed=None):
        return self._architecture()

    def build_model(self, architecture, num_classes):
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, num_classes),
        )
        model.image_size = architecture.spec["resolution"]
        return model


def _successful_proxy_result(proxy_id):
    return ScoreResult(
        score=1.0,
        components={"score": 1.0},
        status=RecordStatus.OK,
        proxy_version=f"{proxy_id}-v1",
        direction=ScoreDirection.MAXIMIZE,
    )


def test_ofa_evaluate_reuses_one_candidate_batch_for_multiple_proxies(monkeypatch, tmp_path):
    space = _TinyOFAResolutionSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    calls = []

    def evaluate(proxy_id, model, inputs, *args, **kwargs):
        calls.append((model.image_size, tuple(inputs.shape[-2:]), inputs.data_ptr()))
        return _successful_proxy_result(proxy_id)

    monkeypatch.setattr(cli, "evaluate_proxy", evaluate)
    output = tmp_path / "evaluate"
    cli.main(
        [
            "evaluate",
            "--space",
            "ofa_proxyless_mbv2",
            "--proxies",
            "first,second",
            "--count",
            "2",
            "--device",
            "cpu",
            "--input-source",
            "random",
            "--batch-size",
            "1",
            "--classes",
            "3",
            "--output",
            str(output),
        ]
    )

    run = next(output.iterdir())
    rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
    assert [(row["candidate_resolution"], row["actual_input_size"]) for row in rows] == [
        (128, 128),
        (128, 128),
        (132, 132),
        (132, 132),
    ]
    assert rows[0]["input_fingerprint"] == rows[1]["input_fingerprint"]
    assert rows[2]["input_fingerprint"] == rows[3]["input_fingerprint"]
    assert rows[0]["input_fingerprint"] != rows[2]["input_fingerprint"]
    assert calls[0][2] == calls[1][2]
    assert calls[2][2] == calls[3][2]
    assert calls[0][1] == (128, 128)
    assert calls[2][1] == (132, 132)


def test_ofa_search_rows_record_resolution_fingerprint_and_cache_identity(monkeypatch, tmp_path):
    space = _TinyOFAResolutionSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    monkeypatch.setattr(
        cli,
        "evaluate_proxy",
        lambda proxy_id, *args, **kwargs: _successful_proxy_result(proxy_id),
    )
    output = tmp_path / "search"
    cli.main(
        [
            "search",
            "--space",
            "ofa_proxyless_mbv2",
            "--proxy",
            "params",
            "--population",
            "2",
            "--generations",
            "0",
            "--device",
            "cpu",
            "--input-source",
            "random",
            "--batch-size",
            "1",
            "--classes",
            "3",
            "--output",
            str(output),
        ]
    )

    run = next(output.iterdir())
    rows = [
        json.loads(line)
        for line in (run / "search.jsonl").read_text().splitlines()
        if json.loads(line)["record_kind"] == "candidate"
    ]
    assert {row["actual_input_size"] for row in rows} == {128, 132}
    assert len({row["input_fingerprint"] for row in rows}) == 2
    assert len({row["evaluation_cache_key"] for row in rows}) == 2
    assert all(row["input_size_policy"] == "architecture_resolution" for row in rows)


def test_ofa_search_resume_preserves_candidate_input_identity(monkeypatch, capsys, tmp_path):
    space = _TinyOFAResolutionSpace()
    monkeypatch.setattr(cli.SPACES, "create", lambda name: space)
    monkeypatch.setattr(
        cli,
        "evaluate_proxy",
        lambda proxy_id, *args, **kwargs: _successful_proxy_result(proxy_id),
    )
    common = [
        "--space",
        "ofa_proxyless_mbv2",
        "--proxy",
        "params",
        "--population",
        "2",
        "--device",
        "cpu",
        "--input-source",
        "random",
        "--batch-size",
        "1",
        "--classes",
        "3",
    ]
    first_output = tmp_path / "first"
    cli.main(["search", *common, "--generations", "0", "--output", str(first_output)])
    first_result = json.loads(capsys.readouterr().out)

    resumed_output = tmp_path / "resumed"
    cli.main(
        [
            "search",
            *common,
            "--generations",
            "1",
            "--resume",
            first_result["search_state"],
            "--output",
            str(resumed_output),
        ]
    )
    resumed_result = json.loads(capsys.readouterr().out)
    resumed_run = Path(resumed_result["run"])
    rows = [json.loads(line) for line in (resumed_run / "search.jsonl").read_text().splitlines()]
    summaries = [row["generation"] for row in rows if row["record_kind"] == "generation_summary"]
    candidates = [row for row in rows if row["record_kind"] == "candidate"]
    state = json.loads((resumed_run / "search-state.json").read_text())

    assert summaries == [0, 1]
    assert all(row["input_size_policy"] == "architecture_resolution" for row in candidates)
    assert all(row["evaluation_cache_key"] for row in candidates)
    assert all(item["evaluation_metadata"] for item in state["population"])


def test_ofa_cli_explicit_input_size_mismatch_fails_before_run(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyOFAResolutionSpace())
    output = tmp_path / "runs"

    with pytest.raises(ValueError, match="Explicit input_size conflicts"):
        cli.main(
            [
                "evaluate",
                "--space",
                "ofa_proxyless_mbv2",
                "--proxies",
                "params",
                "--count",
                "1",
                "--device",
                "cpu",
                "--input-source",
                "random",
                "--input-size",
                "224",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_ofa_search_rejects_fixed_explicit_input_size_before_run(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyOFAResolutionSpace())
    output = tmp_path / "search"

    with pytest.raises(ValueError, match="controlled by each candidate resolution"):
        cli.main(
            [
                "search",
                "--space",
                "ofa_proxyless_mbv2",
                "--proxy",
                "params",
                "--population",
                "2",
                "--generations",
                "0",
                "--device",
                "cpu",
                "--input-source",
                "random",
                "--input-size",
                "224",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_ofa_equals_syntax_input_size_is_still_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.SPACES, "create", lambda name: _TinyOFAResolutionSpace())

    with pytest.raises(ValueError, match="Explicit input_size conflicts"):
        cli.main(
            [
                "evaluate",
                "--space",
                "ofa_proxyless_mbv2",
                "--proxies",
                "params",
                "--count",
                "1",
                "--device",
                "cpu",
                "--input-source",
                "random",
                "--input-size=224",
                "--output",
                str(tmp_path / "runs"),
            ]
        )
