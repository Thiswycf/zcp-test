from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from zcp_test import cli
from zcp_test.inputs import InputBatch, make_input_batch
from zcp_test.types import Architecture, RecordStatus, ScoreDirection, ScoreResult


def _args(source: str, *, seed: int = 41, batches: int = 3) -> Namespace:
    return Namespace(
        input_source=source,
        dataset="cifar10",
        batch_size=2,
        classes=5,
        seed=seed,
        proxy_batches=batches,
    )


@pytest.mark.parametrize("source", ["random", "noise"])
def test_synthetic_proxy_batch_provider_is_deterministic_and_reiterable(source: str) -> None:
    args = _args(source)
    first = make_input_batch(source, args.dataset, 2, 4, 5, args.seed, torch.device("cpu"))

    first_plan = cli._make_proxy_batch_plan(
        args, 4, torch.device("cpu"), first, role="evaluate-proxy"
    )
    second_plan = cli._make_proxy_batch_plan(
        args, 4, torch.device("cpu"), first, role="evaluate-proxy"
    )

    assert first_plan.fingerprint == second_plan.fingerprint
    assert first_plan.batch_fingerprints == second_plan.batch_fingerprints
    assert len(first_plan.batch_fingerprints) == args.proxy_batches
    assert len(set(first_plan.batch_fingerprints)) == args.proxy_batches
    assert first_plan.protocol["seed"] == args.seed
    assert first_plan.protocol["batch_seed_protocol"] == "base-seed-plus-batch-index"
    assert first_plan.protocol["batch_seeds"] == [41, 42, 43]

    first_iteration = list(first_plan.provider())
    second_iteration = list(first_plan.provider())
    assert len(first_iteration) == args.proxy_batches
    for (first_inputs, first_labels), (second_inputs, second_labels) in zip(
        first_iteration, second_iteration, strict=True
    ):
        torch.testing.assert_close(first_inputs, second_inputs)
        torch.testing.assert_close(first_labels, second_labels)


def test_synthetic_proxy_batch_fingerprint_changes_with_base_seed() -> None:
    first_args = _args("random", seed=7)
    second_args = _args("random", seed=8)
    first_batch = make_input_batch("random", "cifar10", 2, 4, 5, 7, torch.device("cpu"))
    second_batch = make_input_batch("random", "cifar10", 2, 4, 5, 8, torch.device("cpu"))

    first_plan = cli._make_proxy_batch_plan(
        first_args, 4, torch.device("cpu"), first_batch, role="search-proxy"
    )
    second_plan = cli._make_proxy_batch_plan(
        second_args, 4, torch.device("cpu"), second_batch, role="search-proxy"
    )

    assert first_plan.fingerprint != second_plan.fingerprint
    assert first_plan.batch_fingerprints != second_plan.batch_fingerprints


class _DatasetStream:
    def __init__(self) -> None:
        self.sample_ids = ((8, 2), (5, 1), (9, 0))
        self.protocol = {
            "source": "dataset",
            "dataset": "cifar10",
            "seed": 41,
            "sample_ids": self.sample_ids,
        }

    def __iter__(self):
        for batch_index, identifiers in enumerate(self.sample_ids):
            inputs = torch.full((2, 3, 4, 4), float(batch_index))
            labels = torch.tensor(identifiers)
            yield inputs, labels


def test_dataset_proxy_batches_use_one_deterministic_without_replacement_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args("dataset")
    first = InputBatch(
        torch.zeros(2, 3, 4, 4),
        torch.tensor([8, 2]),
        {"source": "dataset", "seed": 41},
        "first-batch",
    )
    calls = []

    def fake_stream(*stream_args, **stream_kwargs):
        calls.append((stream_args, stream_kwargs))
        return _DatasetStream()

    monkeypatch.setattr(cli, "make_dataset_batch_stream", fake_stream)
    monkeypatch.setattr(cli, "_resolve_data_root", lambda *_args: "/dataset")
    monkeypatch.setattr("zcp_test.data.transnas_inputs.is_transnas_task", lambda _dataset: False)

    plan = cli._make_proxy_batch_plan(args, 4, torch.device("cpu"), first, role="evaluate-proxy")

    assert len(calls) == 1
    assert calls[0][0][4:6] == (args.seed, args.proxy_batches)
    assert calls[0][1] == {"role": "evaluate-proxy"}
    assert len(list(plan.provider())) == args.proxy_batches
    assert len(plan.batch_fingerprints) == args.proxy_batches
    assert len(set(plan.batch_fingerprints)) == args.proxy_batches
    assert plan.protocol["batch_seed_protocol"] == "single-seed-without-replacement"
    flattened_ids = [identifier for batch in plan.protocol["sample_ids"] for identifier in batch]
    assert len(flattened_ids) == len(set(flattened_ids))


def test_single_proxy_batch_preserves_existing_input_identity() -> None:
    args = _args("random", batches=1)
    first = make_input_batch("random", "cifar10", 2, 4, 5, 41, torch.device("cpu"))

    plan = cli._make_proxy_batch_plan(args, 4, torch.device("cpu"), first, role="evaluate-proxy")

    assert plan.fingerprint == first.fingerprint
    assert plan.batch_fingerprints == (first.fingerprint,)
    [(inputs, labels)] = list(plan.provider())
    torch.testing.assert_close(inputs, first.inputs)
    torch.testing.assert_close(labels, first.labels)


class _TinySpace:
    search_space_id = "proxy_batch_fixture"
    model_family = "cnn"
    model_fidelity = "reference_model"
    implementation_source = "test-fixture"
    implementation_commit = "fixture"

    def sample(self, seed=None):
        return Architecture(self.search_space_id, f"candidate-{seed}", {})

    def build_model(self, architecture, num_classes):
        del architecture
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, num_classes),
        )


def test_evaluate_cli_wires_reiterable_provider_into_proxy_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(cli.SPACES, "create", lambda _name: _TinySpace())
    contexts = []

    def evaluate(proxy_id, model, inputs, labels, loss_fn, *, context, **_kwargs):
        del model, inputs, labels, loss_fn
        contexts.append((proxy_id, context))
        return ScoreResult(
            score=1.0,
            components={"score": 1.0},
            status=RecordStatus.OK,
            proxy_version="fixture-v1",
            direction=ScoreDirection.MAXIMIZE,
        )

    monkeypatch.setattr(cli, "evaluate_proxy", evaluate)
    cli.main(
        [
            "evaluate",
            "--space",
            "proxy_batch_fixture",
            "--proxies",
            "first,second",
            "--count",
            "1",
            "--device",
            "cpu",
            "--input-source",
            "random",
            "--batch-size",
            "2",
            "--input-size",
            "4",
            "--classes",
            "5",
            "--proxy-batches",
            "3",
            "--output",
            str(tmp_path / "run"),
        ]
    )

    assert [proxy_id for proxy_id, _context in contexts] == ["first", "second"]
    first_context = contexts[0][1]
    second_context = contexts[1][1]
    assert first_context.batch_provider is not None
    assert first_context.proxy_batches == 3
    assert len(first_context.batch_fingerprints) == 3
    assert first_context.input_fingerprint == second_context.input_fingerprint
    first_batches = list(first_context.batch_provider())
    second_batches = list(second_context.batch_provider())
    assert len(first_batches) == len(second_batches) == 3
    assert len({inputs.detach().cpu().numpy().tobytes() for inputs, _labels in first_batches}) == 3
    for first_batch, second_batch in zip(first_batches, second_batches, strict=True):
        torch.testing.assert_close(first_batch[0], second_batch[0])
        torch.testing.assert_close(first_batch[1], second_batch[1])
