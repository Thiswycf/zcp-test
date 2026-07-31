import pytest
import torch
import math
from pathlib import Path

from zcp_test.artifacts import JsonlWriter, merge_jsonl, read_jsonl
from zcp_test.artifacts.run import _package_versions
from zcp_test.proxies.evaluator import evaluate_proxy
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.proxies.builtin import FunctionProxy
from zcp_test.reporting import correlation_summary
from zcp_test.search import EvolutionSearch, cache_key, load_search_state
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.training import TrainingConfig, train_model
from zcp_test.training.checkpoint import atomic_torch_save, load_checkpoint
from zcp_test.training.trainer import (
    _collect_checkpoint_rng,
    _cosine_learning_rate,
    _normalized_checkpoint_config,
    _normalized_checkpoint_identity,
    _optimizer_parameter_groups,
    _restore_checkpoint_rng,
    _restore_training_log,
)
from zcp_test.types import ProxyCapability, ProxyOutput


def test_jsonl_merge_and_partial_recovery(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    JsonlWriter(first, 1).append({"id": "a", "value": 1})
    JsonlWriter(second, 1).append({"id": "a", "value": 2})
    JsonlWriter(second, 1).append({"id": "b", "value": 3})
    assert merge_jsonl([first, second], tmp_path / "merged.jsonl", ("id",)) == 2
    assert [row["value"] for row in read_jsonl(tmp_path / "merged.jsonl")] == [2, 3]


def test_runtime_package_versions_skip_uninstalled_packages(monkeypatch):
    from importlib import metadata

    def version(package):
        if package == "present":
            return "1.2.3"
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr("zcp_test.artifacts.run.metadata.version", version)

    assert _package_versions(("present", "missing")) == {"present": "1.2.3"}


def test_spaces_and_cache_keys():
    load_builtin_spaces()
    expected = {"nb201_topology", "nats_size", "nb101_dag", "nb101_toy_legacy", "darts", "darts_toy_legacy", "transnas_micro", "transnas_macro", "autoformer", "pit", "zennas_plainnet_mbv2", "ofa_proxyless_mbv2", "ofa_mbv3"}
    assert expected == set(SPACES.names())
    architecture = SPACES.create("darts").sample(1)
    assert cache_key(architecture, "er", "cifar10", 1, "x") != cache_key(architecture, "er", "cifar100", 1, "x")


def test_proxy_state_isolation():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = evaluate_proxy("gradnorm", model, torch.randn(2, 3, 8, 8), torch.tensor([0, 1]), torch.nn.CrossEntropyLoss())
    assert result.status.value == "ok"
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    ("proxy_id", "output", "message"),
    [
        (
            "contract_wrong_primary",
            ProxyOutput(1.0, primary_component="aux", components={"aux": 1.0}),
            "returned primary component 'aux', but declared 'score'",
        ),
        (
            "contract_undeclared_component",
            ProxyOutput(1.0, components={"score": 1.0, "hidden": 2.0}),
            "returned undeclared components: hidden",
        ),
        (
            "contract_inconsistent_score",
            ProxyOutput(1.0, components={"score": 2.0}),
            "primary component value inconsistent with score",
        ),
    ],
)
def test_proxy_output_must_match_declared_capability(monkeypatch, proxy_id, output, message):
    monkeypatch.setitem(
        PROXIES._entries,
        proxy_id,
        lambda: FunctionProxy(ProxyCapability(proxy_id), lambda *args: output),
    )

    result = evaluate_proxy(proxy_id, torch.nn.Linear(2, 2), torch.ones(1, 2))

    assert result.status.value == "failed"
    assert message in result.error_message


def test_params_and_flops_separate_accuracy_and_resource_directions():
    load_builtin_proxies()
    for proxy_id, version in (("params", "count-v2"), ("flops", "thop-v2")):
        capability = PROXIES.create(proxy_id).capability
        assert capability.version == version
        assert capability.direction.value == "maximize"
        assert capability.resource_direction.value == "minimize"


def test_proxy_state_isolation_removes_injected_buffers():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.SiLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    )
    before = set(model.state_dict())
    inputs = torch.randn(2, 3, 8, 8)
    assert evaluate_proxy("naswot", model, inputs).status.value == "ok"
    assert evaluate_proxy("flops", model, inputs).status.value == "ok"
    assert set(model.state_dict()) == before


def test_synflow_uses_float64_for_deep_models_and_restores_dtype():
    layers = []
    for _ in range(50):
        layer = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.constant_(layer.weight, 2.0)
        layers.extend((layer, torch.nn.ReLU()))
    model = torch.nn.Sequential(*layers)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    result = evaluate_proxy("synflow", model, torch.ones(2, 4))

    assert result.status.value == "ok"
    assert result.score is not None and math.isfinite(result.score)
    assert result.proxy_version == "double-v2"
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert PROXIES.create("synflow").capability.requires_data is False


def test_proxy_can_be_explicitly_marked_unsupported_by_input_contract():
    model = torch.nn.Linear(2, 2)

    result = evaluate_proxy(
        "params",
        model,
        unsupported_reason="task-specific labels are unavailable",
    )

    assert result.status.value == "unsupported"
    assert result.error_message == "task-specific labels are unavailable"
    assert result.primary_component == "score"


def test_failed_multicomponent_proxy_preserves_declared_primary_component():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))

    result = evaluate_proxy(
        "az_nas",
        model,
        torch.randn(2, 1, 2, 2),
        labels=None,
        loss_fn=None,
    )

    assert result.status.value == "failed"
    assert result.primary_component == "expressivity"


def test_all_builtin_proxies_have_finite_cpu_contracts_and_provenance():
    load_builtin_spaces()
    load_builtin_proxies()
    architecture = SPACES.create("nb201_topology").sample(11)
    model = SPACES.create("nb201_topology").build_model(architecture, 3)
    inputs = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 0])
    for proxy_id in PROXIES.names():
        result = evaluate_proxy(
            proxy_id,
            model,
            inputs,
            labels,
            torch.nn.CrossEntropyLoss(),
        )
        assert result.status.value == "ok", (proxy_id, result.error_message)
        assert result.score is not None and math.isfinite(result.score)
        assert result.implementation_fidelity != "unverified"


def test_declared_transformer_proxies_execute_on_transformer_model():
    from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer

    load_builtin_proxies()
    model = StaticAutoFormer(
        profile=AZNAS_SCRATCH_PROFILE,
        image_size=32,
        patch_size=16,
        num_classes=3,
        embed_dim=24,
        depth=2,
        num_heads=[2, 2],
        mlp_ratio=[2.0, 2.0],
        super_depth=14,
    )
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 0])
    supported = [
        proxy_id
        for proxy_id in PROXIES.names()
        if "transformer" in PROXIES.create(proxy_id).capability.model_families
    ]
    for proxy_id in supported:
        result = evaluate_proxy(
            proxy_id,
            model,
            inputs,
            labels,
            torch.nn.CrossEntropyLoss(),
            model_family="transformer",
        )
        assert result.status.value == "ok", (proxy_id, result.error_message)
        assert result.score is not None and math.isfinite(result.score)


def test_checkpoint_loading_requires_trust(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 0}, path)
    with pytest.raises(PermissionError):
        load_checkpoint(path)
    assert load_checkpoint(path, trusted=True)["epoch"] == 0


def test_statistics_and_search(tmp_path):
    assert correlation_summary([1, 2, 3], [2, 4, 6])["spearman"] == 1.0
    load_builtin_spaces()
    search = EvolutionSearch(
        SPACES.create("darts"),
        lambda architecture: float(int(architecture.architecture_id[:8], 16)),
        JsonlWriter(tmp_path / "search.jsonl", 1),
        4,
        seed=3,
        record_metadata={"weight_mode": "inherited_supernet"},
    )
    assert search.run(1).architecture.search_space_id == "darts"
    assert all(
        row["weight_mode"] == "inherited_supernet"
        for row in read_jsonl(tmp_path / "search.jsonl")
    )


def test_evolution_search_resume_restores_population_rng_cache_and_history(tmp_path):
    load_builtin_spaces()
    space = SPACES.create("darts")

    def evaluator(architecture):
        return float(int(architecture.architecture_id[:8], 16))

    identity = {
        "search_space_id": "darts",
        "proxy_id": "fixture",
        "input_fingerprint": "batch-a",
        "seed": 19,
    }
    first_log = tmp_path / "first.jsonl"
    state_path = tmp_path / "search-state.json"
    EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(first_log, 1),
        6,
        seed=19,
        state_path=state_path,
        state_identity=identity,
    ).run(1)

    resumed_log = tmp_path / "resumed.jsonl"
    resumed_log.touch()
    resumed = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(resumed_log, 1),
        6,
        seed=19,
        state_path=tmp_path / "resumed-state.json",
        resume_state=load_search_state(state_path),
        state_identity=identity,
    ).run(3)
    uninterrupted_log = tmp_path / "uninterrupted.jsonl"
    uninterrupted = EvolutionSearch(
        space,
        evaluator,
        JsonlWriter(uninterrupted_log, 1),
        6,
        seed=19,
    ).run(3)

    def scientific_trace(path):
        return [
            (
                row["record_kind"],
                row["generation"],
                row.get("architecture_id"),
                row.get("parents"),
                row.get("operation"),
                row.get("score"),
                row["cumulative_evaluations"],
                row["cumulative_cache_hits"],
            )
            for row in read_jsonl(path)
        ]

    assert resumed.architecture.architecture_id == uninterrupted.architecture.architecture_id
    assert scientific_trace(resumed_log) == scientific_trace(uninterrupted_log)
    summaries = [
        row["generation"]
        for row in read_jsonl(resumed_log)
        if row["record_kind"] == "generation_summary"
    ]
    assert summaries == [0, 1, 2, 3]
    assert not list(tmp_path.glob(".*search-state*.tmp"))

    mismatched = dict(identity, input_fingerprint="batch-b")
    with pytest.raises(ValueError, match="identity does not match"):
        EvolutionSearch(
            space,
            evaluator,
            JsonlWriter(tmp_path / "bad.jsonl", 1),
            6,
            seed=19,
            resume_state=load_search_state(state_path),
            state_identity=mismatched,
        )


def test_training_artifacts(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(2, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    train_model(model, loader, loader, TrainingConfig(1, "sgd", 0.01, 0), tmp_path, torch.device("cpu"))
    assert (tmp_path / "checkpoints" / "last.pt").exists()


def test_autoformer_optimizer_exempts_bias_norm_and_tokens_from_weight_decay():
    from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer

    model = StaticAutoFormer(
        profile=AZNAS_SCRATCH_PROFILE,
        image_size=32,
        patch_size=16,
        num_classes=3,
        embed_dim=24,
        depth=2,
        num_heads=[2, 2],
        mlp_ratio=[2.0, 2.0],
        super_depth=14,
    )
    groups = _optimizer_parameter_groups(model, 0.05, True)
    decay_ids = {id(parameter) for parameter in groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
    names = {name: id(parameter) for name, parameter in model.named_parameters()}

    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    assert names["class_token"] in no_decay_ids
    assert names["position_embedding"] in no_decay_ids
    assert names["blocks.0.attention.relative_key.vertical"] in no_decay_ids
    assert names["blocks.0.attention.relative_value.horizontal"] in no_decay_ids
    assert names["norm.weight"] in no_decay_ids
    assert names["head.bias"] in no_decay_ids
    assert names["head.weight"] in decay_ids
    assert not decay_ids & no_decay_ids


def test_autoformer_cosine_schedule_matches_aznas_warmup_and_floor():
    config = TrainingConfig(
        500,
        "adamw",
        5e-4,
        0.05,
        warmup_epochs=20,
        warmup_learning_rate=1e-6,
        minimum_learning_rate=1e-5,
    )

    assert _cosine_learning_rate(config, 0, 500) == pytest.approx(1e-6)
    assert _cosine_learning_rate(config, 20, 500) == pytest.approx(5e-4)
    assert _cosine_learning_rate(config, 500, 500) == pytest.approx(1e-5)
    assert _cosine_learning_rate(config, 499, 500) > 1e-5


def test_validation_uses_plain_cross_entropy_when_training_uses_smoothing(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 3))
    inputs = torch.randn(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels), batch_size=2, shuffle=False
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.01,
        0.0,
        scheduler="none",
        label_smoothing=0.2,
        validation_label_smoothing=0.0,
        nesterov=False,
    )
    train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "last.pt", trusted=True)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(model(inputs), labels).item()
        smoothed = torch.nn.functional.cross_entropy(
            model(inputs), labels, label_smoothing=0.2
        ).item()

    assert record["valid_loss"] == pytest.approx(expected)
    assert record["valid_loss"] != pytest.approx(smoothed)


def test_training_keeps_label_smoothing_without_mixup(tmp_path, monkeypatch):
    observed: list[float] = []
    original = torch.nn.functional.cross_entropy

    def recording_cross_entropy(inputs, targets, *args, **kwargs):
        if torch.is_grad_enabled():
            observed.append(float(kwargs.get("label_smoothing", 0.0)))
        return original(inputs, targets, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", recording_cross_entropy)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 3))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 1])
        ),
        batch_size=2,
        shuffle=False,
    )
    config = TrainingConfig(
        1,
        "sgd",
        0.01,
        0.0,
        scheduler="none",
        label_smoothing=0.2,
        validation_label_smoothing=0.0,
        nesterov=False,
    )

    train_model(model, loader, loader, config, tmp_path, torch.device("cpu"))

    assert observed
    assert set(observed) == {0.2}


def test_checkpoint_config_normalization_accepts_missing_default_fields():
    current = TrainingConfig(1, "sgd", 0.1, 0.0)
    legacy = dict(current.__dict__)
    for key in (
        "warmup_learning_rate",
        "minimum_learning_rate",
        "validation_label_smoothing",
        "exclude_bias_norm_from_weight_decay",
    ):
        legacy.pop(key)

    assert _normalized_checkpoint_config(legacy) == current.__dict__

    legacy["minimum_learning_rate"] = 1e-5
    assert _normalized_checkpoint_config(legacy) != current.__dict__


@pytest.mark.parametrize(
    ("training_mode", "acceptance_protocol", "expected_fraction"),
    [
        ("real_data_preflight", None, 1.0),
        ("acceptance_smoke", "one_percent_data_protocol", 0.01),
        ("acceptance_smoke", "one_percent_epochs_protocol", 1.0),
    ],
)
def test_checkpoint_identity_normalizes_only_known_legacy_fraction_protocols(
    training_mode, acceptance_protocol, expected_fraction
):
    identity = {
        "training_mode": training_mode,
        "acceptance_protocol": acceptance_protocol,
    }

    assert _normalized_checkpoint_identity(identity)["data_fraction"] == expected_fraction

    unknown = {"training_mode": "formal", "acceptance_protocol": None}
    assert "data_fraction" not in _normalized_checkpoint_identity(unknown)


def test_completed_checkpoint_does_not_require_rng_restore(tmp_path, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(
        torch.randn(4, 3, 4, 4), torch.randint(2, (4,))
    )
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    config = TrainingConfig(1, "sgd", 0.01, 0.0, scheduler="none", nesterov=False)
    source = tmp_path / "source"
    train_model(model, loader, loader, config, source, torch.device("cpu"))
    checkpoint_path = source / "checkpoints" / "last.pt"
    checkpoint = load_checkpoint(checkpoint_path, trusted=True)
    checkpoint.pop("rng_by_rank", None)
    atomic_torch_save(checkpoint, checkpoint_path)
    monkeypatch.setattr(
        "zcp_test.training.trainer._restore_checkpoint_rng",
        lambda *args, **kwargs: pytest.fail("RNG restore should not run after the final epoch"),
    )

    result = train_model(
        model,
        loader,
        loader,
        config,
        tmp_path / "resumed",
        torch.device("cpu"),
        checkpoint_path,
        resume_trusted=True,
    )

    assert result["last_epoch"] == 0
    assert result["resumed_training_rows"] == 1


def test_training_scheduler_dispatch_and_resume_identity(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    config = TrainingConfig(
        1,
        "sgd",
        0.01,
        0,
        scheduler="step",
        scheduler_gamma=0.5,
        nesterov=False,
    )
    identity = {
        "architecture_id": "a",
        "protocol": "test",
        "acceptance_protocol": "one_percent_data_protocol",
        "data_fraction": 0.01,
    }
    train_model(
        model,
        loader,
        loader,
        config,
        tmp_path,
        torch.device("cpu"),
        run_identity=identity,
    )
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    assert record["next_learning_rate"] == pytest.approx(0.005)
    assert record["train_duration_seconds"] > 0
    assert record["valid_duration_seconds"] > 0
    assert record["train_samples_per_second"] > 0
    assert record["valid_samples_per_second"] > 0
    assert record["peak_memory_mb"] is None
    assert record["peak_reserved_memory_mb"] is None
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "last.pt", trusted=True)
    assert checkpoint["run_identity"] == identity
    resumed = tmp_path / "resumed"
    result = train_model(
        torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2)),
        loader,
        loader,
        config,
        resumed,
        torch.device("cpu"),
        tmp_path / "checkpoints" / "last.pt",
        resume_trusted=True,
        run_identity=identity,
    )
    assert result["resumed_training_rows"] == 1
    assert [row["epoch"] for row in read_jsonl(resumed / "training.jsonl")] == [0]
    with pytest.raises(ValueError, match="identity"):
        train_model(
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2)),
            loader,
            loader,
            config,
            tmp_path / "resume",
            torch.device("cpu"),
            tmp_path / "checkpoints" / "last.pt",
            resume_trusted=True,
            run_identity={**identity, "data_fraction": 0.0100001},
        )


def test_short_training_prefix_preserves_formal_cosine_schedule(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        model,
        loader,
        loader,
        TrainingConfig(
            1,
            "sgd",
            0.025,
            0,
            nesterov=False,
            drop_path_prob=0.2,
            schedule_epochs=600,
        ),
        tmp_path,
        torch.device("cpu"),
    )
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    assert record["learning_rate"] == pytest.approx(0.025)
    assert record["next_learning_rate"] == pytest.approx(
        0.025 * 0.5 * (1 + math.cos(math.pi / 600))
    )
    assert record["drop_path_prob"] == 0.0


def test_restore_training_log_handles_missing_source_duplicates_and_non_primary(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"epoch": 0, "valid_accuracy": 1.0}\n'
        '{"epoch": 1, "valid_accuracy": 2.0}\n'
        '{"epoch": 2, "valid_accuracy": 3.0}\n',
        encoding="utf-8",
    )
    destination = JsonlWriter(tmp_path / "destination.jsonl", fsync_every=1)
    destination.append({"epoch": 1, "valid_accuracy": 2.0})
    assert _restore_training_log(destination, source, checkpoint_epoch=1) == 1
    assert [row["epoch"] for row in read_jsonl(destination.path)] == [1, 0]
    assert _restore_training_log(destination, source, checkpoint_epoch=1) == 0
    assert _restore_training_log(destination, tmp_path / "missing.jsonl", 1) == 0
    assert _restore_training_log(None, source, 1) == 0

    portable = JsonlWriter(tmp_path / "portable.jsonl", fsync_every=1)
    history = [{"epoch": 0}, {"epoch": 1}, {"epoch": 2}]
    assert _restore_training_log(portable, tmp_path / "moved.jsonl", 1, history) == 2
    assert [row["epoch"] for row in read_jsonl(portable.path)] == [0, 1]


def test_atomic_checkpoint_removes_temporary_file_on_failure(monkeypatch, tmp_path):
    def fail_save(payload, path):
        del payload
        Path(path).write_bytes(b"partial")
        raise InterruptedError("injected")

    monkeypatch.setattr(torch, "save", fail_save)
    target = tmp_path / "checkpoint.pt"
    with pytest.raises(InterruptedError, match="injected"):
        atomic_torch_save({"epoch": 1}, target)
    assert not target.exists()
    assert not target.with_suffix(".pt.tmp").exists()


def test_training_sets_epoch_on_stateful_samplers(tmp_path):
    class EpochSampler(torch.utils.data.Sampler[int]):
        def __init__(self, dataset):
            self.dataset = dataset
            self.epochs = []

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self):
            return len(self.dataset)

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    train_sampler = EpochSampler(data)
    valid_sampler = EpochSampler(data)
    train_loader = torch.utils.data.DataLoader(data, batch_size=2, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(data, batch_size=2, sampler=valid_sampler)
    train_model(
        model,
        train_loader,
        valid_loader,
        TrainingConfig(2, "sgd", 0.01, 0, nesterov=False),
        tmp_path,
        torch.device("cpu"),
    )
    assert train_sampler.epochs == [0, 1]
    assert valid_sampler.epochs == [0, 1]


def test_training_gradient_accumulation_steps_and_no_sync(tmp_path):
    from contextlib import contextmanager

    class AccumulationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2)
            )
            self.no_sync_calls = 0

        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield

        def forward(self, inputs):
            return self.network(inputs)

    model = AccumulationModel()
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 4, 4), torch.randint(2, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        model,
        loader,
        loader,
        TrainingConfig(
            1,
            "sgd",
            0.01,
            0,
            nesterov=False,
            gradient_accumulation_steps=2,
        ),
        tmp_path,
        torch.device("cpu"),
    )
    record = next(read_jsonl(tmp_path / "training.jsonl"))
    assert record["optimizer_steps"] == 2
    assert model.no_sync_calls == 2


def test_non_primary_distributed_rank_does_not_write_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, state: output.__setitem__(slice(None), [state, state]),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(2, (4,)))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    train_model(
        model,
        loader,
        loader,
        TrainingConfig(1, "sgd", 0.01, 0, nesterov=False),
        tmp_path,
        torch.device("cpu"),
    )
    assert not (tmp_path / "training.jsonl").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_distributed_checkpoint_collects_and_restores_rank_local_rng(monkeypatch):
    rank_one_state = {"rank": 1}
    monkeypatch.setattr(
        "zcp_test.training.trainer.rng_state", lambda: rank_one_state
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(states, local_state):
        states[:] = [{"rank": 0}, local_state]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    checkpoint_rng = _collect_checkpoint_rng(True, 1)
    assert checkpoint_rng == {
        "rng": {"rank": 0},
        "rng_by_rank": [{"rank": 0}, rank_one_state],
    }

    restored = []
    monkeypatch.setattr("zcp_test.training.trainer.restore_rng", restored.append)
    _restore_checkpoint_rng(checkpoint_rng, True, 1)
    assert restored == [rank_one_state]


def test_distributed_resume_rejects_legacy_or_wrong_world_size_rng(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="rank-local RNG"):
        _restore_checkpoint_rng({"rng": {"rank": 0}}, True, 1)
    with pytest.raises(ValueError, match="world size"):
        _restore_checkpoint_rng(
            {"rng": {"rank": 0}, "rng_by_rank": [{"rank": 0}]}, True, 1
        )
