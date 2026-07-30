import pytest
import torch
import math

from zcp_test.artifacts import JsonlWriter, merge_jsonl, read_jsonl
from zcp_test.proxies.evaluator import evaluate_proxy
from zcp_test.proxies import PROXIES, load_builtin_proxies
from zcp_test.reporting import correlation_summary
from zcp_test.search import EvolutionSearch, cache_key
from zcp_test.spaces import SPACES, load_builtin_spaces
from zcp_test.training import TrainingConfig, train_model
from zcp_test.training.checkpoint import load_checkpoint


def test_jsonl_merge_and_partial_recovery(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    JsonlWriter(first, 1).append({"id": "a", "value": 1})
    JsonlWriter(second, 1).append({"id": "a", "value": 2})
    JsonlWriter(second, 1).append({"id": "b", "value": 3})
    assert merge_jsonl([first, second], tmp_path / "merged.jsonl", ("id",)) == 2
    assert [row["value"] for row in read_jsonl(tmp_path / "merged.jsonl")] == [2, 3]


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


def test_proxy_can_be_explicitly_marked_unsupported_by_input_contract():
    model = torch.nn.Linear(2, 2)

    result = evaluate_proxy(
        "params",
        model,
        unsupported_reason="task-specific labels are unavailable",
    )

    assert result.status.value == "unsupported"
    assert result.error_message == "task-specific labels are unavailable"


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
    from zcp_test.models.autoformer import StaticAutoFormer

    load_builtin_proxies()
    model = StaticAutoFormer(
        image_size=32,
        patch_size=16,
        num_classes=3,
        embed_dim=24,
        depth=2,
        num_heads=[2, 2],
        mlp_ratio=[2.0, 2.0],
        qkv_head_dim=8,
        relative_position=False,
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


def test_training_artifacts(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(2, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    train_model(model, loader, loader, TrainingConfig(1, "sgd", 0.01, 0), tmp_path, torch.device("cpu"))
    assert (tmp_path / "checkpoints" / "last.pt").exists()


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
    identity = {"architecture_id": "a", "protocol": "test"}
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
            run_identity={"architecture_id": "different", "protocol": "test"},
        )


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
