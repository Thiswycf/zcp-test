import pytest
import torch

from zcp_test.artifacts import JsonlWriter, merge_jsonl, read_jsonl
from zcp_test.proxies.evaluator import evaluate_proxy
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
    expected = {"nb201_topology", "nats_size", "nb101_dag", "darts", "darts_toy_legacy", "transnas_micro", "transnas_macro", "autoformer", "pit", "ofa_proxyless_mbv2", "ofa_mbv3"}
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
    )
    assert search.run(1).architecture.search_space_id == "darts"


def test_training_artifacts(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(2, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    train_model(model, loader, loader, TrainingConfig(1, "sgd", 0.01, 0), tmp_path, torch.device("cpu"))
    assert (tmp_path / "checkpoints" / "last.pt").exists()
