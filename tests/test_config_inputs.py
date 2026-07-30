import json
from pathlib import Path

import pytest
import torch

import zcp_test.cli as cli
from zcp_test.config import dump_config, load_config, merge_config
from zcp_test.inputs import make_dataset_batch_stream, make_input_batch
from zcp_test.training.protocols import resolve_gradient_accumulation, scale_learning_rate


class _InputDataset(torch.utils.data.Dataset):
    def __init__(self, *args, transform=None, **kwargs):
        self.transform = transform
        self.targets = [0, 1, 0, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.full((3, 8, 8), float(index)), self.targets[index]


def test_autoformer_linear_global_batch_learning_rate():
    learning_rate, global_batch_size = scale_learning_rate(0.0005, 256, 8, 512)
    assert global_batch_size == 2048
    assert learning_rate == pytest.approx(0.002)
    with pytest.raises(ValueError, match="world_size"):
        scale_learning_rate(0.0005, 256, 0, 512)


def test_autoformer_gradient_accumulation_preserves_target_global_batch():
    assert resolve_gradient_accumulation("auto", 256, 4, 2048, smoke=False) == 2
    assert resolve_gradient_accumulation("auto", 256, 8, 2048, smoke=False) == 1
    assert resolve_gradient_accumulation("auto", 2, 2, 2048, smoke=True) == 1
    assert resolve_gradient_accumulation(3, 256, 4, None, smoke=False) == 3
    with pytest.raises(ValueError, match="requires target"):
        resolve_gradient_accumulation("auto", 256, 4, None, smoke=False)
    with pytest.raises(ValueError, match="divisible"):
        resolve_gradient_accumulation("auto", 300, 4, 2048, smoke=False)
    with pytest.raises(ValueError, match="must be positive"):
        resolve_gradient_accumulation(0, 256, 4, None, smoke=False)


def test_config_load_merge_and_atomic_dump(tmp_path):
    assert load_config(None) == {}
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == {}
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(invalid)

    base = {"nested": {"left": 1, "keep": 2}, "value": 1}
    merged = merge_config(base, {"nested": {"left": 3}, "value": None, "new": Path("x")})
    assert merged == {"nested": {"left": 3, "keep": 2}, "value": 1, "new": Path("x")}
    assert base["nested"]["left"] == 1

    destination = tmp_path / "nested" / "config.yaml"
    dump_config(merged, destination)
    loaded = load_config(destination)
    assert loaded["nested"]["left"] == 3
    assert loaded["new"] == "x"
    assert not destination.with_suffix(".yaml.tmp").exists()


def test_dataset_input_protocols_and_errors(monkeypatch, tmp_path):
    from torchvision import datasets

    monkeypatch.setattr(datasets, "CIFAR10", _InputDataset)
    monkeypatch.setattr(datasets, "CIFAR100", _InputDataset)
    monkeypatch.setattr(datasets, "ImageFolder", _InputDataset)
    device = torch.device("cpu")

    cifar = make_input_batch("dataset", "cifar10-valid", 2, 8, 10, 7, device, str(tmp_path))
    assert cifar.inputs.shape == (2, 3, 8, 8)
    assert cifar.protocol["dataset"] == "cifar10-valid"
    assert cifar.protocol["label_protocol"] == "published-labels"

    (tmp_path / "train").mkdir()
    imagenet = make_input_batch("dataset", "imagenet1k", 2, 8, 1000, 8, device, str(tmp_path))
    assert imagenet.protocol["transform"].startswith("resize-256")
    assert imagenet.fingerprint != cifar.fingerprint

    with pytest.raises(ValueError, match="not implemented"):
        make_input_batch("dataset", "unknown", 2, 8, 3, 1, device, str(tmp_path))
    with pytest.raises(ValueError, match="Unknown input source"):
        make_input_batch("unsupported", "cifar10", 2, 8, 10, 1, device)
    with pytest.raises(ValueError, match="positive"):
        make_input_batch("random", "cifar10", 0, 8, 10, 1, device)


def test_dataset_batch_stream_is_deterministic_and_nonoverlapping(monkeypatch, tmp_path):
    from torchvision import datasets

    monkeypatch.setattr(datasets, "CIFAR10", _InputDataset)
    first = make_dataset_batch_stream(
        "cifar10", tmp_path, 2, 8, 17, 2, torch.device("cpu"), role="bn"
    )
    second = make_dataset_batch_stream(
        "cifar10", tmp_path, 2, 8, 17, 2, torch.device("cpu"), role="bn"
    )

    assert first.sample_ids == second.sample_ids
    assert first.fingerprint == second.fingerprint
    assert len({item for batch in first.sample_ids for item in batch}) == 4
    assert [inputs.shape for inputs, _ in first] == [(2, 3, 8, 8)] * 2
    with pytest.raises(ValueError, match="fewer than required"):
        make_dataset_batch_stream(
            "cifar10", tmp_path, 3, 8, 17, 2, torch.device("cpu"), role="bn"
        )


def test_cutout_and_proxy_scaffold_in_temporary_checkout(monkeypatch, capsys, tmp_path):
    image = torch.ones(3, 8, 8)
    output = cli.Cutout(4)(image)
    assert output.shape == image.shape
    assert torch.count_nonzero(output) < output.numel()

    fake_cli = tmp_path / "src" / "zcp_test" / "cli.py"
    (tmp_path / "src" / "zcp_test" / "proxies" / "custom").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(cli, "__file__", str(fake_cli))
    cli._scaffold_proxy("research_proxy")
    report = json.loads(capsys.readouterr().out)
    module = Path(report["module"])
    test = Path(report["test"])
    assert module.exists() and test.exists()
    assert "compute_research_proxy" in module.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli._scaffold_proxy("research_proxy")
    with pytest.raises(ValueError, match="public Python identifier"):
        cli._scaffold_proxy("invalid-name")
