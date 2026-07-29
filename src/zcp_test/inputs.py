from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputBatch:
    inputs: Any
    labels: Any
    protocol: dict[str, Any]
    fingerprint: str


def _fingerprint(inputs: Any, labels: Any, protocol: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    digest.update(inputs.detach().cpu().contiguous().numpy().tobytes())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_input_batch(
    source: str,
    dataset: str,
    batch_size: int,
    input_size: int,
    classes: int,
    seed: int,
    device: Any,
    data_root: str | None = None,
) -> InputBatch:
    import torch

    if batch_size <= 0 or input_size <= 0:
        raise ValueError("batch_size and input_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if source in {"random", "noise"}:
        if source == "random":
            inputs = torch.randn(batch_size, 3, input_size, input_size, generator=generator)
        else:
            inputs = torch.rand(batch_size, 3, input_size, input_size, generator=generator)
        labels = torch.randint(classes, (batch_size,), generator=generator)
        protocol = {
            "source": source,
            "dataset": None,
            "seed": seed,
            "sample_ids": list(range(batch_size)),
            "transform": "synthetic-normal" if source == "random" else "synthetic-uniform",
            "batch_size": batch_size,
            "input_size": input_size,
            "label_protocol": "synthetic-uniform",
        }
    elif source == "dataset":
        if not data_root:
            raise ValueError("--input-source dataset requires --data-root or a configured dataset asset")
        inputs, labels, sample_ids, transform_name = _dataset_batch(
            dataset, Path(data_root).expanduser(), batch_size, input_size, seed
        )
        protocol = {
            "source": "dataset",
            "dataset": dataset,
            "seed": seed,
            "sample_ids": sample_ids,
            "transform": transform_name,
            "batch_size": batch_size,
            "input_size": input_size,
            "label_protocol": "published-labels",
            "data_root": str(Path(data_root).expanduser().resolve()),
        }
    else:
        raise ValueError(f"Unknown input source: {source}")
    fingerprint = _fingerprint(inputs, labels, protocol)
    return InputBatch(inputs.to(device), labels.to(device), protocol, fingerprint)


def _dataset_batch(
    dataset: str, root: Path, batch_size: int, input_size: int, seed: int
) -> tuple[Any, Any, list[int], str]:
    import torch
    from torchvision import datasets, transforms

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    normalization = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar10-valid": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }
    if dataset in normalization:
        mean, standard_deviation = normalization[dataset]
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, standard_deviation),
            ]
        )
        dataset_type = datasets.CIFAR10 if dataset in {"cifar10", "cifar10-valid"} else datasets.CIFAR100
        table = dataset_type(root, train=True, transform=transform, download=False)
        transform_name = f"resize-{input_size}+tensor+{dataset}-normalize"
    elif dataset in {"imagenet1k", "imagenet"}:
        train_root = root / "train" if (root / "train").exists() else root
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        table = datasets.ImageFolder(train_root, transform=transform)
        transform_name = f"resize-256+center-crop-{input_size}+imagenet-normalize"
    else:
        raise ValueError(f"Dataset input protocol is not implemented for {dataset!r}")
    if len(table) < batch_size:
        raise ValueError(f"Dataset has {len(table)} samples, fewer than batch size {batch_size}")
    sample_ids = random.Random(seed).sample(range(len(table)), batch_size)
    samples = [table[index] for index in sample_ids]
    return (
        torch.stack([sample[0] for sample in samples]),
        torch.tensor([int(sample[1]) for sample in samples], dtype=torch.long),
        sample_ids,
        transform_name,
    )
