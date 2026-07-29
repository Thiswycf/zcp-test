from __future__ import annotations

import re
from typing import Any

from zcp_test.types import Architecture


def num_classes(dataset: str) -> int:
    return {"cifar10": 10, "cifar10-valid": 10, "cifar100": 100, "ImageNet16-120": 120, "imagenet1k": 1000, "flowers": 102, "chaoyang": 4}.get(dataset, 10)


def nb201_model(architecture: Architecture, dataset: str) -> Any:
    import torch.nn as nn

    specification = str(architecture.spec["architecture"])
    nodes = []
    for node_text in specification.split("+"):
        edges = [(operation, int(source)) for operation, source in re.findall(r"([a-zA-Z0-9_]+)~(\d+)", node_text)]
        if edges:
            nodes.append(edges)
    if len(nodes) != 3:
        raise ValueError(f"Invalid NB201 architecture string: {specification}")

    class Operation(nn.Module):
        def __init__(self, name: str, channels: int) -> None:
            super().__init__()
            if name == "none":
                self.operation = None
            elif name == "skip_connect":
                self.operation = nn.Identity()
            elif name == "avg_pool_3x3":
                self.operation = nn.AvgPool2d(3, 1, 1)
            else:
                kernel = 1 if name == "nor_conv_1x1" else 3
                self.operation = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False), nn.BatchNorm2d(channels))

        def forward(self, inputs: Any) -> Any:
            return inputs.mul(0) if self.operation is None else self.operation(inputs)

    class Cell(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.edges = nn.ModuleList([nn.ModuleList([Operation(name, channels) for name, _ in edges]) for edges in nodes])

        def forward(self, inputs: Any) -> Any:
            states = [inputs]
            for edge_specs, edge_modules in zip(nodes, self.edges, strict=True):
                states.append(
                    sum(
                        module(states[source])
                        for (_, source), module in zip(edge_specs, edge_modules, strict=True)
                    )
                )
            return states[-1]

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = 16
            self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
            self.cells = nn.Sequential(*[Cell(channels) for _ in range(5)])
            self.head = nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, num_classes(dataset)))

        def forward(self, inputs: Any) -> Any:
            return self.head(self.cells(self.stem(inputs)))

    return Network()


def nb101_model(architecture: Architecture, dataset: str) -> Any:
    import torch.nn as nn

    matrix = architecture.spec["matrix"]
    operations = architecture.spec["operations"]
    channels = 32

    def operation(name: str) -> nn.Module:
        if name == "conv1x1-bn-relu":
            return nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        if name == "conv3x3-bn-relu":
            return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        if name == "maxpool3x3":
            return nn.MaxPool2d(3, 1, 1)
        return nn.Identity()

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, channels, 3, padding=1)
            self.vertices = nn.ModuleList([operation(name) for name in operations])
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, num_classes(dataset)))

        def forward(self, inputs: Any) -> Any:
            states = [self.stem(inputs)]
            for target in range(1, len(matrix)):
                sources = [states[source] for source in range(target) if matrix[source][target]]
                value = sum(sources) if sources else states[-1].mul(0)
                states.append(self.vertices[target](value))
            return self.head(states[-1])

    return Network()


def registered_space_model(architecture: Architecture, dataset: str) -> Any:
    from zcp_test.spaces import SPACES, load_builtin_spaces

    load_builtin_spaces()
    if architecture.search_space_id in SPACES.names():
        space = SPACES.create(architecture.search_space_id)
        try:
            canonical = space.canonicalize(architecture.spec)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot build {architecture.search_space_id} architecture "
                f"{architecture.architecture_id}: invalid specification"
            ) from error
        return space.build_model(canonical, num_classes(dataset))
    raise NotImplementedError(f"No model factory for {architecture.search_space_id}")


def model_builder(architecture: Architecture, dataset: str) -> Any:
    if architecture.search_space_id == "nb201_topology":
        return nb201_model(architecture, dataset)
    if architecture.search_space_id == "nb101_dag":
        return nb101_model(architecture, dataset)
    return registered_space_model(architecture, dataset)
