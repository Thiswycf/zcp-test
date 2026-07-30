from __future__ import annotations

import re
from collections.abc import Sequence

import torch
import torch.nn as nn


class ReLUConvBN(nn.Sequential):
    def __init__(self, channels_in: int, channels_out: int, kernel: int, stride: int) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel,
                stride=stride,
                padding=kernel // 2,
                bias=False,
            ),
            nn.BatchNorm2d(channels_out),
        )


class Zero(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int) -> None:
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs[:, :, :: self.stride, :: self.stride]
        if self.channels_in == self.channels_out:
            return output.mul(0)
        if self.channels_in > self.channels_out:
            return output[:, : self.channels_out].mul(0)
        return torch.nn.functional.pad(
            output.mul(0), (0, 0, 0, 0, 0, self.channels_out - self.channels_in)
        )


class Pooling(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int) -> None:
        super().__init__()
        self.preprocess = (
            nn.Identity()
            if channels_in == channels_out
            else ReLUConvBN(channels_in, channels_out, 1, 1)
        )
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.preprocess(inputs))


def operation(name: str, channels_in: int, channels_out: int, stride: int) -> nn.Module:
    if name == "none":
        return Zero(channels_in, channels_out, stride)
    if name == "skip_connect":
        if stride == 1 and channels_in == channels_out:
            return nn.Identity()
        return ReLUConvBN(channels_in, channels_out, 1, stride)
    if name == "avg_pool_3x3":
        return Pooling(channels_in, channels_out, stride)
    if name == "nor_conv_1x1":
        return ReLUConvBN(channels_in, channels_out, 1, stride)
    if name == "nor_conv_3x3":
        return ReLUConvBN(channels_in, channels_out, 3, stride)
    raise ValueError(f"Unknown NAS-Bench-201 operation: {name!r}")


def parse_architecture(specification: str) -> tuple[tuple[tuple[str, int], ...], ...]:
    nodes = tuple(
        tuple((name, int(source)) for name, source in re.findall(r"([a-zA-Z0-9_]+)~(\d+)", node))
        for node in specification.split("+")
    )
    if tuple(len(node) for node in nodes) != (1, 2, 3):
        raise ValueError(f"Invalid NAS-Bench-201 architecture string: {specification}")
    if any(source >= node_index for node_index, node in enumerate(nodes, 1) for _, source in node):
        raise ValueError(f"Invalid NAS-Bench-201 source node: {specification}")
    return nodes


class InferCell(nn.Module):
    def __init__(self, specification: str, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.nodes = parse_architecture(specification)
        self.edges = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        operation(
                            name,
                            channels_in if source == 0 else channels_out,
                            channels_out,
                            1,
                        )
                        for name, source in node
                    ]
                )
                for node in self.nodes
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = [inputs]
        for specifications, modules in zip(self.nodes, self.edges, strict=True):
            states.append(
                sum(
                    module(states[source])
                    for (_, source), module in zip(specifications, modules, strict=True)
                )
            )
        return states[-1]


class ResNetBasicBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int) -> None:
        super().__init__()
        self.conv_a = ReLUConvBN(channels_in, channels_out, 3, stride)
        self.conv_b = ReLUConvBN(channels_out, channels_out, 3, 1)
        self.downsample = (
            nn.Identity()
            if stride == 1 and channels_in == channels_out
            else nn.Sequential(
                nn.AvgPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(channels_in, channels_out, 1, bias=False),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_b(self.conv_a(inputs)) + self.downsample(inputs)


class TinyNetwork(nn.Module):
    def __init__(
        self,
        specification: str,
        num_classes: int,
        channels: int = 16,
        cells_per_stage: int = 5,
        stage_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if stage_channels is None:
            layer_channels = (
                [channels] * cells_per_stage
                + [channels * 2]
                + [channels * 2] * cells_per_stage
                + [channels * 4]
                + [channels * 4] * cells_per_stage
            )
        else:
            layer_channels = [int(value) for value in stage_channels]
            if len(layer_channels) % 3 != 2:
                raise ValueError("NATS-SSS channels must have 3N+2 entries")
            cells_per_stage = len(layer_channels) // 3
        reductions = (
            [False] * cells_per_stage
            + [True]
            + [False] * cells_per_stage
            + [True]
            + [False] * cells_per_stage
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, layer_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(layer_channels[0]),
        )
        previous = layer_channels[0]
        layers: list[nn.Module] = []
        for current, reduction in zip(layer_channels, reductions, strict=True):
            layer = (
                ResNetBasicBlock(previous, current, 2)
                if reduction
                else InferCell(specification, previous, current)
            )
            layers.append(layer)
            previous = current
        self.cells = nn.ModuleList(layers)
        self.last_activation = nn.Sequential(nn.BatchNorm2d(previous), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(previous, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        for cell in self.cells:
            features = cell(features)
        features = self.pool(self.last_activation(features)).flatten(1)
        return self.classifier(features)


NATS_SSS_GENOTYPE = (
    "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+"
    "|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|"
)


def build_nb201(specification: str, num_classes: int) -> TinyNetwork:
    return TinyNetwork(specification, num_classes)


def build_nats_sss(channels: Sequence[int], num_classes: int) -> TinyNetwork:
    return TinyNetwork(
        NATS_SSS_GENOTYPE,
        num_classes,
        stage_channels=channels,
    )
