from __future__ import annotations

import itertools
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
    def __init__(self, channels_out: int, stride: int) -> None:
        super().__init__()
        self.channels_out = channels_out
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs[:, :, :: self.stride, :: self.stride]
        return output.new_zeros(
            output.shape[0], self.channels_out, output.shape[2], output.shape[3]
        )


def edge_operation(code: str, channels_in: int, channels_out: int, stride: int) -> nn.Module:
    if code == "0":
        return Zero(channels_out, stride)
    if code == "1":
        if channels_in == channels_out and stride == 1:
            return nn.Identity()
        return ReLUConvBN(channels_in, channels_out, 1, stride)
    if code == "2":
        return ReLUConvBN(channels_in, channels_out, 1, stride)
    if code == "3":
        return ReLUConvBN(channels_in, channels_out, 3, stride)
    raise ValueError(f"Unknown TransNAS micro operation: {code!r}")


class MicroCell(nn.Module):
    def __init__(
        self,
        code: Sequence[str],
        channels_in: int,
        channels_out: int,
        stride: int,
    ) -> None:
        super().__init__()
        if tuple(map(len, code)) != (0, 1, 2, 3):
            raise ValueError(f"Invalid TransNAS micro code: {code!r}")
        self.code = tuple(code)
        self.edges = nn.ModuleList()
        for node_code in self.code:
            for source, operation_code in enumerate(node_code):
                self.edges.append(
                    edge_operation(
                        operation_code,
                        channels_in if source == 0 else channels_out,
                        channels_out,
                        stride if source == 0 else 1,
                    )
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = [inputs]
        edge_index = 0
        for node_code in self.code[1:]:
            node_outputs = []
            for source in range(len(node_code)):
                node_outputs.append(self.edges[edge_index](states[source]))
                edge_index += 1
            states.append(torch.stack(node_outputs).sum(0))
        return states[-1]


class BasicBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ReLUConvBN(channels_in, channels_out, 3, stride),
            ReLUConvBN(channels_out, channels_out, 3, 1),
        )
        self.residual = (
            nn.Identity()
            if channels_in == channels_out and stride == 1
            else nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs) + self.residual(inputs)


def parse_code(code: str) -> tuple[int, str, tuple[str, ...] | None]:
    parts = code.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid TransNAS architecture: {code!r}")
    base_channels = int(parts[0])
    macro = parts[1]
    if base_channels <= 0 or len(macro) not in {4, 5, 6} or any(value not in "1234" for value in macro):
        raise ValueError(f"Invalid TransNAS macro code: {code!r}")
    if parts[2] == "basic":
        return base_channels, macro, None
    micro = ("", *parts[2].split("_"))
    if tuple(map(len, micro)) != (0, 1, 2, 3) or any(
        value not in "0123" for group in micro for value in group
    ):
        raise ValueError(f"Invalid TransNAS micro code: {code!r}")
    return base_channels, macro, micro


def macro_codes() -> tuple[str, ...]:
    values = []
    for length in (4, 5, 6):
        for code in itertools.product("1234", repeat=length):
            channel_changes = sum(int(value) % 2 == 0 for value in code)
            reductions = sum(int(value) > 2 for value in code)
            if 1 <= channel_changes <= 3 and 1 <= reductions <= 4:
                values.append("".join(code))
    return tuple(values)


class TransNasNetwork(nn.Module):
    def __init__(self, code: str, num_classes: int) -> None:
        super().__init__()
        base_channels, macro, micro = parse_code(code)
        self.architecture_code = code
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            ReLUConvBN(base_channels // 2, base_channels, 3, 2),
        )
        channels = base_channels
        layers: list[nn.Module] = []
        for layer_type in map(int, macro):
            target = channels * 2 if layer_type % 2 == 0 else channels
            stride = 2 if layer_type > 2 else 1
            blocks: list[nn.Module] = []
            for block_index in range(2):
                block_stride = stride if block_index == 0 else 1
                block_channels_in = channels if block_index == 0 else target
                blocks.append(
                    BasicBlock(block_channels_in, target, block_stride)
                    if micro is None
                    else MicroCell(micro, block_channels_in, target, block_stride)
                )
            layers.append(nn.Sequential(*blocks))
            channels = target
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        for layer in self.layers:
            features = layer(features)
        return self.classifier(self.pool(features).flatten(1))
