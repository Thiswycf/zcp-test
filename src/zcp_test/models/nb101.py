from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def compute_vertex_channels(
    input_channels: int, output_channels: int, matrix: list[list[int]]
) -> list[int]:
    vertices = len(matrix)
    channels = [0] * vertices
    channels[0], channels[-1] = input_channels, output_channels
    if vertices == 2:
        return channels
    output_fan_in = sum(matrix[source][-1] for source in range(1, vertices - 1))
    if output_fan_in <= 0:
        raise ValueError("NAS-Bench-101 module output has no interior fan-in")
    base, correction = divmod(output_channels, output_fan_in)
    for vertex in range(1, vertices - 1):
        if matrix[vertex][-1]:
            channels[vertex] = base + int(correction > 0)
            correction = max(0, correction - 1)
    for vertex in range(vertices - 3, 0, -1):
        if not matrix[vertex][-1]:
            channels[vertex] = max(
                (channels[target] for target in range(vertex + 1, vertices - 1) if matrix[vertex][target]),
                default=0,
            )
        if channels[vertex] <= 0:
            raise ValueError("NAS-Bench-101 vertex has no path to output")
    return channels


class ConvBnRelu(nn.Sequential):
    def __init__(self, inputs: int, outputs: int, kernel: int) -> None:
        super().__init__(
            nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=False),
        )


class VertexOperation(nn.Module):
    def __init__(self, name: str, channels: int) -> None:
        super().__init__()
        if name == "conv1x1-bn-relu":
            self.operation = ConvBnRelu(channels, channels, 1)
        elif name == "conv3x3-bn-relu":
            self.operation = ConvBnRelu(channels, channels, 3)
        elif name == "maxpool3x3":
            self.operation = nn.MaxPool2d(3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported NAS-Bench-101 operation: {name}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.operation(inputs)


class Module(nn.Module):
    def __init__(self, specification: Mapping[str, Any], input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.matrix = [list(map(int, row)) for row in specification["matrix"]]
        self.operations = list(specification["operations"])
        self.channels = compute_vertex_channels(input_channels, output_channels, self.matrix)
        self.input_projections = nn.ModuleDict(
            {
                str(vertex): ConvBnRelu(input_channels, self.channels[vertex], 1)
                for vertex in range(1, len(self.matrix) - 1)
                if self.matrix[0][vertex]
            }
        )
        self.output_projection = (
            ConvBnRelu(input_channels, output_channels, 1) if self.matrix[0][-1] else None
        )
        self.vertex_operations = nn.ModuleDict(
            {
                str(vertex): VertexOperation(self.operations[vertex], self.channels[vertex])
                for vertex in range(1, len(self.matrix) - 1)
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = [inputs]
        output_inputs = []
        captured = getattr(self, "_edge_activation_capture", None)
        prefix = getattr(self, "_edge_activation_prefix", "nb101")
        for target in range(1, len(self.matrix) - 1):
            incoming = []
            for source in range(target):
                if not self.matrix[source][target]:
                    continue
                if source == 0:
                    edge_value = self.input_projections[str(target)](states[0])
                else:
                    edge_value = states[source][:, : self.channels[target]]
                incoming.append(edge_value)
                if captured is not None:
                    captured.append((f"{prefix}:{source}", f"{prefix}:{target}", edge_value))
            if not incoming:
                raise RuntimeError(f"NAS-Bench-101 vertex {target} has no input")
            value = incoming[0] if len(incoming) == 1 else torch.stack(incoming).sum(0)
            states.append(self.vertex_operations[str(target)](value))
            if self.matrix[target][-1]:
                output_inputs.append(states[target])
                if captured is not None:
                    captured.append(
                        (f"{prefix}:{target}", f"{prefix}:{len(self.matrix) - 1}", states[target])
                    )
        if output_inputs:
            output = output_inputs[0] if len(output_inputs) == 1 else torch.cat(output_inputs, dim=1)
            if self.output_projection is not None:
                projected = self.output_projection(inputs)
                output = output + projected
                if captured is not None:
                    captured.append(
                        (f"{prefix}:0", f"{prefix}:{len(self.matrix) - 1}", projected)
                    )
            return output
        if self.output_projection is None:
            raise RuntimeError("NAS-Bench-101 output is disconnected")
        projected = self.output_projection(inputs)
        if captured is not None:
            captured.append(
                (f"{prefix}:0", f"{prefix}:{len(self.matrix) - 1}", projected)
            )
        return projected


class Network(nn.Module):
    def __init__(self, specification: Mapping[str, Any], num_classes: int) -> None:
        super().__init__()
        channels = 128
        self.stem = ConvBnRelu(3, channels, 3)
        layers: list[nn.Module] = []
        for stack in range(3):
            if stack:
                layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))
                output_channels = channels * 2
            else:
                output_channels = channels
            for _ in range(3):
                layers.append(Module(specification, channels, output_channels))
                channels = output_channels
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.layers(self.stem(inputs))
        return self.classifier(features.mean(dim=(2, 3)))


__all__ = ["Module", "Network", "compute_vertex_channels"]
