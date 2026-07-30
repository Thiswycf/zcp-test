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


class TransNasEncoder(nn.Module):
    def __init__(self, code: str, input_size: int = 256) -> None:
        super().__init__()
        base_channels, macro, micro = parse_code(code)
        self.architecture_code = code
        self.input_size = input_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            ReLUConvBN(base_channels // 2, base_channels, 3, 2),
        )
        channels = base_channels
        feature_size = input_size // 4
        layers: list[nn.Module] = []
        for layer_type in map(int, macro):
            target = channels * 2 if layer_type % 2 == 0 else channels
            stride = 2 if layer_type > 2 else 1
            feature_size //= stride
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
        self.output_channels = channels
        self.output_size = feature_size
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        for layer in self.layers:
            features = layer(features)
        return features


class TransNasNetwork(nn.Module):
    def __init__(self, code: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = TransNasEncoder(code)
        self.architecture_code = code
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.encoder.output_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        return self.classifier(self.pool(features).flatten(1))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        *,
        transpose: bool = False,
        activation: nn.Module | None = None,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        convolution: nn.Module
        if transpose:
            convolution = nn.ConvTranspose2d(
                channels_in, channels_out, 3, stride=2, padding=1, output_padding=1
            )
        else:
            convolution = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.convolution = convolution
        self.batch_norm = nn.BatchNorm2d(channels_out) if batch_norm else nn.Identity()
        self.activation = activation or nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.convolution(inputs)))


class SegmentationDecoder(nn.Module):
    def __init__(self, channels_in: int, feature_size: int, classes: int = 17) -> None:
        super().__init__()
        upsample_count = int(torch.tensor(256 / feature_size).log2().item())
        if upsample_count not in {2, 3, 4, 5, 6}:
            raise ValueError("TransNAS segmentation decoder requires 2..6 upsampling stages")
        layers: list[nn.Module] = [
            DecoderLayer(channels_in, 1024, activation=nn.LeakyReLU(0.2, inplace=False))
        ]
        channels = 1024
        for _ in range(6 - upsample_count):
            layers.append(
                DecoderLayer(
                    channels,
                    channels // 2,
                    activation=nn.LeakyReLU(0.2, inplace=False),
                )
            )
            channels //= 2
        for _ in range(upsample_count):
            layers.append(
                DecoderLayer(
                    channels,
                    channels // 2 if len(layers) < 6 else channels,
                    transpose=True,
                    activation=nn.LeakyReLU(0.2, inplace=False),
                )
            )
            channels = layers[-1].convolution.out_channels
        layers.append(DecoderLayer(channels, classes, batch_norm=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class GenerativeDecoder(nn.Module):
    def __init__(self, channels_in: int, feature_size: int) -> None:
        super().__init__()
        upsample_count = int(torch.tensor(256 / feature_size).log2().item())
        if upsample_count not in {2, 3, 4, 5, 6}:
            raise ValueError("TransNAS generative decoder requires 2..6 upsampling stages")
        specifications = (
            (channels_in, 1024, False),
            (1024, 1024, False),
            (1024, 512, upsample_count == 6),
            (512, 512, False),
            (512, 256, upsample_count >= 5),
            (256, 128, False),
            (128, 64, upsample_count >= 4),
            (64, 64, False),
            (64, 32, upsample_count >= 3),
            (32, 32, False),
            (32, 16, True),
            (16, 32, False),
            (32, 16, True),
        )
        self.layers = nn.Sequential(
            *[
                DecoderLayer(
                    channels_in_layer,
                    channels_out,
                    transpose=transpose,
                    activation=nn.LeakyReLU(0.2, inplace=False),
                )
                for channels_in_layer, channels_out, transpose in specifications
            ],
            DecoderLayer(
                16,
                3,
                activation=nn.Tanh(),
                batch_norm=False,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


TRANSNAS_TASK_OUTPUTS = {
    "class_scene": 47,
    "class_object": 75,
    "room_layout": 9,
    "jigsaw": 1000,
    "segmentsemantic": 17,
    "normal": 3,
    "autoencoder": 3,
}


class TransNasTaskModel(nn.Module):
    implementation_commit = "6d4231b1eb04e95750a5b2b6cf391db770bc25d6"

    def __init__(self, code: str, task: str) -> None:
        super().__init__()
        if task not in TRANSNAS_TASK_OUTPUTS:
            raise ValueError(f"Unknown TransNAS task: {task!r}")
        self.task = task
        encoder_input_size = 64 if task == "jigsaw" else 256
        self.encoder = TransNasEncoder(code, encoder_input_size)
        outputs = TRANSNAS_TASK_OUTPUTS[task]
        if task in {"class_scene", "class_object", "room_layout"}:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.decoder = nn.Linear(self.encoder.output_channels, outputs)
        elif task == "jigsaw":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.decoder = nn.Linear(self.encoder.output_channels * 9, outputs)
        elif task == "segmentsemantic":
            self.decoder = SegmentationDecoder(
                self.encoder.output_channels, self.encoder.output_size, outputs
            )
        else:
            self.decoder = GenerativeDecoder(
                self.encoder.output_channels, self.encoder.output_size
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.task == "jigsaw":
            if inputs.ndim != 5 or inputs.shape[1:3] != (9, 3):
                raise ValueError("TransNAS jigsaw input must have shape [batch, 9, 3, H, W]")
            batch_size = inputs.shape[0]
            features = self.encoder(inputs.flatten(0, 1))
            pooled = self.pool(features).flatten(1).reshape(batch_size, -1)
            return self.decoder(pooled)
        features = self.encoder(inputs)
        if self.task in {"class_scene", "class_object", "room_layout"}:
            features = self.pool(features).flatten(1)
        return self.decoder(features)

    def reference_metadata(self) -> dict[str, object]:
        return {
            "model_fidelity": "reference_topology_pytorch_port",
            "model_protocol": "official-encoder-and-task-head-pytorch-port",
            "implementation_source": "https://github.com/yawen-d/TransNASBench",
            "implementation_commit": self.implementation_commit,
            "task": self.task,
            "output_channels": TRANSNAS_TASK_OUTPUTS[self.task],
            "expected_input_size": 64 if self.task == "jigsaw" else 256,
        }
