from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import torch
from torch import Tensor, nn


class Genotype(NamedTuple):
    normal: list[tuple[str, int]]
    normal_concat: list[int]
    reduce: list[tuple[str, int]]
    reduce_concat: list[int]


class ReLUConvBN(nn.Sequential):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(channels_out, affine=affine),
        )


class DilConv(nn.Sequential):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels_in,
                channels_in,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=channels_in,
                bias=False,
            ),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out, affine=affine),
        )


class SepConv(nn.Sequential):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels_in,
                channels_in,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=channels_in,
                bias=False,
            ),
            nn.Conv2d(channels_in, channels_in, 1, bias=False),
            nn.BatchNorm2d(channels_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels_in,
                channels_in,
                kernel_size,
                stride=1,
                padding=padding,
                groups=channels_in,
                bias=False,
            ),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out, affine=affine),
        )


class Identity(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class Zero(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        if self.stride == 1:
            return inputs.mul(0.0)
        return inputs[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, affine: bool = True) -> None:
        super().__init__()
        if channels_out % 2:
            raise ValueError("FactorizedReduce requires an even output channel count")
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(channels_in, channels_out // 2, 1, stride=2, bias=False)
        self.conv_2 = nn.Conv2d(channels_in, channels_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(channels_out, affine=affine)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.relu(inputs)
        return self.bn(torch.cat([self.conv_1(inputs), self.conv_2(inputs[:, :, 1:, 1:])], dim=1))


OperationFactory = Callable[[int, int, bool], nn.Module]


OPS: dict[str, OperationFactory] = {
    "none": lambda channels, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda channels, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda channels, stride, affine: nn.MaxPool2d(
        3, stride=stride, padding=1
    ),
    "skip_connect": lambda channels, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(channels, channels, affine=affine),
    "sep_conv_3x3": lambda channels, stride, affine: SepConv(
        channels, channels, 3, stride, 1, affine
    ),
    "sep_conv_5x5": lambda channels, stride, affine: SepConv(
        channels, channels, 5, stride, 2, affine
    ),
    "sep_conv_7x7": lambda channels, stride, affine: SepConv(
        channels, channels, 7, stride, 3, affine
    ),
    "dil_conv_3x3": lambda channels, stride, affine: DilConv(
        channels, channels, 3, stride, 2, 2, affine
    ),
    "dil_conv_5x5": lambda channels, stride, affine: DilConv(
        channels, channels, 5, stride, 4, 2, affine
    ),
    "conv_7x1_1x7": lambda channels, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(channels, channels, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(channels, channels, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(channels, affine=affine),
    ),
}


def drop_path(inputs: Tensor, drop_prob: float = 0.0, training: bool = True) -> Tensor:
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError("drop_prob must be in [0, 1)")
    if drop_prob == 0.0 or not training:
        return inputs
    keep_probability = 1.0 - drop_prob
    mask_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
    mask = inputs.new_empty(mask_shape).bernoulli_(keep_probability)
    return inputs * mask / keep_probability


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs: Tensor) -> Tensor:
        return drop_path(inputs, self.drop_prob, self.training)


class Cell(nn.Module):
    def __init__(
        self,
        genotype: Genotype,
        channels_previous_previous: int,
        channels_previous: int,
        channels: int,
        reduction: bool,
        reduction_previous: bool,
    ) -> None:
        super().__init__()
        if reduction_previous:
            self.preprocess0 = FactorizedReduce(channels_previous_previous, channels)
        else:
            self.preprocess0 = ReLUConvBN(channels_previous_previous, channels, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(channels_previous, channels, 1, 1, 0)
        gene = genotype.reduce if reduction else genotype.normal
        concat = genotype.reduce_concat if reduction else genotype.normal_concat
        self._compile(channels, gene, concat, reduction)

    def _compile(
        self,
        channels: int,
        gene: Sequence[tuple[str, int]],
        concat: Sequence[int],
        reduction: bool,
    ) -> None:
        if len(gene) % 2:
            raise ValueError("A DARTS cell requires two edges per intermediate node")
        self._steps = len(gene) // 2
        self._concat = tuple(concat)
        self.multiplier = len(concat)
        self._indices = tuple(index for _, index in gene)
        operations: list[nn.Module] = []
        for operation_name, index in gene:
            if operation_name not in OPS:
                raise ValueError(f"Unknown DARTS operation: {operation_name}")
            stride = 2 if reduction and index < 2 else 1
            operations.append(OPS[operation_name](channels, stride, True))
        self._ops = nn.ModuleList(operations)

    def forward(self, state0: Tensor, state1: Tensor, drop_probability: float = 0.0) -> Tensor:
        states = [self.preprocess0(state0), self.preprocess1(state1)]
        for step in range(self._steps):
            first_op, second_op = self._ops[2 * step], self._ops[2 * step + 1]
            first = first_op(states[self._indices[2 * step]])
            second = second_op(states[self._indices[2 * step + 1]])
            if self.training and drop_probability > 0.0:
                if not isinstance(first_op, Identity):
                    first = drop_path(first, drop_probability, True)
                if not isinstance(second_op, Identity):
                    second = drop_path(second, drop_probability, True)
            states.append(first + second)
        return torch.cat([states[index] for index in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class _NetworkBase(nn.Module):
    auxiliary: bool
    auxiliary_head: nn.Module | None
    cells: nn.ModuleList

    def _forward_cells(self, state0: Tensor, state1: Tensor) -> tuple[Tensor, Tensor | None]:
        auxiliary_logits = None
        for index, cell in enumerate(self.cells):
            state0, state1 = state1, cell(state0, state1, self.drop_path_prob)
            if self.training and self.auxiliary and index == self._auxiliary_index:
                if self.auxiliary_head is not None:
                    auxiliary_logits = self.auxiliary_head(state1)
        return state1, auxiliary_logits

    def _format_output(
        self, logits: Tensor, auxiliary_logits: Tensor | None, return_auxiliary: bool
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        self.auxiliary_logits = auxiliary_logits
        if return_auxiliary:
            return logits, auxiliary_logits
        return logits


class NetworkCIFAR(_NetworkBase):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        auxiliary: bool,
        genotype: Genotype,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self.auxiliary = auxiliary
        self._auxiliary_index = 2 * layers // 3
        stem_channels = 3 * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
        )
        channels_previous_previous = channels_previous = stem_channels
        channels_current = C
        reduction_previous = False
        cells: list[Cell] = []
        channels_auxiliary = 0
        for index in range(layers):
            reduction = index in (layers // 3, 2 * layers // 3)
            if reduction:
                channels_current *= 2
            cell = Cell(
                genotype,
                channels_previous_previous,
                channels_previous,
                channels_current,
                reduction,
                reduction_previous,
            )
            reduction_previous = reduction
            cells.append(cell)
            channels_previous_previous, channels_previous = (
                channels_previous,
                cell.multiplier * channels_current,
            )
            if index == self._auxiliary_index:
                channels_auxiliary = channels_previous
        self.cells = nn.ModuleList(cells)
        self.auxiliary_head = (
            AuxiliaryHeadCIFAR(channels_auxiliary, num_classes) if auxiliary else None
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels_previous, num_classes)
        self.auxiliary_logits: Tensor | None = None

    def forward(
        self, inputs: Tensor, return_auxiliary: bool = False
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        state0 = state1 = self.stem(inputs)
        features, auxiliary_logits = self._forward_cells(state0, state1)
        logits = self.classifier(self.global_pooling(features).flatten(1))
        return self._format_output(logits, auxiliary_logits, return_auxiliary)


class NetworkImageNet(_NetworkBase):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        auxiliary: bool,
        genotype: Genotype,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self.auxiliary = auxiliary
        self._auxiliary_index = 2 * layers // 3
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        channels_previous_previous = channels_previous = C
        channels_current = C
        reduction_previous = True
        cells: list[Cell] = []
        channels_auxiliary = 0
        for index in range(layers):
            reduction = index in (layers // 3, 2 * layers // 3)
            if reduction:
                channels_current *= 2
            cell = Cell(
                genotype,
                channels_previous_previous,
                channels_previous,
                channels_current,
                reduction,
                reduction_previous,
            )
            reduction_previous = reduction
            cells.append(cell)
            channels_previous_previous, channels_previous = (
                channels_previous,
                cell.multiplier * channels_current,
            )
            if index == self._auxiliary_index:
                channels_auxiliary = channels_previous
        self.cells = nn.ModuleList(cells)
        self.auxiliary_head = (
            AuxiliaryHeadImageNet(channels_auxiliary, num_classes) if auxiliary else None
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels_previous, num_classes)
        self.auxiliary_logits: Tensor | None = None

    def forward(
        self, inputs: Tensor, return_auxiliary: bool = False
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        state0 = self.stem0(inputs)
        state1 = self.stem1(state0)
        features, auxiliary_logits = self._forward_cells(state0, state1)
        logits = self.classifier(self.global_pooling(features).flatten(1))
        return self._format_output(logits, auxiliary_logits, return_auxiliary)


__all__ = [
    "OPS",
    "AuxiliaryHeadCIFAR",
    "AuxiliaryHeadImageNet",
    "Cell",
    "DilConv",
    "DropPath",
    "FactorizedReduce",
    "Genotype",
    "Identity",
    "NetworkCIFAR",
    "NetworkImageNet",
    "ReLUConvBN",
    "SepConv",
    "Zero",
    "drop_path",
]
