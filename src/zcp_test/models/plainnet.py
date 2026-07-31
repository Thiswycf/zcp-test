from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

import torch
from torch import Tensor, nn


AZNAS_COMMIT = "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47"
ZENNAS_COMMIT = "d1d617e0352733d39890fb64ea758f9c85b28c1a"
OFFICIAL_COMPLEXITY_VERSION = "aznas-5e6683-plainnet-get-flops-v1"
INITIAL_STRUCTURE = (
    "SuperConvK3BNRELU(3,8,2,1)"
    "SuperResIDWE6K3(8,32,2,8,1)"
    "SuperResIDWE6K3(32,48,2,32,1)"
    "SuperResIDWE6K3(48,96,2,48,1)"
    "SuperResIDWE6K3(96,128,2,96,1)"
    "SuperConvK1BNRELU(128,2048,1,1)"
)

_TOKEN = re.compile(r"([A-Za-z][A-Za-z0-9]*)\(([^()]*)\)")
_RESIDUAL = re.compile(r"SuperResIDWE([1246])K([357])")
_CONV = re.compile(r"SuperConvK([13])BNRELU")


@dataclass(frozen=True)
class PlainNetBlockSpec:
    kind: str
    in_channels: int
    out_channels: int
    stride: int
    sub_layers: int
    kernel_size: int
    expansion: int | None = None
    bottleneck_channels: int | None = None

    def encode(self) -> str:
        if self.kind == "conv":
            return (
                f"SuperConvK{self.kernel_size}BNRELU("
                f"{self.in_channels},{self.out_channels},{self.stride},{self.sub_layers})"
            )
        return (
            f"SuperResIDWE{self.expansion}K{self.kernel_size}("
            f"{self.in_channels},{self.out_channels},{self.stride},"
            f"{self.bottleneck_channels},{self.sub_layers})"
        )


def parse_plainnet_structure(structure: str) -> tuple[PlainNetBlockSpec, ...]:
    compact = "".join(structure.split())
    if not compact:
        raise ValueError("PlainNet structure cannot be empty")
    blocks: list[PlainNetBlockSpec] = []
    position = 0
    for match in _TOKEN.finditer(compact):
        if match.start() != position:
            raise ValueError(f"Unsupported PlainNet syntax at offset {position}")
        position = match.end()
        name, argument_text = match.groups()
        try:
            arguments = tuple(int(value) for value in argument_text.split(","))
        except ValueError as error:
            raise ValueError(f"PlainNet block {name!r} requires integer arguments") from error
        conv = _CONV.fullmatch(name)
        residual = _RESIDUAL.fullmatch(name)
        if conv:
            if len(arguments) != 4:
                raise ValueError(f"{name} requires four arguments")
            channels_in, channels_out, stride, sub_layers = arguments
            block = PlainNetBlockSpec(
                "conv", channels_in, channels_out, stride, sub_layers, int(conv.group(1))
            )
        elif residual:
            if len(arguments) != 5:
                raise ValueError(f"{name} requires five arguments")
            channels_in, channels_out, stride, bottleneck, sub_layers = arguments
            block = PlainNetBlockSpec(
                "residual",
                channels_in,
                channels_out,
                stride,
                sub_layers,
                int(residual.group(2)),
                int(residual.group(1)),
                bottleneck,
            )
        else:
            raise ValueError(f"Unsupported PlainNet block type {name!r}")
        if min(block.in_channels, block.out_channels, block.stride, block.sub_layers) <= 0:
            raise ValueError("PlainNet channel, stride, and sub-layer values must be positive")
        if block.stride not in {1, 2}:
            raise ValueError("PlainNet stride must be 1 or 2")
        if block.bottleneck_channels is not None and block.bottleneck_channels <= 0:
            raise ValueError("PlainNet bottleneck channels must be positive")
        if blocks and block.in_channels != blocks[-1].out_channels:
            raise ValueError("PlainNet adjacent block channels do not match")
        blocks.append(block)
    if position != len(compact):
        raise ValueError(f"Unsupported PlainNet syntax at offset {position}")
    if not blocks or blocks[0].in_channels != 3:
        raise ValueError("PlainNet structure must start with three input channels")
    return tuple(blocks)


def canonical_plainnet_structure(blocks: Iterable[PlainNetBlockSpec]) -> str:
    return "".join(block.encode() for block in blocks)


class _ConvBnRelu(nn.Sequential):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride: int,
        *,
        groups: int = 1,
        activation: bool = True,
        bn_momentum: float = 0.1,
    ) -> None:
        modules: list[nn.Module] = [
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(channels_out, eps=1e-3, momentum=bn_momentum),
        ]
        if activation:
            modules.append(nn.ReLU(inplace=True))
        super().__init__(*modules)


class _PlainNetSE(nn.Module):
    def __init__(self, channels: int, bn_momentum: float) -> None:
        super().__init__()
        reduced = max(1, round(channels * 0.25))
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced, eps=1e-3, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=bn_momentum),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.layers(inputs)


class _ResidualUnit(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        hidden_channels: int,
        kernel_size: int,
        stride: int,
        *,
        force_projection: bool,
        use_se: bool,
        bn_momentum: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _ConvBnRelu(channels_in, hidden_channels, 1, 1, bn_momentum=bn_momentum),
            _ConvBnRelu(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride,
                groups=hidden_channels,
                bn_momentum=bn_momentum,
            ),
        ]
        if use_se:
            layers.append(_PlainNetSE(hidden_channels, bn_momentum))
        projection = _ConvBnRelu(
            hidden_channels,
            channels_out,
            1,
            1,
            activation=False,
            bn_momentum=bn_momentum,
        )
        layers.append(projection)
        self.branch = nn.Sequential(*layers)
        if force_projection or stride != 1 or channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride),
                nn.BatchNorm2d(channels_out, eps=1e-3, momentum=bn_momentum),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.final_batch_norm = projection[1]

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.branch(inputs) + self.shortcut(inputs))


class _PlainNetResidualBlock(nn.Sequential):
    def __init__(self, spec: PlainNetBlockSpec, use_se: bool, bn_momentum: float) -> None:
        assert spec.expansion is not None and spec.bottleneck_channels is not None
        units: list[nn.Module] = []
        channels_in = spec.in_channels
        stride = spec.stride
        for _ in range(spec.sub_layers):
            first_hidden = _round_channels(spec.bottleneck_channels * spec.expansion)
            units.append(
                _ResidualUnit(
                    channels_in,
                    spec.bottleneck_channels,
                    first_hidden,
                    spec.kernel_size,
                    stride,
                    force_projection=True,
                    use_se=use_se,
                    bn_momentum=bn_momentum,
                )
            )
            second_hidden = _round_channels(spec.out_channels * spec.expansion)
            units.append(
                _ResidualUnit(
                    spec.bottleneck_channels,
                    spec.out_channels,
                    second_hidden,
                    spec.kernel_size,
                    1,
                    force_projection=False,
                    use_se=use_se,
                    bn_momentum=bn_momentum,
                )
            )
            channels_in = spec.out_channels
            stride = 1
        super().__init__(*units)


def _round_channels(value: float) -> int:
    return max(8, int(value + 4) // 8 * 8)


class AZPlainNetMobileNetV2(nn.Module):
    model_fidelity = "reference_model"

    def __init__(
        self,
        structure: str,
        *,
        num_classes: int = 1000,
        use_se: bool = False,
        bn_momentum: float = 0.1,
        initialization: str = "kaiming",
    ) -> None:
        super().__init__()
        self.block_specs = parse_plainnet_structure(structure)
        self.structure = canonical_plainnet_structure(self.block_specs)
        self.use_se = bool(use_se)
        self.bn_momentum = float(bn_momentum)
        modules: list[nn.Module] = []
        for spec in self.block_specs:
            if spec.kind == "conv":
                layers: list[nn.Module] = []
                channels_in = spec.in_channels
                stride = spec.stride
                for _ in range(spec.sub_layers):
                    layers.append(
                        _ConvBnRelu(
                            channels_in,
                            spec.out_channels,
                            spec.kernel_size,
                            stride,
                            bn_momentum=self.bn_momentum,
                        )
                    )
                    channels_in = spec.out_channels
                    stride = 1
                modules.append(nn.Sequential(*layers))
            else:
                modules.append(_PlainNetResidualBlock(spec, self.use_se, self.bn_momentum))
        self.blocks = nn.ModuleList(modules)
        self.classifier = nn.Linear(self.block_specs[-1].out_channels, num_classes)
        feature_relus: list[nn.ReLU] = []
        for spec, block in zip(self.block_specs, self.blocks, strict=True):
            if spec.kind == "conv":
                feature_relus.extend(layer[-1] for layer in block)
            else:
                for unit in block:
                    feature_relus.extend(
                        (unit.branch[0][-1], unit.branch[1][-1], unit.activation)
                    )
        self._feature_relus = tuple(feature_relus)
        self.init_parameters(initialization)

    def init_parameters(self, method: str = "kaiming") -> None:
        if method not in {"kaiming", "xavier", "custom"}:
            raise ValueError(f"Unsupported PlainNet initialization {method!r}")
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    if method == "kaiming":
                        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                    elif method == "xavier":
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        nn.init.xavier_normal_(module.weight, gain=3.26033)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.fill_(1)
                    module.bias.zero_()
                elif isinstance(module, nn.Linear):
                    if method == "kaiming":
                        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                    elif method == "xavier":
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        standard_deviation = 3.26033 * math.sqrt(
                            2 / (module.weight.shape[0] + module.weight.shape[1])
                        )
                        nn.init.normal_(module.weight, 0, standard_deviation)
                    if module.bias is not None:
                        module.bias.zero_()
            for module in self.modules():
                if isinstance(module, _ResidualUnit):
                    module.final_batch_norm.weight.zero_()

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("PlainNet inputs must have shape [batch, 3, height, width]")
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))

    def extract_layer_features_and_logit(self, inputs: Tensor) -> tuple[list[Tensor], Tensor]:
        features: list[Tensor] = []

        def capture_feature(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            if output.requires_grad:
                output.retain_grad()
            features.append(output)

        handles = [module.register_forward_hook(capture_feature) for module in self._feature_relus]
        try:
            logits = self(inputs)
        finally:
            for handle in handles:
                handle.remove()
        return features, logits

    def official_complexity_ops(self, input_resolution: int = 224) -> float:
        if isinstance(input_resolution, bool) or not isinstance(input_resolution, int):
            raise TypeError("PlainNet input resolution must be an integer")
        if input_resolution <= 0:
            raise ValueError("PlainNet input resolution must be positive")

        resolution = input_resolution
        operations = 0.0
        for spec in self.block_specs:
            if spec.kind == "conv":
                channels_in = spec.in_channels
                stride = spec.stride
                for _ in range(spec.sub_layers):
                    operations += (
                        channels_in
                        * spec.out_channels
                        * spec.kernel_size**2
                        * resolution**2
                        // stride**2
                    )
                    resolution //= stride
                    operations += resolution**2 * spec.out_channels
                    channels_in = spec.out_channels
                    stride = 1
                continue

            assert spec.expansion is not None and spec.bottleneck_channels is not None
            channels_in = spec.in_channels
            stride = spec.stride
            for sub_layer in range(spec.sub_layers):
                hidden_channels = _round_channels(
                    spec.bottleneck_channels * spec.expansion
                )
                operations += _official_conv_bn_ops(
                    channels_in, hidden_channels, 1, 1, resolution
                )
                operations += _official_depthwise_bn_ops(
                    hidden_channels, spec.kernel_size, stride, resolution
                )
                resolution //= stride
                if self.use_se:
                    operations += _official_se_ops(hidden_channels, resolution)
                operations += _official_conv_bn_ops(
                    hidden_channels, spec.bottleneck_channels, 1, 1, resolution
                )
                if (
                    sub_layer == 0
                    or stride > 1
                    or channels_in != spec.bottleneck_channels
                ):
                    projected_resolution = resolution / stride
                    operations += (
                        channels_in
                        * spec.bottleneck_channels
                        * projected_resolution**2
                        + projected_resolution**2 * spec.bottleneck_channels
                    )

                hidden_channels = _round_channels(spec.out_channels * spec.expansion)
                operations += _official_conv_bn_ops(
                    spec.bottleneck_channels, hidden_channels, 1, 1, resolution
                )
                operations += _official_depthwise_bn_ops(
                    hidden_channels, spec.kernel_size, 1, resolution
                )
                if self.use_se:
                    operations += _official_se_ops(hidden_channels, resolution)
                operations += _official_conv_bn_ops(
                    hidden_channels, spec.out_channels, 1, 1, resolution
                )
                if spec.bottleneck_channels != spec.out_channels:
                    operations += (
                        spec.bottleneck_channels * spec.out_channels * resolution**2
                        + resolution**2 * spec.out_channels
                    )
                channels_in = spec.out_channels
                stride = 1

        operations += self.classifier.in_features * self.classifier.out_features
        return float(operations)

    def reference_metadata(self) -> dict[str, Any]:
        return {
            "family": "aznas_zennas_plainnet_mbv2",
            "model_fidelity": self.model_fidelity,
            "implementation_source": "https://github.com/cvlab-yonsei/AZ-NAS",
            "implementation_commit": AZNAS_COMMIT,
            "official_complexity_version": OFFICIAL_COMPLEXITY_VERSION,
            "search_space_source": "https://github.com/idstcv/ZenNAS",
            "search_space_commit": ZENNAS_COMMIT,
            "structure": self.structure,
            "use_se": self.use_se,
            "bn_momentum": self.bn_momentum,
            "weight_mode": "independent_scratch",
        }


def reconnect_plainnet_blocks(
    blocks: Iterable[PlainNetBlockSpec],
) -> tuple[PlainNetBlockSpec, ...]:
    connected: list[PlainNetBlockSpec] = []
    channels = 3
    for block in blocks:
        connected_block = replace(block, in_channels=channels)
        connected.append(connected_block)
        channels = connected_block.out_channels
    return tuple(connected)


def _official_conv_bn_ops(
    channels_in: int,
    channels_out: int,
    kernel_size: int,
    stride: int,
    input_resolution: int,
) -> int:
    output_resolution = input_resolution // stride
    convolution = (
        channels_in
        * channels_out
        * kernel_size**2
        * input_resolution**2
        // stride**2
    )
    return convolution + output_resolution**2 * channels_out


def _official_depthwise_bn_ops(
    channels: int, kernel_size: int, stride: int, input_resolution: int
) -> int:
    output_resolution = input_resolution // stride
    convolution = channels * kernel_size**2 * input_resolution**2 // stride**2
    return convolution + output_resolution**2 * channels


def _official_se_ops(channels: int, input_resolution: int) -> int:
    se_channels = max(1, round(channels * 0.25))
    return (
        channels * se_channels
        + se_channels * channels
        + channels
        + channels * input_resolution**2
    )


__all__ = [
    "AZNAS_COMMIT",
    "AZPlainNetMobileNetV2",
    "INITIAL_STRUCTURE",
    "OFFICIAL_COMPLEXITY_VERSION",
    "PlainNetBlockSpec",
    "ZENNAS_COMMIT",
    "canonical_plainnet_structure",
    "parse_plainnet_structure",
    "reconnect_plainnet_blocks",
]
