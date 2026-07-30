"""Static MobileNetV2-family models for independent scratch training.

PlainNet-style search models follow the public ZenNAS representation:
https://github.com/idstcv/ZenNAS

The residual MBConv model follows the public ProxylessNAS and OFA static
building blocks:
https://github.com/mit-han-lab/proxylessnas
https://github.com/mit-han-lab/once-for-all

No dynamic supernet, inherited weights, or predictor is implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(channels_out),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedBottleneck(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        expand_ratio: float,
        stride: int,
        use_residual: bool,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, int(round(channels_in * expand_ratio)))
        layers: list[nn.Module] = []
        if hidden_channels != channels_in:
            layers.append(ConvBnAct(channels_in, hidden_channels, 1))
        layers.extend(
            [
                ConvBnAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride=stride,
                    groups=hidden_channels,
                ),
                ConvBnAct(hidden_channels, channels_out, 1, activation=False),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.use_residual = use_residual

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.layers(inputs)
        return inputs + outputs if self.use_residual else outputs


def make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = make_divisible(channels // 4)
        self.reduce = nn.Conv2d(channels, reduced, 1, bias=True)
        self.expand = nn.Conv2d(reduced, channels, 1, bias=True)

    def forward(self, inputs: Tensor) -> Tensor:
        scale = inputs.mean((2, 3), keepdim=True)
        scale = torch.relu(self.reduce(scale))
        scale = torch.nn.functional.relu6(self.expand(scale) + 3.0, inplace=True) / 6.0
        return inputs * scale


def mobile_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "h_swish":
        return nn.Hardswish(inplace=True)
    raise ValueError(f"Unknown MobileNetV3 activation: {name}")


class MobileNetV3Conv(nn.Sequential):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        activation: str,
        batch_norm: bool = True,
        bias: bool = False,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=bias,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(channels_out, eps=1e-5, momentum=0.1))
        layers.append(mobile_activation(activation))
        super().__init__(*layers)


class MobileNetV3Block(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        expand_ratio: float,
        stride: int,
        activation: str,
        use_se: bool,
    ) -> None:
        super().__init__()
        hidden = int(round(channels_in * expand_ratio))
        self.inverted = (
            MobileNetV3Conv(
                channels_in, hidden, 1, activation=activation
            )
            if expand_ratio != 1
            else nn.Identity()
        )
        depthwise: list[nn.Module] = [
            MobileNetV3Conv(
                hidden,
                hidden,
                kernel_size,
                stride=stride,
                groups=hidden,
                activation=activation,
            )
        ]
        if use_se:
            depthwise.append(SqueezeExcite(hidden))
        self.depthwise = nn.Sequential(*depthwise)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out, eps=1e-5, momentum=0.1),
        )
        self.use_residual = stride == 1 and channels_in == channels_out

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.project(self.depthwise(self.inverted(inputs)))
        return inputs + outputs if self.use_residual else outputs


class StaticMobileNetV3(nn.Module):
    model_fidelity = "reference_model"

    def __init__(
        self,
        *,
        num_classes: int,
        width_mult: float,
        stage_depths: Sequence[int],
        kernel_sizes: Sequence[int],
        expand_ratios: Sequence[float],
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        depths = tuple(int(value) for value in stage_depths)
        kernels = tuple(int(value) for value in kernel_sizes)
        expansions = tuple(float(value) for value in expand_ratios)
        if len(depths) != 5 or any(value not in (2, 3, 4) for value in depths):
            raise ValueError("OFA-MBV3 requires five stage depths in {2, 3, 4}")
        if len(kernels) != 20 or any(value not in (3, 5, 7) for value in kernels):
            raise ValueError("OFA-MBV3 kernel_sizes must contain 20 values in {3, 5, 7}")
        if len(expansions) != 20 or any(value not in (3, 4, 6) for value in expansions):
            raise ValueError("OFA-MBV3 expand_ratios must contain 20 values in {3, 4, 6}")
        if width_mult not in (1.0, 1.2):
            raise ValueError("OFA-MBV3 width_mult must be 1.0 or 1.2")

        widths = [make_divisible(value * width_mult) for value in (16, 16, 24, 40, 80, 112, 160)]
        final_expand = make_divisible(960 * width_mult)
        last_channel = make_divisible(1280 * width_mult)
        self.width_mult = width_mult
        self.stage_depths = depths
        self.kernel_sizes = kernels
        self.expand_ratios = expansions
        self.first_conv = MobileNetV3Conv(3, widths[0], 3, stride=2, activation="h_swish")
        self.first_block = MobileNetV3Block(
            widths[0], widths[1], 3, 1, 1, "relu", False
        )

        stage_strides = (2, 2, 2, 1, 2)
        stage_activations = ("relu", "relu", "h_swish", "h_swish", "h_swish")
        stage_se = (False, True, False, True, True)
        stages: list[nn.Sequential] = []
        channels_in = widths[1]
        for stage_index, (channels_out, depth, stride, activation, use_se) in enumerate(
            zip(
                widths[2:],
                depths,
                stage_strides,
                stage_activations,
                stage_se,
                strict=True,
            )
        ):
            blocks: list[nn.Module] = []
            for block_index in range(depth):
                encoding_index = stage_index * 4 + block_index
                blocks.append(
                    MobileNetV3Block(
                        channels_in,
                        channels_out,
                        kernels[encoding_index],
                        expansions[encoding_index],
                        stride if block_index == 0 else 1,
                        activation,
                        use_se,
                    )
                )
                channels_in = channels_out
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.final_expand = MobileNetV3Conv(
            channels_in, final_expand, 1, activation="h_swish"
        )
        self.feature_mix = MobileNetV3Conv(
            final_expand,
            last_channel,
            1,
            activation="h_swish",
            batch_norm=False,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.first_block(self.first_conv(inputs))
        for stage in self.stages:
            outputs = stage(outputs)
        outputs = self.final_expand(outputs).mean((2, 3), keepdim=True)
        outputs = self.feature_mix(outputs).flatten(1)
        return self.classifier(self.dropout(outputs))

    def reference_metadata(self) -> dict[str, Any]:
        return {
            "family": "ofa_static_mbv3",
            "model_fidelity": "reference_model",
            "implementation_source": "mit-han-lab/once-for-all",
            "implementation_commit": "f03b2673db313b9167e2a1c2b7a5cad540cc1313",
            "weight_mode": "independent_scratch",
            "supports_inherited_supernet": False,
            "supports_bn_recalibration": True,
            "architecture": {
                "kernel_size": list(self.kernel_sizes),
                "expand_ratio": list(self.expand_ratios),
                "depth": list(self.stage_depths),
                "width_mult": self.width_mult,
            },
        }


def recalibrate_batch_norm(
    model: nn.Module,
    batches: Iterable[Tensor | Sequence[Tensor]],
    *,
    device: torch.device | str,
    max_batches: int | None = None,
) -> int:
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if not batch_norms:
        raise ValueError("model has no BatchNorm2d modules to recalibrate")
    training_states = {module: module.training for module in model.modules()}
    momenta = {module: module.momentum for module in batch_norms}
    model.eval()
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    processed = 0
    try:
        with torch.no_grad():
            for batch in batches:
                if max_batches is not None and processed >= max_batches:
                    break
                inputs = batch[0] if isinstance(batch, Sequence) else batch
                model(inputs.to(device))
                processed += 1
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, training in training_states.items():
            module.train(training)
    if processed == 0:
        raise ValueError("BatchNorm recalibration requires at least one batch")
    return processed


def _validate_architecture(
    *,
    num_classes: int,
    stem_channels: int,
    head_channels: int,
    stage_channels: tuple[int, ...],
    stage_depths: tuple[int, ...],
    stage_strides: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    expand_ratios: tuple[float, ...],
) -> None:
    if num_classes <= 0 or stem_channels <= 0 or head_channels <= 0:
        raise ValueError("class count, stem channels, and head channels must be positive")
    if not stage_channels:
        raise ValueError("at least one stage is required")
    if not (len(stage_channels) == len(stage_depths) == len(stage_strides)):
        raise ValueError("stage_channels, stage_depths, and stage_strides must have equal length")
    if any(value <= 0 for value in stage_channels) or any(value <= 0 for value in stage_depths):
        raise ValueError("stage channels and depths must be positive")
    if any(value not in (1, 2) for value in stage_strides):
        raise ValueError("stage strides must be 1 or 2")
    block_count = sum(stage_depths)
    if len(kernel_sizes) != block_count or len(expand_ratios) != block_count:
        raise ValueError("kernel_sizes and expand_ratios must contain one value per active block")
    if any(value not in (3, 5, 7) for value in kernel_sizes):
        raise ValueError("kernel sizes must be one of 3, 5, or 7")
    if any(value <= 0 for value in expand_ratios):
        raise ValueError("expand ratios must be positive")


class _MobileNetV2Base(nn.Module):
    model_fidelity = "reference_model"

    def __init__(
        self,
        *,
        num_classes: int,
        stem_channels: int,
        head_channels: int,
        stage_channels: Sequence[int],
        stage_depths: Sequence[int],
        stage_strides: Sequence[int],
        kernel_sizes: Sequence[int],
        expand_ratios: Sequence[float],
        skip: Sequence[bool] | None,
        residual_style: bool,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in stage_channels)
        depths = tuple(int(value) for value in stage_depths)
        strides = tuple(int(value) for value in stage_strides)
        kernels = tuple(int(value) for value in kernel_sizes)
        expansions = tuple(float(value) for value in expand_ratios)
        _validate_architecture(
            num_classes=num_classes,
            stem_channels=stem_channels,
            head_channels=head_channels,
            stage_channels=channels,
            stage_depths=depths,
            stage_strides=strides,
            kernel_sizes=kernels,
            expand_ratios=expansions,
        )
        if skip is not None and len(skip) != sum(depths):
            raise ValueError("skip must contain one boolean per active block")
        if skip is not None and any(not isinstance(value, bool) for value in skip):
            raise TypeError("skip values must be booleans")
        if not residual_style and skip is not None and any(skip):
            raise ValueError("PlainNetMobileNetV2 does not support residual skips")

        self.num_classes = num_classes
        self.stem_channels = stem_channels
        self.head_channels = head_channels
        self.stage_channels = channels
        self.stage_depths = depths
        self.stage_strides = strides
        self.kernel_sizes = kernels
        self.expand_ratios = expansions
        self.residual_style = residual_style
        self.stem = ConvBnAct(3, stem_channels, 3, stride=2)
        blocks: list[nn.Module] = []
        actual_skips: list[bool] = []
        channels_in = stem_channels
        block_index = 0
        requested_skips = None if skip is None else tuple(skip)
        for stage_index, depth in enumerate(depths):
            channels_out = channels[stage_index]
            for depth_index in range(depth):
                stride = strides[stage_index] if depth_index == 0 else 1
                eligible_for_skip = stride == 1 and channels_in == channels_out
                if residual_style:
                    use_residual = (
                        eligible_for_skip
                        if requested_skips is None
                        else requested_skips[block_index]
                    )
                    if use_residual and not eligible_for_skip:
                        raise ValueError(
                            f"block {block_index} cannot use skip with stride/channel change"
                        )
                else:
                    use_residual = False
                blocks.append(
                    InvertedBottleneck(
                        channels_in,
                        channels_out,
                        kernels[block_index],
                        expansions[block_index],
                        stride,
                        use_residual,
                    )
                )
                actual_skips.append(use_residual)
                channels_in = channels_out
                block_index += 1
        self.skip = tuple(actual_skips)
        self.blocks = nn.Sequential(*blocks)
        self.head = ConvBnAct(channels_in, head_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_channels, num_classes)

    def _metadata(self, family: str, sources: list[str]) -> dict[str, Any]:
        return {
            "family": family,
            "model_fidelity": self.model_fidelity,
            "weight_mode": "independent_scratch",
            "supports_inherited_supernet": False,
            "sources": sources,
            "architecture": {
                "stem_channels": self.stem_channels,
                "head_channels": self.head_channels,
                "stage_channels": list(self.stage_channels),
                "stage_depths": list(self.stage_depths),
                "stage_strides": list(self.stage_strides),
                "kernel_sizes": list(self.kernel_sizes),
                "expand_ratios": list(self.expand_ratios),
                "skip": list(self.skip),
            },
        }

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("MobileNetV2 inputs must have shape [batch, 3, height, width]")
        features = self.head(self.blocks(self.stem(inputs)))
        return self.pool(features).flatten(1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))


class PlainNetMobileNetV2(_MobileNetV2Base):
    """A skip-free PlainNet MobileNetV2-style static network."""

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        stem_channels: int = 32,
        head_channels: int = 1280,
        stage_channels: Sequence[int],
        stage_depths: Sequence[int],
        stage_strides: Sequence[int],
        kernel_sizes: Sequence[int],
        expand_ratios: Sequence[float],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            stem_channels=stem_channels,
            head_channels=head_channels,
            stage_channels=stage_channels,
            stage_depths=stage_depths,
            stage_strides=stage_strides,
            kernel_sizes=kernel_sizes,
            expand_ratios=expand_ratios,
            skip=None,
            residual_style=False,
        )

    def reference_metadata(self) -> dict[str, Any]:
        return self._metadata(
            "plainnet_mbv2",
            ["https://github.com/idstcv/ZenNAS"],
        )


class StaticMobileNetV2(_MobileNetV2Base):
    """A Proxyless/OFA-style static MBConv network with explicit skips."""

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        stem_channels: int = 32,
        head_channels: int = 1280,
        stage_channels: Sequence[int],
        stage_depths: Sequence[int],
        stage_strides: Sequence[int],
        kernel_sizes: Sequence[int],
        expand_ratios: Sequence[float],
        skip: Sequence[bool] | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            stem_channels=stem_channels,
            head_channels=head_channels,
            stage_channels=stage_channels,
            stage_depths=stage_depths,
            stage_strides=stage_strides,
            kernel_sizes=kernel_sizes,
            expand_ratios=expand_ratios,
            skip=skip,
            residual_style=True,
        )

    def reference_metadata(self) -> dict[str, Any]:
        return self._metadata(
            "proxyless_ofa_static_mbv2",
            [
                "https://github.com/mit-han-lab/proxylessnas",
                "https://github.com/mit-han-lab/once-for-all",
            ],
        )


__all__ = ["PlainNetMobileNetV2", "StaticMobileNetV2"]
