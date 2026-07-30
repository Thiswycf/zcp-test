"""Static PiT subnet used by the released ViT-Bench-101 PiT slice."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from zcp_test.models.autoformer import DropPath


class PitMlp(nn.Sequential):
    def __init__(self, dimension: int, ratio: float, dropout: float) -> None:
        hidden = int(dimension * ratio)
        super().__init__(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )


class PitAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("PiT attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.scale = self.head_dimension**-0.5
        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.projection = nn.Linear(dimension, dimension)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, tokens, dimension = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, tokens, 3, self.num_heads, self.head_dimension
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.softmax((query @ key.transpose(-1, -2)) * self.scale, dim=-1)
        attention = self.attention_dropout(attention)
        outputs = (attention @ value).transpose(1, 2).reshape(batch, tokens, dimension)
        return self.projection_dropout(self.projection(outputs))


class PitBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attention = PitAttention(
            dimension, num_heads, attention_dropout, dropout
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = PitMlp(dimension, mlp_ratio, dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm1(inputs)
        inputs = inputs + self.drop_path(self.attention(normalized))
        return inputs + self.drop_path(self.mlp(self.norm2(inputs)))


class PitPooling(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(
            input_dimension,
            output_dimension,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=input_dimension,
        )
        self.class_projection = nn.Linear(input_dimension, output_dimension)

    def forward(self, spatial: Tensor, class_token: Tensor) -> tuple[Tensor, Tensor]:
        return self.spatial(spatial), self.class_projection(class_token)


class StaticPiT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        patch_stride: int = 8,
        in_channels: int = 3,
        num_classes: int = 100,
        base_dim: int,
        depth: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        depths = tuple(int(value) for value in depth)
        heads = tuple(int(value) for value in num_heads)
        if len(depths) != 3 or len(heads) != 3:
            raise ValueError("PiT requires exactly three stage depths and head counts")
        if any(value <= 0 for value in depths + heads) or base_dim <= 0:
            raise ValueError("PiT dimensions, depths, and head counts must be positive")
        if mlp_ratio <= 0:
            raise ValueError("PiT mlp_ratio must be positive")
        if patch_size > image_size or patch_stride <= 0:
            raise ValueError("PiT patch geometry is incompatible with the image size")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.base_dim = int(base_dim)
        self.depth = depths
        self.num_heads = heads
        self.mlp_ratio = float(mlp_ratio)
        self.stage_dimensions = tuple(self.base_dim * value for value in heads)
        grid = (image_size - patch_size) // patch_stride + 1

        self.patch_embed = nn.Conv2d(
            in_channels,
            self.stage_dimensions[0],
            kernel_size=patch_size,
            stride=patch_stride,
        )
        self.position = nn.Parameter(torch.empty(1, self.stage_dimensions[0], grid, grid))
        self.class_token = nn.Parameter(torch.empty(1, 1, self.stage_dimensions[0]))
        self.position_dropout = nn.Dropout(dropout)

        total_blocks = sum(depths)
        probabilities = [drop_path_rate * index / total_blocks for index in range(total_blocks)]
        block_index = 0
        stages: list[nn.ModuleList] = []
        for dimension, stage_depth, stage_heads in zip(
            self.stage_dimensions, depths, heads, strict=True
        ):
            blocks = nn.ModuleList(
                PitBlock(
                    dimension,
                    stage_heads,
                    self.mlp_ratio,
                    dropout,
                    attention_dropout,
                    probabilities[block_index + offset],
                )
                for offset in range(stage_depth)
            )
            stages.append(blocks)
            block_index += stage_depth
        self.stages = nn.ModuleList(stages)
        self.pools = nn.ModuleList(
            PitPooling(self.stage_dimensions[index], self.stage_dimensions[index + 1])
            for index in range(2)
        )
        self.norm = nn.LayerNorm(self.stage_dimensions[-1], eps=1e-6)
        self.head = nn.Linear(self.stage_dimensions[-1], num_classes)
        self._initialize()

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _tokens(spatial: Tensor, class_token: Tensor) -> Tensor:
        return torch.cat((class_token, spatial.flatten(2).transpose(1, 2)), dim=1)

    @staticmethod
    def _spatial(tokens: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
        class_token = tokens[:, :1]
        spatial = tokens[:, 1:].transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        return spatial, class_token

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"StaticPiT expects {self.image_size}x{self.image_size} inputs")
        spatial = self.patch_embed(inputs)
        spatial = self.position_dropout(spatial + self.position)
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        for stage_index, blocks in enumerate(self.stages):
            height, width = spatial.shape[-2:]
            tokens = self._tokens(spatial, class_token)
            for block in blocks:
                tokens = block(tokens)
            spatial, class_token = self._spatial(tokens, height, width)
            if stage_index < len(self.pools):
                spatial, class_token = self.pools[stage_index](spatial, class_token)
        return self.norm(class_token)[:, 0]

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.forward_features(inputs))

    def reference_metadata(self) -> dict[str, Any]:
        return {
            "family": "pit_static_subnet",
            "model_fidelity": "reference_topology_pytorch_port",
            "implementation_source": "lliai/Auto-Prox-AAAI24",
            "implementation_commit": "90ed458",
            "weight_mode": "independent_scratch",
            "supports_inherited_supernet": False,
            "architecture": {
                "base_dim": self.base_dim,
                "depth": list(self.depth),
                "num_heads": list(self.num_heads),
                "mlp_ratio": self.mlp_ratio,
            },
        }


__all__ = ["PitAttention", "StaticPiT"]
