"""Source-pinned static AutoFormer subnet models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn


VITBENCH_AUTOPROX_PROFILE = "vitbench-autoprox-90ed458"
AZNAS_SCRATCH_PROFILE = "aznas-scratch-5e6683"
AUTOFORMER_PROFILES = frozenset({VITBENCH_AUTOPROX_PROFILE, AZNAS_SCRATCH_PROFILE})


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep_probability)
        return inputs * mask / keep_probability


class RelativePosition2D(nn.Module):
    def __init__(self, head_dim: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.vertical = nn.Parameter(torch.empty(max_distance * 2 + 2, head_dim))
        self.horizontal = nn.Parameter(torch.empty(max_distance * 2 + 2, head_dim))
        nn.init.trunc_normal_(self.vertical, std=0.02)
        nn.init.trunc_normal_(self.horizontal, std=0.02)

    def forward(self, token_count: int) -> Tensor:
        patch_count = token_count - 1
        side = int(patch_count**0.5)
        if side * side != patch_count:
            raise ValueError("relative position requires a square patch grid")
        coordinates = torch.arange(patch_count, device=self.vertical.device)
        vertical = coordinates[None, :] // side - coordinates[:, None] // side
        horizontal = coordinates[None, :] % side - coordinates[:, None] % side
        vertical = vertical.clamp(-self.max_distance, self.max_distance) + self.max_distance + 1
        horizontal = horizontal.clamp(-self.max_distance, self.max_distance) + self.max_distance + 1
        vertical = torch.nn.functional.pad(vertical, (1, 0, 1, 0))
        horizontal = torch.nn.functional.pad(horizontal, (1, 0, 1, 0))
        return self.vertical[vertical.long()] + self.horizontal[horizontal.long()]


class AutoFormerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        scale_dim: int,
        attention_dropout: float,
        projection_dropout: float,
        relative_position: bool,
        max_relative_position: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = scale_dim**-0.5
        self.qkv = nn.Linear(embed_dim, self.inner_dim * 3)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(self.inner_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.relative_key = (
            RelativePosition2D(head_dim, max_relative_position)
            if relative_position
            else None
        )
        self.relative_value = (
            RelativePosition2D(head_dim, max_relative_position)
            if relative_position
            else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, tokens, _ = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query @ key.transpose(-2, -1)).mul(self.scale)
        if self.relative_key is not None:
            relative_key = self.relative_key(tokens)
            relative_logits = (
                query.permute(2, 0, 1, 3).reshape(tokens, self.num_heads * batch, -1)
                @ relative_key.transpose(2, 1)
            )
            attention = attention + relative_logits.transpose(1, 0).reshape(
                batch, self.num_heads, tokens, tokens
            ).mul(self.scale)
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        outputs = (attention @ value).transpose(1, 2).reshape(batch, tokens, self.inner_dim)
        if self.relative_value is not None:
            relative_value = self.relative_value(tokens)
            relative_outputs = (
                attention.permute(2, 0, 1, 3).reshape(tokens, batch * self.num_heads, -1)
                @ relative_value
            )
            outputs = outputs + relative_outputs.transpose(1, 0).reshape(
                batch, self.num_heads, tokens, -1
            ).transpose(1, 2).reshape(batch, tokens, self.inner_dim)
        return self.projection_dropout(self.projection(outputs))


class AutoFormerMlp(nn.Sequential):
    def __init__(self, embed_dim: int, ratio: float, dropout: float) -> None:
        hidden_dim = int(embed_dim * ratio)
        super().__init__(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )


class AutoFormerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        head_dim: int,
        scale_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path_probability: float,
        relative_position: bool,
        max_relative_position: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attention = AutoFormerAttention(
            embed_dim,
            num_heads,
            head_dim,
            scale_dim,
            attention_dropout,
            dropout,
            relative_position,
            max_relative_position,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = AutoFormerMlp(embed_dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, inputs: Tensor) -> Tensor:
        _, outputs = self.extract_res_features(inputs)
        return outputs

    def extract_res_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        attention_outputs = inputs + self.drop_path(
            self.attention(self.attention_norm(inputs))
        )
        outputs = attention_outputs + self.drop_path(self.mlp(self.mlp_norm(attention_outputs)))
        return attention_outputs, outputs


class StaticAutoFormer(nn.Module):
    """An independently initialized static AutoFormer classification subnet."""

    model_fidelity = "reference_model"

    def __init__(
        self,
        *,
        profile: str,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int,
        depth: int,
        num_heads: Sequence[int],
        mlp_ratio: Sequence[float],
        qkv_head_dim: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        relative_position: bool | None = None,
        max_relative_position: int = 14,
        global_pool: bool = True,
        super_depth: int | None = None,
    ) -> None:
        super().__init__()
        if profile not in AUTOFORMER_PROFILES:
            raise ValueError(
                f"Unknown AutoFormer profile {profile!r}; expected one of {sorted(AUTOFORMER_PROFILES)}"
            )
        heads = tuple(int(value) for value in num_heads)
        ratios = tuple(float(value) for value in mlp_ratio)
        if any(value <= 0 for value in heads):
            raise ValueError("num_heads values must be positive")
        if dropout != 0.0 or attention_dropout != 0.0:
            raise ValueError(f"AutoFormer profile {profile!r} requires zero dropout")
        if not global_pool:
            raise ValueError(f"AutoFormer profile {profile!r} requires patch-token global pooling")
        if profile == VITBENCH_AUTOPROX_PROFILE:
            if qkv_head_dim is not None:
                raise ValueError("ViTBench Auto-Prox fixes QKV width to embed_dim")
            if relative_position not in {None, False}:
                raise ValueError("ViTBench Auto-Prox does not implement relative position")
            if super_depth is not None and super_depth != depth:
                raise ValueError("ViTBench Auto-Prox uses actual-depth stochastic depth")
            if any(embed_dim % value for value in heads):
                raise ValueError("ViTBench Auto-Prox embed_dim must be divisible by every head count")
            resolved_head_dims = tuple(embed_dim // value for value in heads)
            resolved_scale_dims = resolved_head_dims
            resolved_relative_position = False
            resolved_super_depth = depth
            layer_norm_eps = 1e-6
        else:
            if qkv_head_dim not in {None, 64}:
                raise ValueError("AZ-NAS scratch fixes QKV head width to 64")
            if relative_position is False:
                raise ValueError("AZ-NAS scratch profile requires relative position")
            if super_depth is None or super_depth < depth:
                raise ValueError("AZ-NAS scratch requires super_depth >= active depth")
            resolved_head_dims = (64,) * depth
            resolved_scale_dims = tuple(embed_dim // value for value in heads)
            resolved_relative_position = True
            resolved_super_depth = super_depth
            layer_norm_eps = 1e-5
        self._validate_configuration(
            image_size,
            patch_size,
            num_classes,
            embed_dim,
            depth,
            heads,
            ratios,
            min(resolved_head_dims),
            dropout,
            attention_dropout,
            drop_path_rate,
        )
        self.image_size = image_size
        self.profile = profile
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = heads
        self.mlp_ratio = ratios
        self.qkv_head_dim = 64 if profile == AZNAS_SCRATCH_PROFILE else None
        self.relative_position = resolved_relative_position
        self.max_relative_position = max_relative_position
        self.global_pool = global_pool
        self.super_depth = resolved_super_depth
        self.layer_norm_eps = layer_norm_eps
        patch_count = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        self.position_dropout = nn.Dropout(dropout)
        path_rates = torch.linspace(0.0, drop_path_rate, resolved_super_depth).tolist()[:depth]
        self.blocks = nn.ModuleList(
            AutoFormerBlock(
                embed_dim,
                heads[index],
                ratios[index],
                resolved_head_dims[index],
                resolved_scale_dims[index],
                dropout,
                attention_dropout,
                path_rates[index],
                resolved_relative_position,
                max_relative_position,
                layer_norm_eps,
            )
            for index in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.head = nn.Linear(embed_dim, num_classes)
        self._initialize_weights()

    @staticmethod
    def _validate_configuration(
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: tuple[int, ...],
        mlp_ratio: tuple[float, ...],
        qkv_head_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path_rate: float,
    ) -> None:
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if num_classes <= 0 or embed_dim <= 0 or depth <= 0 or qkv_head_dim <= 0:
            raise ValueError("class count, embedding dimensions, and depth must be positive")
        if len(num_heads) != depth or len(mlp_ratio) != depth:
            raise ValueError("num_heads and mlp_ratio must contain one value per layer")
        if any(value <= 0 for value in num_heads):
            raise ValueError("num_heads values must be positive")
        if any(value <= 0 for value in mlp_ratio):
            raise ValueError("mlp_ratio values must be positive")
        for name, value in (
            ("dropout", dropout),
            ("attention_dropout", attention_dropout),
            ("drop_path_rate", drop_path_rate),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def no_weight_decay(self) -> set[str]:
        names = {"class_token", "position_embedding"}
        names.update(
            name
            for name, _parameter in self.named_parameters()
            if ".relative_key." in name or ".relative_value." in name
        )
        return names

    def reference_metadata(self) -> dict[str, Any]:
        if self.profile == VITBENCH_AUTOPROX_PROFILE:
            source = "https://github.com/lliai/Auto-Prox-AAAI24"
            source_commit = "90ed458eff6948a6f0d23e440a8d21bbec50d091"
            cost_protocol = None
        else:
            source = "https://github.com/cvlab-yonsei/AZ-NAS"
            source_commit = "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47"
            cost_protocol = {
                "name": "cream-autoformer-get-complexity",
                "source_commit": "b799630a29995163f282b15e2f38701160272fd1",
                "official_complexity_ops": self.official_complexity_ops(),
                "generic_flops": False,
            }
        return {
            "family": "autoformer",
            "profile": self.profile,
            "model_fidelity": self.model_fidelity,
            "weight_mode": "independent_scratch",
            "supports_inherited_supernet": False,
            "source": source,
            "source_commit": source_commit,
            "cost_protocol": cost_protocol,
            "architecture": {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": list(self.num_heads),
                "mlp_ratio": list(self.mlp_ratio),
                "qkv_head_dim": self.qkv_head_dim,
                "relative_position": self.relative_position,
                "global_pool": self.global_pool,
                "super_depth": self.super_depth,
                "layer_norm_eps": self.layer_norm_eps,
            },
        }

    def official_complexity_ops(self) -> int:
        """Reproduce Cream/AZ-NAS ``get_complexity`` without relabelling it FLOPs."""
        if self.profile != AZNAS_SCRATCH_PROFILE:
            raise NotImplementedError(
                "Cream/AZ-NAS get_complexity is not valid for the ViTBench Auto-Prox profile"
            )
        patch_count = (self.image_size // self.patch_size) ** 2
        total = self.patch_embed.bias.numel()
        total += patch_count * self.patch_embed.weight.numel()
        total += self.position_embedding.numel() / 2.0
        block_sequence_length = patch_count + 2
        for block in self.blocks:
            attention = block.attention
            total += block_sequence_length * self.embed_dim
            total += block_sequence_length * attention.qkv.weight.numel()
            total += 2 * block_sequence_length**2 * attention.inner_dim
            total += block_sequence_length * attention.projection.weight.numel()
            if self.relative_position:
                total += 2 * self.max_relative_position * block_sequence_length**2
                total += block_sequence_length**2 / 2.0
                total += block_sequence_length * attention.inner_dim / 2.0
            total += block_sequence_length * self.embed_dim
            total += block_sequence_length * block.mlp[0].weight.numel()
            total += block_sequence_length * block.mlp[3].weight.numel()
        total += (patch_count + 1) * self.head.weight.numel()
        return int(total)

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("AutoFormer inputs must have shape [batch, 3, height, width]")
        if inputs.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"AutoFormer inputs must be {self.image_size}x{self.image_size}")
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = self.position_dropout(tokens + self.position_embedding)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 1:].mean(1) if self.global_pool else tokens[:, 0]

    def extract_res_features(self, inputs: Tensor) -> list[Tensor]:
        if self.profile != AZNAS_SCRATCH_PROFILE:
            raise NotImplementedError(
                "AZ-NAS residual features are only defined for the AZ-NAS scratch profile"
            )
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("AutoFormer inputs must have shape [batch, 3, height, width]")
        if inputs.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"AutoFormer inputs must be {self.image_size}x{self.image_size}")
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = self.position_dropout(tokens + self.position_embedding)
        residual_features = []
        for block in self.blocks:
            attention_outputs, tokens = block.extract_res_features(tokens)
            residual_features.extend((attention_outputs, tokens))
        return residual_features

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.forward_features(inputs))


__all__ = [
    "AUTOFORMER_PROFILES",
    "AZNAS_SCRATCH_PROFILE",
    "StaticAutoFormer",
    "VITBENCH_AUTOPROX_PROFILE",
]
