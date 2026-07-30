from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping

from zcp_test.spaces import SPACES
from zcp_test.spaces.base import SearchSpace
from zcp_test.spaces.darts import DartsSpace
from zcp_test.spaces.nb101 import Nb101Space
from zcp_test.types import Architecture


def _stable_id(space: str, specification: Mapping[str, Any]) -> str:
    payload = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{space}:{payload}".encode()).hexdigest()[:20]


class _ConvBnAct:
    @staticmethod
    def build(in_channels: int, out_channels: int, kernel: int, stride: int = 1, groups: int = 1) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class TinyConvNet:
    @staticmethod
    def build(channels: list[int], num_classes: int) -> Any:
        import torch.nn as nn

        layers = []
        current = 3
        for index, channel in enumerate(channels):
            layers.append(_ConvBnAct.build(current, channel, 3, 2 if index in (1, 3) else 1))
            current = channel
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(current, num_classes))


@dataclass
class DiscreteSpace(SearchSpace):
    search_space_id: str
    fields: Mapping[str, list[Any]]
    model_family: str = "cnn"
    model_fidelity = "proxy_approximation"

    def _rng(self, seed: int | None) -> random.Random:
        return random.Random(seed)

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        canonical: dict[str, Any] = {}
        for field, choices in self.fields.items():
            value = specification[field]
            if value not in choices:
                raise ValueError(f"Invalid {field}={value!r}; choices={choices}")
            canonical[field] = value
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = self._rng(seed)
        return self.canonicalize({field: rng.choice(choices) for field, choices in self.fields.items()})

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = self._rng(seed)
        specification = dict(architecture.spec)
        field = rng.choice(list(self.fields))
        alternatives = [value for value in self.fields[field] if value != specification[field]]
        specification[field] = rng.choice(alternatives)
        return self.canonicalize(specification)

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = self._rng(seed)
        return self.canonicalize({field: rng.choice([left.spec[field], right.spec[field]]) for field in self.fields})

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        width = int(architecture.spec.get("width", architecture.spec.get("hidden_dim", 32)))
        depth = int(architecture.spec.get("depth", 4))
        return TinyConvNet.build([width] * max(2, min(depth, 8)), num_classes)


class Nb201TopologySpace(DiscreteSpace):
    model_fidelity = "reference_topology_pytorch_port"
    implementation_source = "https://github.com/D-X-Y/NAS-Bench-201"

    def __init__(self) -> None:
        operations = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
        super().__init__("nb201_topology", {f"edge_{index}": operations for index in range(6)})

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.benchmarks.model_builders import nb201_model

        edges = [str(architecture.spec[f"edge_{index}"]) for index in range(6)]
        specification = (
            f"|{edges[0]}~0|+"
            f"|{edges[1]}~0|{edges[2]}~1|+"
            f"|{edges[3]}~0|{edges[4]}~1|{edges[5]}~2|"
        )
        model_architecture = Architecture(
            "nb201_topology",
            architecture.architecture_id,
            {"architecture": specification},
        )
        dataset = {10: "cifar10", 100: "cifar100", 120: "ImageNet16-120"}.get(
            num_classes, "cifar10"
        )
        return nb201_model(model_architecture, dataset)


class NatsSizeSpace(DiscreteSpace):
    model_fidelity = "reference_topology_pytorch_port"
    implementation_source = "https://github.com/D-X-Y/NATS-Bench"

    def __init__(self) -> None:
        super().__init__("nats_size", {f"stage_{index}": [8, 16, 24, 32, 40, 48, 56, 64] for index in range(5)})

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.nb201 import build_nats_sss

        return build_nats_sss(
            [int(architecture.spec[f"stage_{index}"]) for index in range(5)],
            num_classes,
        )


class DartsToyLegacySpace(DiscreteSpace):
    def __init__(self) -> None:
        super().__init__("darts_toy_legacy", {"width": [16, 24, 36], "depth": [8, 14, 20], "op": ["sep3", "sep5", "dil3"]})


class Nb101ToyLegacySpace(DiscreteSpace):
    def __init__(self) -> None:
        super().__init__("nb101_toy_legacy", {"width": [16, 32, 64], "depth": [3, 5, 7], "op": ["conv1", "conv3", "maxpool3"]})


class TransNasSpace(SearchSpace):
    model_family = "cnn"
    model_fidelity = "reference_topology_pytorch_port"
    implementation_source = "https://github.com/yawen-d/TransNASBench"

    def __init__(self, variant: str) -> None:
        if variant not in {"micro", "macro"}:
            raise ValueError(f"Unknown TransNAS variant: {variant!r}")
        self.variant = variant
        self.search_space_id = f"transnas_{variant}"

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        from zcp_test.models.transnas import macro_codes, parse_code

        code = str(specification["architecture"])
        _, macro, micro = parse_code(code)
        if self.variant == "micro" and (macro != "41414" or micro is None):
            raise ValueError("TransNAS micro requires macro code 41414 and a micro cell")
        if self.variant == "macro" and (micro is not None or macro not in macro_codes()):
            raise ValueError("TransNAS macro requires an official basic-block macro code")
        canonical = {"architecture": code}
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        from zcp_test.models.transnas import macro_codes

        rng = random.Random(seed)
        if self.variant == "micro":
            operations = "".join(rng.choice("0123") for _ in range(6))
            code = f"64-41414-{operations[0]}_{operations[1:3]}_{operations[3:]}"
        else:
            code = f"64-{rng.choice(macro_codes())}-basic"
        return self.canonicalize({"architecture": code})

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        from zcp_test.models.transnas import macro_codes, parse_code

        rng = random.Random(seed)
        _, macro, micro = parse_code(str(architecture.spec["architecture"]))
        if self.variant == "micro" and micro is not None:
            operations = list("".join(micro))
            index = rng.randrange(len(operations))
            operations[index] = rng.choice([value for value in "0123" if value != operations[index]])
            code = f"64-41414-{operations[0]}_{''.join(operations[1:3])}_{''.join(operations[3:])}"
        else:
            candidates = [value for value in macro_codes() if value != macro]
            code = f"64-{rng.choice(candidates)}-basic"
        return self.canonicalize({"architecture": code})

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        return self.canonicalize(dict(rng.choice([left, right]).spec))

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.transnas import TransNasNetwork

        return TransNasNetwork(str(architecture.spec["architecture"]), num_classes)


class AutoFormerSpace(SearchSpace):
    search_space_id = "autoformer"
    model_family = "transformer"
    model_fidelity = "reference_model"
    implementation_source = "https://github.com/cvlab-yonsei/AZ-NAS/tree/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/ImageNet_AutoFormer"
    implementation_commit = "5e6683a2cfa5c6d0dc34a1317a842497ba7eae47"

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        hidden_dim, depth = int(specification["hidden_dim"]), int(specification["depth"])
        heads, ratios = list(specification["num_heads"]), list(specification["mlp_ratio"])
        if hidden_dim not in (192, 216, 240) or depth not in (12, 13, 14):
            raise ValueError("Invalid AutoFormer hidden_dim/depth")
        if len(heads) != depth or len(ratios) != depth or not set(heads) <= {3, 4} or not set(ratios) <= {3.5, 4.0}:
            raise ValueError("Invalid per-layer AutoFormer configuration")
        canonical = {"hidden_dim": hidden_dim, "depth": depth, "num_heads": heads, "mlp_ratio": ratios}
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        depth = rng.choice([12, 13, 14])
        return self.canonicalize({"hidden_dim": rng.choice([192, 216, 240]), "depth": depth, "num_heads": [rng.choice([3, 4]) for _ in range(depth)], "mlp_ratio": [rng.choice([3.5, 4.0]) for _ in range(depth)]})

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        specification = dict(architecture.spec)
        specification["num_heads"] = list(specification["num_heads"])
        specification["mlp_ratio"] = list(specification["mlp_ratio"])
        index = rng.randrange(int(specification["depth"]))
        if rng.random() < 0.5:
            specification["num_heads"][index] = 7 - specification["num_heads"][index]
        else:
            specification["mlp_ratio"][index] = 7.5 - specification["mlp_ratio"][index]
        return self.canonicalize(specification)

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        depth = min(int(left.spec["depth"]), int(right.spec["depth"]))
        return self.canonicalize({"hidden_dim": rng.choice([left.spec["hidden_dim"], right.spec["hidden_dim"]]), "depth": depth, "num_heads": [rng.choice([left.spec["num_heads"][i], right.spec["num_heads"][i]]) for i in range(depth)], "mlp_ratio": [rng.choice([left.spec["mlp_ratio"][i], right.spec["mlp_ratio"][i]]) for i in range(depth)]})

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer

        return StaticAutoFormer(
            profile=AZNAS_SCRATCH_PROFILE,
            num_classes=num_classes,
            embed_dim=int(architecture.spec["hidden_dim"]),
            depth=int(architecture.spec["depth"]),
            num_heads=architecture.spec["num_heads"],
            mlp_ratio=architecture.spec["mlp_ratio"],
            global_pool=True,
            super_depth=14,
        )

    def build_training_model(
        self, architecture: Architecture, num_classes: int, config: Mapping[str, Any]
    ) -> Any:
        from zcp_test.models.autoformer import AZNAS_SCRATCH_PROFILE, StaticAutoFormer

        if not bool(config.get("change_qkv", True)):
            raise ValueError("AZ-NAS scratch profile requires change_qkv=true")
        return StaticAutoFormer(
            profile=AZNAS_SCRATCH_PROFILE,
            image_size=int(config.get("input_size", 224)),
            patch_size=int(config.get("patch_size", 16)),
            num_classes=num_classes,
            embed_dim=int(architecture.spec["hidden_dim"]),
            depth=int(architecture.spec["depth"]),
            num_heads=architecture.spec["num_heads"],
            mlp_ratio=architecture.spec["mlp_ratio"],
            qkv_head_dim=int(config.get("qkv_head_dim", 64)),
            dropout=float(config.get("dropout", 0.0)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
            drop_path_rate=float(config.get("drop_path_prob", 0.0)),
            relative_position=bool(config.get("relative_position", True)),
            max_relative_position=int(config.get("max_relative_position", 14)),
            global_pool=bool(config.get("global_pool", True)),
            super_depth=14,
        )


class PitSpace(SearchSpace):
    search_space_id = "pit"
    model_family = "transformer"
    model_fidelity = "reference_topology_pytorch_port"
    implementation_source = "https://github.com/lliai/Auto-Prox-AAAI24"
    implementation_commit = "90ed458"

    _depth_choices = ({1, 2, 3}, {4, 6, 8}, {2, 4, 6})
    _head_choices = {2, 4, 8}
    _head_patterns = tuple(
        (first, second, third)
        for first in (2, 4, 8)
        for second in (2, 4, 8)
        for third in (2, 4, 8)
        if first <= second <= third
    )
    _base_choices = {16, 24, 32, 40}
    _ratio_choices = {2.0, 4.0, 6.0, 8.0}

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        base_dim = int(specification["base_dim"])
        depths = [int(value) for value in specification["depth"]]
        heads = [int(value) for value in specification["num_heads"]]
        ratio = float(specification["mlp_ratio"])
        if base_dim not in self._base_choices:
            raise ValueError("Invalid PiT base_dim")
        if len(depths) != 3 or any(
            value not in choices for value, choices in zip(depths, self._depth_choices, strict=True)
        ):
            raise ValueError("Invalid PiT stage depths")
        if tuple(heads) not in self._head_patterns:
            raise ValueError("Invalid PiT stage head counts")
        if ratio not in self._ratio_choices:
            raise ValueError("Invalid PiT mlp_ratio")
        canonical = {
            "base_dim": base_dim,
            "depth": depths,
            "num_heads": heads,
            "mlp_ratio": int(ratio),
        }
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        return self.canonicalize(
            {
                "base_dim": rng.choice(sorted(self._base_choices)),
                "depth": [rng.choice(sorted(choices)) for choices in self._depth_choices],
                "num_heads": list(rng.choice(self._head_patterns)),
                "mlp_ratio": rng.choice(sorted(self._ratio_choices)),
            }
        )

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        specification = {
            **architecture.spec,
            "depth": list(architecture.spec["depth"]),
            "num_heads": list(architecture.spec["num_heads"]),
        }
        field = rng.choice(["base_dim", "depth", "num_heads", "mlp_ratio"])
        if field == "base_dim":
            choices = self._base_choices
        elif field == "mlp_ratio":
            choices = self._ratio_choices
        elif field == "num_heads":
            current = tuple(specification["num_heads"])
            specification["num_heads"] = list(
                rng.choice([value for value in self._head_patterns if value != current])
            )
            return self.canonicalize(specification)
        else:
            index = rng.randrange(3)
            choices = self._depth_choices[index]
            current = specification[field][index]
            specification[field][index] = rng.choice(sorted(choices - {current}))
            return self.canonicalize(specification)
        specification[field] = rng.choice(sorted(choices - {specification[field]}))
        return self.canonicalize(specification)

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        return self.canonicalize(
            {
                "base_dim": rng.choice([left.spec["base_dim"], right.spec["base_dim"]]),
                "depth": [
                    rng.choice([left.spec["depth"][index], right.spec["depth"][index]])
                    for index in range(3)
                ],
                "num_heads": list(rng.choice([left.spec["num_heads"], right.spec["num_heads"]])),
                "mlp_ratio": rng.choice([left.spec["mlp_ratio"], right.spec["mlp_ratio"]]),
            }
        )

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.pit import StaticPiT

        return StaticPiT(
            num_classes=num_classes,
            base_dim=int(architecture.spec["base_dim"]),
            depth=architecture.spec["depth"],
            num_heads=architecture.spec["num_heads"],
            mlp_ratio=float(architecture.spec["mlp_ratio"]),
        )


class MobileSpace(SearchSpace):
    model_family = "cnn"

    def __init__(self, variant: str) -> None:
        if variant not in {"zennas_plainnet_mbv2", "ofa_proxyless_mbv2"}:
            raise ValueError(f"Unknown mobile search space: {variant!r}")
        self.search_space_id = variant
        self.model_fidelity = (
            "reference_model"
            if variant == "ofa_proxyless_mbv2"
            else "proxy_approximation"
        )
        self.implementation_source = {
            "zennas_plainnet_mbv2": "https://github.com/idstcv/ZenNAS",
            "ofa_proxyless_mbv2": "https://github.com/mit-han-lab/once-for-all",
        }[variant]
        self.implementation_commit = (
            "f03b2673db313b9167e2a1c2b7a5cad540cc1313"
            if variant == "ofa_proxyless_mbv2"
            else "d1d617e0352733d39890fb64ea758f9c85b28c1a"
        )

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        canonical = {
            "kernel_size": [int(value) for value in specification["kernel_size"]],
            "expand_ratio": [int(value) for value in specification["expand_ratio"]],
            "depth": [int(value) for value in specification["depth"]],
            "width_mult": float(specification["width_mult"]),
            "resolution": int(specification["resolution"]),
        }
        if not set(canonical["kernel_size"]) <= {3, 5, 7} or not set(canonical["expand_ratio"]) <= {3, 4, 6} or not set(canonical["depth"]) <= {2, 3, 4}:
            raise ValueError("Invalid OFA choices")
        if len(canonical["depth"]) != 5:
            raise ValueError("MobileNetV2 requires five searchable stages")
        if self.search_space_id == "ofa_proxyless_mbv2":
            if len(canonical["kernel_size"]) != 21 or len(canonical["expand_ratio"]) != 21:
                raise ValueError(
                    "OFA Proxyless requires 21 positional kernel and expansion values"
                )
            if canonical["width_mult"] != 1.3:
                raise ValueError("official OFA Proxyless supernet uses width multiplier 1.3")
            if not 128 <= canonical["resolution"] <= 224 or canonical["resolution"] % 4:
                raise ValueError("OFA Proxyless resolution must be 128..224 with step 4")
        else:
            if len(canonical["kernel_size"]) != sum(canonical["depth"]) or len(
                canonical["expand_ratio"]
            ) != sum(canonical["depth"]):
                raise ValueError("kernel_size and expand_ratio must match active block count")
            if canonical["width_mult"] not in {1.0, 1.2} or canonical["resolution"] not in {
                192,
                208,
                224,
            }:
                raise ValueError("Invalid PlainNet MobileNetV2 width multiplier or resolution")
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        depth = [rng.choice([2, 3, 4]) for _ in range(5)]
        blocks = 21 if self.search_space_id == "ofa_proxyless_mbv2" else sum(depth)
        return self.canonicalize(
            {
                "kernel_size": [rng.choice([3, 5, 7]) for _ in range(blocks)],
                "expand_ratio": [rng.choice([3, 4, 6]) for _ in range(blocks)],
                "depth": depth,
                "width_mult": 1.3
                if self.search_space_id == "ofa_proxyless_mbv2"
                else rng.choice([1.0, 1.2]),
                "resolution": rng.choice(range(128, 225, 4))
                if self.search_space_id == "ofa_proxyless_mbv2"
                else rng.choice([192, 208, 224]),
            }
        )

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        specification = {key: list(value) if isinstance(value, list) else value for key, value in architecture.spec.items()}
        field = rng.choice(["kernel_size", "expand_ratio", "depth", "resolution"])
        choices = {
            "kernel_size": [3, 5, 7],
            "expand_ratio": [3, 4, 6],
            "depth": [2, 3, 4],
            "resolution": list(range(128, 225, 4))
            if self.search_space_id == "ofa_proxyless_mbv2"
            else [192, 208, 224],
        }
        if field in {"kernel_size", "expand_ratio", "depth"}:
            index = rng.randrange(len(specification[field]))
            specification[field][index] = rng.choice(choices[field])
        else:
            specification[field] = rng.choice(choices[field])
        return self.canonicalize(specification)

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        if self.search_space_id == "zennas_plainnet_mbv2":
            return self.mutate(rng.choice((left, right)), seed)
        specification: dict[str, Any] = {}
        for field in ("kernel_size", "expand_ratio", "depth"):
            specification[field] = [
                rng.choice((left.spec[field][index], right.spec[field][index]))
                for index in range(len(left.spec[field]))
            ]
        specification["width_mult"] = rng.choice(
            (left.spec["width_mult"], right.spec["width_mult"])
        )
        specification["resolution"] = rng.choice(
            (left.spec["resolution"], right.spec["resolution"])
        )
        return self.canonicalize(specification)

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.mobile import OFAProxylessMobileNetV2, PlainNetMobileNetV2

        if self.search_space_id == "ofa_proxyless_mbv2":
            return OFAProxylessMobileNetV2(
                num_classes=num_classes,
                width_mult=float(architecture.spec["width_mult"]),
                stage_depths=architecture.spec["depth"],
                kernel_sizes=architecture.spec["kernel_size"],
                expand_ratios=architecture.spec["expand_ratio"],
                image_size=int(architecture.spec["resolution"]),
            )

        width = float(architecture.spec["width_mult"])
        stage_channels = [max(8, int(round(value * width / 8) * 8)) for value in (24, 40, 80, 96, 192)]
        arguments = {
            "num_classes": num_classes,
            "stem_channels": max(8, int(round(32 * width / 8) * 8)),
            "head_channels": max(8, int(round(1280 * width / 8) * 8)),
            "stage_channels": stage_channels,
            "stage_depths": architecture.spec["depth"],
            "stage_strides": [2, 2, 2, 1, 2],
            "kernel_sizes": architecture.spec["kernel_size"],
            "expand_ratios": architecture.spec["expand_ratio"],
        }
        return PlainNetMobileNetV2(**arguments)


class OfaMobileNetV3Space(SearchSpace):
    search_space_id = "ofa_mbv3"
    model_family = "cnn"
    model_fidelity = "reference_model"
    implementation_source = "https://github.com/mit-han-lab/once-for-all"
    implementation_commit = "f03b2673db313b9167e2a1c2b7a5cad540cc1313"

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        canonical = {
            "kernel_size": [int(value) for value in specification["kernel_size"]],
            "expand_ratio": [int(value) for value in specification["expand_ratio"]],
            "depth": [int(value) for value in specification["depth"]],
            "width_mult": float(specification["width_mult"]),
            "resolution": int(specification["resolution"]),
        }
        if len(canonical["kernel_size"]) != 20 or not set(canonical["kernel_size"]) <= {3, 5, 7}:
            raise ValueError("OFA-MBV3 kernel_size requires 20 values from {3, 5, 7}")
        if len(canonical["expand_ratio"]) != 20 or not set(canonical["expand_ratio"]) <= {3, 4, 6}:
            raise ValueError("OFA-MBV3 expand_ratio requires 20 values from {3, 4, 6}")
        if len(canonical["depth"]) != 5 or not set(canonical["depth"]) <= {2, 3, 4}:
            raise ValueError("OFA-MBV3 depth requires five values from {2, 3, 4}")
        if canonical["width_mult"] not in {1.0, 1.2} or canonical["resolution"] not in {192, 208, 224}:
            raise ValueError("Invalid OFA-MBV3 width multiplier or resolution")
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        return self.canonicalize(
            {
                "kernel_size": [rng.choice([3, 5, 7]) for _ in range(20)],
                "expand_ratio": [rng.choice([3, 4, 6]) for _ in range(20)],
                "depth": [rng.choice([2, 3, 4]) for _ in range(5)],
                "width_mult": rng.choice([1.0, 1.2]),
                "resolution": rng.choice([192, 208, 224]),
            }
        )

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        specification = {
            key: list(value) if isinstance(value, list) else value
            for key, value in architecture.spec.items()
        }
        field = rng.choice(["kernel_size", "expand_ratio", "depth", "width_mult", "resolution"])
        choices = {
            "kernel_size": [3, 5, 7],
            "expand_ratio": [3, 4, 6],
            "depth": [2, 3, 4],
            "width_mult": [1.0, 1.2],
            "resolution": [192, 208, 224],
        }[field]
        if isinstance(specification[field], list):
            index = rng.randrange(len(specification[field]))
            current = specification[field][index]
            specification[field][index] = rng.choice([value for value in choices if value != current])
        else:
            specification[field] = rng.choice([value for value in choices if value != specification[field]])
        return self.canonicalize(specification)

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        return self.canonicalize(
            {
                key: (
                    [rng.choice([left.spec[key][index], right.spec[key][index]]) for index in range(len(left.spec[key]))]
                    if isinstance(left.spec[key], list)
                    else rng.choice([left.spec[key], right.spec[key]])
                )
                for key in left.spec
            }
        )

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        from zcp_test.models.mobile import StaticMobileNetV3

        return StaticMobileNetV3(
            num_classes=num_classes,
            width_mult=float(architecture.spec["width_mult"]),
            stage_depths=architecture.spec["depth"],
            kernel_sizes=architecture.spec["kernel_size"],
            expand_ratios=architecture.spec["expand_ratio"],
        )


SPACES.register("nb201_topology", Nb201TopologySpace)
SPACES.register("nats_size", NatsSizeSpace)
SPACES.register("nb101_dag", Nb101Space)
SPACES.register("nb101_toy_legacy", Nb101ToyLegacySpace)
SPACES.register("darts", lambda: DartsSpace("zcp"))
SPACES.register("darts_toy_legacy", DartsToyLegacySpace)
SPACES.register("transnas_micro", lambda: TransNasSpace("micro"))
SPACES.register("transnas_macro", lambda: TransNasSpace("macro"))
SPACES.register("autoformer", AutoFormerSpace)
SPACES.register("pit", PitSpace)
SPACES.register("ofa_proxyless_mbv2", lambda: MobileSpace("ofa_proxyless_mbv2"))
SPACES.register("zennas_plainnet_mbv2", lambda: MobileSpace("zennas_plainnet_mbv2"))
SPACES.register("ofa_mbv3", OfaMobileNetV3Space)
