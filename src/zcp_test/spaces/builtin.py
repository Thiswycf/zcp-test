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
    model_fidelity = "reference_topology"

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
    def __init__(self) -> None:
        super().__init__("nats_size", {f"stage_{index}": [8, 16, 24, 32, 40, 48, 56, 64] for index in range(5)})

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        return TinyConvNet.build([int(value) for value in architecture.spec.values()], num_classes)


class DartsToyLegacySpace(DiscreteSpace):
    def __init__(self) -> None:
        super().__init__("darts_toy_legacy", {"width": [16, 24, 36], "depth": [8, 14, 20], "op": ["sep3", "sep5", "dil3"]})


class Nb101ToyLegacySpace(DiscreteSpace):
    def __init__(self) -> None:
        super().__init__("nb101_toy_legacy", {"width": [16, 32, 64], "depth": [3, 5, 7], "op": ["conv1", "conv3", "maxpool3"]})


class TransNasSpace(DiscreteSpace):
    def __init__(self, variant: str) -> None:
        super().__init__(f"transnas_{variant}", {"width": [16, 32, 64], "depth": [4, 5, 6], "op": ["conv3", "conv5", "skip"]})


class AutoFormerSpace(SearchSpace):
    search_space_id = "autoformer"
    model_family = "transformer"

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
        import torch.nn as nn

        hidden = int(architecture.spec["hidden_dim"])
        depth = int(architecture.spec["depth"])

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch = nn.Conv2d(3, hidden, 16, 16)
                layer = nn.TransformerEncoderLayer(hidden, 3, int(hidden * 4), batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(layer, depth)
                self.norm = nn.LayerNorm(hidden)
                self.head = nn.Linear(hidden, num_classes)

            def forward(self, inputs: Any) -> Any:
                tokens = self.patch(inputs).flatten(2).transpose(1, 2)
                return self.head(self.norm(self.blocks(tokens).mean(1)))

        return Model()


class MobileSpace(SearchSpace):
    model_family = "cnn"

    def __init__(self, variant: str) -> None:
        self.search_space_id = variant

    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture:
        canonical = {"kernel_size": list(specification["kernel_size"]), "expand_ratio": list(specification["expand_ratio"]), "depth": list(specification["depth"]), "width_mult": float(specification["width_mult"]), "resolution": int(specification["resolution"])}
        if not set(canonical["kernel_size"]) <= {3, 5, 7} or not set(canonical["expand_ratio"]) <= {3, 4, 6} or not set(canonical["depth"]) <= {2, 3, 4}:
            raise ValueError("Invalid OFA choices")
        return Architecture(self.search_space_id, _stable_id(self.search_space_id, canonical), canonical)

    def sample(self, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        depth = [rng.choice([2, 3, 4]) for _ in range(5)]
        blocks = sum(depth)
        return self.canonicalize({"kernel_size": [rng.choice([3, 5, 7]) for _ in range(blocks)], "expand_ratio": [rng.choice([3, 4, 6]) for _ in range(blocks)], "depth": depth, "width_mult": rng.choice([1.0, 1.2]), "resolution": rng.choice([192, 208, 224])})

    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture:
        rng = random.Random(seed)
        specification = {key: list(value) if isinstance(value, list) else value for key, value in architecture.spec.items()}
        index = rng.randrange(len(specification["kernel_size"]))
        specification["kernel_size"][index] = rng.choice([3, 5, 7])
        specification["expand_ratio"][index] = rng.choice([3, 4, 6])
        return self.canonicalize(specification)

    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture:
        return self.mutate(random.Random(seed).choice([left, right]), seed)

    def build_model(self, architecture: Architecture, num_classes: int) -> Any:
        import torch.nn as nn

        width = int(24 * float(architecture.spec["width_mult"]))
        layers: list[Any] = [_ConvBnAct.build(3, width, 3, 2)]
        current = width
        block = 0
        for stage, depth in enumerate(architecture.spec["depth"]):
            output = int(width * (1 + stage))
            for index in range(depth):
                kernel = architecture.spec["kernel_size"][block]
                expand = architecture.spec["expand_ratio"][block]
                hidden = current * expand
                layers.extend([_ConvBnAct.build(current, hidden, 1), _ConvBnAct.build(hidden, hidden, kernel, 2 if index == 0 and stage else 1, hidden), nn.Conv2d(hidden, output, 1, bias=False), nn.BatchNorm2d(output)])
                current = output
                block += 1
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(current, num_classes)])
        return nn.Sequential(*layers)


SPACES.register("nb201_topology", Nb201TopologySpace)
SPACES.register("nats_size", NatsSizeSpace)
SPACES.register("nb101_dag", Nb101Space)
SPACES.register("nb101_toy_legacy", Nb101ToyLegacySpace)
SPACES.register("darts", lambda: DartsSpace("zcp"))
SPACES.register("darts_toy_legacy", DartsToyLegacySpace)
SPACES.register("transnas_micro", lambda: TransNasSpace("micro"))
SPACES.register("transnas_macro", lambda: TransNasSpace("macro"))
SPACES.register("autoformer", AutoFormerSpace)
SPACES.register("pit", lambda: DiscreteSpace("pit", {"width": [16, 24, 32, 40], "depth": [3, 4, 6, 8]}, "transformer"))
SPACES.register("ofa_proxyless_mbv2", lambda: MobileSpace("ofa_proxyless_mbv2"))
SPACES.register("ofa_mbv3", lambda: MobileSpace("ofa_mbv3"))
