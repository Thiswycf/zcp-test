from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder


VIT_SLICES = {
    "autoformer_main": ("autoformer", "auto-prox-90ed458-autoformer-main"),
    "autoformer_ext": ("autoformer", "auto-prox-90ed458-autoformer-ext"),
    "pit": ("pit", "auto-prox-90ed458-pit"),
}

VIT_SOURCE_SHA256 = {
    "autoformer_main": "712ad277546d9f7f565ce07885be7e0b98dcd8d0724fdd1120f595b517436eca",
    "autoformer_ext": "05f5df6a41f338fb5f47eafebfc8758c75e451606856b278ccda1c60b26e7bca",
    "pit": "bdda89841d4105f99ab759e3243e7a2402929ba7a8430dac12a50256aa533bb2",
}

_AUTOFORMER_DEPTHS = {12, 13, 14}
_AUTOFORMER_HIDDEN_DIMS = {192, 216, 240}
_AUTOFORMER_HEADS = {3, 4}
_AUTOFORMER_MLP_RATIOS = {3.5, 4.0}
_PIT_DEPTHS = ({1, 2, 3}, {4, 6, 8}, {2, 4, 6})
_PIT_BASE_DIMS = {16, 24, 32, 40}
_PIT_HEADS = {2, 4, 8}
_PIT_MLP_RATIOS = {2.0, 4.0, 6.0, 8.0}


def _canonical_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"ViT {field} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ViT {field} must be an integer") from error
    if isinstance(value, Real) and float(value) != normalized:
        raise ValueError(f"ViT {field} must be an integer")
    if isinstance(value, str) and value.strip() != str(normalized):
        raise ValueError(f"ViT {field} must use canonical integer syntax")
    return normalized


def _canonical_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"ViT {field} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ViT {field} must be numeric") from error
    if not normalized.is_integer() and normalized not in _AUTOFORMER_MLP_RATIOS:
        raise ValueError(f"ViT {field} has unsupported precision")
    return normalized


def _canonical_sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"ViT {field} must be a sequence")
    return list(value)


class VitBench101Adapter(JsonlBenchmarkAdapter):
    benchmark_id = "vitbench101"

    def __init__(self, path: str, *, slice_id: str, model_builder=None) -> None:
        try:
            search_space_id, protocol = VIT_SLICES[slice_id]
        except KeyError as error:
            raise ValueError(f"Unknown ViT-Bench-101 slice {slice_id!r}: {sorted(VIT_SLICES)}") from error
        self.slice_id = slice_id
        self.search_space_id = search_space_id
        super().__init__(
            path,
            benchmark_id=self.benchmark_id,
            search_space_id=search_space_id,
            version="auto-prox-90ed458",
            model_builder=model_builder or default_model_builder,
            required_protocol=protocol,
        )
        source_hashes = {record.get("source_sha256") for record in self._records}
        if len(source_hashes) != 1 or None in source_hashes:
            raise ValueError("ViT-Bench-101 records must contain one source_sha256")
        self.protocol = protocol
        self.source_sha256 = str(next(iter(source_hashes)))
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("ViT-Bench-101 source_sha256 must be a lowercase SHA-256 digest")
        if self.source_sha256 != VIT_SOURCE_SHA256[slice_id]:
            raise ValueError(
                f"ViT-Bench-101 {slice_id} source_sha256 does not match the pinned release asset"
            )

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if not isinstance(specification, Mapping):
            raise TypeError("ViT architecture must be a mapping")
        allowed = (
            {"depth", "hidden_dim", "num_heads", "mlp_ratio"}
            if self.search_space_id == "autoformer"
            else {"depth", "base_dim", "num_heads", "mlp_ratio"}
        )
        fields = set(specification)
        if fields != allowed:
            missing = sorted(allowed - fields)
            extra = sorted(fields - allowed)
            raise ValueError(f"ViT architecture fields mismatch: missing={missing}, extra={extra}")
        if self.search_space_id == "autoformer":
            depth = _canonical_int(specification["depth"], "depth")
            hidden_dim = _canonical_int(specification["hidden_dim"], "hidden_dim")
            heads = [
                _canonical_int(value, "num_heads")
                for value in _canonical_sequence(specification["num_heads"], "num_heads")
            ]
            ratios = [
                _canonical_float(value, "mlp_ratio")
                for value in _canonical_sequence(specification["mlp_ratio"], "mlp_ratio")
            ]
            if depth not in _AUTOFORMER_DEPTHS:
                raise ValueError(f"Unsupported AutoFormer depth: {depth}")
            if hidden_dim not in _AUTOFORMER_HIDDEN_DIMS:
                raise ValueError(f"Unsupported AutoFormer hidden_dim: {hidden_dim}")
            if len(heads) != depth or len(ratios) != depth:
                raise ValueError("AutoFormer num_heads/mlp_ratio lengths must equal depth")
            if not set(heads).issubset(_AUTOFORMER_HEADS):
                raise ValueError("Unsupported AutoFormer num_heads value")
            if not set(ratios).issubset(_AUTOFORMER_MLP_RATIOS):
                raise ValueError("Unsupported AutoFormer mlp_ratio value")
            return {
                "depth": depth,
                "hidden_dim": hidden_dim,
                "num_heads": heads,
                "mlp_ratio": ratios,
            }
        depths = [
            _canonical_int(value, "depth")
            for value in _canonical_sequence(specification["depth"], "depth")
        ]
        base_dim = _canonical_int(specification["base_dim"], "base_dim")
        heads = [
            _canonical_int(value, "num_heads")
            for value in _canonical_sequence(specification["num_heads"], "num_heads")
        ]
        mlp_ratio = int(_canonical_float(specification["mlp_ratio"], "mlp_ratio"))
        if len(depths) != 3 or len(heads) != 3:
            raise ValueError("PiT depth and num_heads must contain three stages")
        if any(
            value not in choices
            for value, choices in zip(depths, _PIT_DEPTHS, strict=True)
        ):
            raise ValueError("Unsupported PiT stage depth value")
        if base_dim not in _PIT_BASE_DIMS:
            raise ValueError(f"Unsupported PiT base_dim: {base_dim}")
        if not set(heads).issubset(_PIT_HEADS) or heads != sorted(heads):
            raise ValueError("Unsupported PiT stage num_heads pattern")
        if mlp_ratio not in _PIT_MLP_RATIOS:
            raise ValueError(f"Unsupported PiT mlp_ratio: {mlp_ratio}")
        return {
            "depth": depths,
            "base_dim": base_dim,
            "num_heads": heads,
            "mlp_ratio": mlp_ratio,
        }

    def metadata(self) -> Mapping[str, Any]:
        return {
            **super().metadata(),
            "slice_id": self.slice_id,
            "protocol": self.protocol,
            "source_commit": "90ed458",
            "source_sha256": self.source_sha256,
        }

    def query_provenance(self) -> Mapping[str, str]:
        return {
            "slice_id": self.slice_id,
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
        }
