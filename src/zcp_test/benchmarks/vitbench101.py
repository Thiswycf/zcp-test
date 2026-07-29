from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder


VIT_SLICES = {
    "autoformer_main": ("autoformer", "auto-prox-90ed458-autoformer-main"),
    "autoformer_ext": ("autoformer", "auto-prox-90ed458-autoformer-ext"),
    "pit": ("pit", "auto-prox-90ed458-pit"),
}


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
            model_builder=model_builder or default_model_builder,
            required_protocol=protocol,
        )

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if not isinstance(specification, Mapping):
            raise TypeError("ViT architecture must be a mapping")
        required = (
            {"depth", "hidden_dim", "num_heads", "mlp_ratio"}
            if self.search_space_id == "autoformer"
            else {"depth", "base_dim", "num_heads", "mlp_ratio"}
        )
        if not required.issubset(specification):
            raise ValueError(f"ViT architecture is missing: {sorted(required - set(specification))}")
        result = dict(specification)
        depth = result["depth"]
        if isinstance(depth, list):
            if not depth or any(int(value) <= 0 for value in depth):
                raise ValueError("ViT stage depths must be positive")
        elif int(depth) <= 0:
            raise ValueError("ViT depth must be positive")
        dimension_key = "hidden_dim" if self.search_space_id == "autoformer" else "base_dim"
        if int(result[dimension_key]) <= 0:
            raise ValueError(f"ViT {dimension_key} must be positive")
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {**super().metadata(), "slice_id": self.slice_id, "source_commit": "90ed458"}
