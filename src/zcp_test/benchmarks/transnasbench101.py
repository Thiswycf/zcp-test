from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder


TRANSNAS_SPACES = {"micro": "transnas_micro", "macro": "transnas_macro"}


class TransNasBench101Adapter(JsonlBenchmarkAdapter):
    benchmark_id = "transnasbench101"

    def __init__(self, path: str, *, space: str, version: str | None = None, model_builder=None) -> None:
        try:
            self.search_space_id = TRANSNAS_SPACES[space]
        except KeyError as error:
            raise ValueError(f"Unknown TransNAS space {space!r}: {sorted(TRANSNAS_SPACES)}") from error
        self.space = space
        super().__init__(
            path,
            benchmark_id=self.benchmark_id,
            search_space_id=self.search_space_id,
            version=version,
            model_builder=model_builder or default_model_builder,
        )

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if isinstance(specification, str):
            specification = {"architecture": specification}
        if not isinstance(specification, Mapping) or not isinstance(
            specification.get("architecture"), str
        ):
            raise ValueError("TransNAS specification must contain an architecture string")
        return {"architecture": specification["architecture"].strip()}

    def metadata(self) -> Mapping[str, Any]:
        return {**super().metadata(), "space": self.space}
