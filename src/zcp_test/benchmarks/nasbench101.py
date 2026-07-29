from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder


class NasBench101Adapter(JsonlBenchmarkAdapter):
    benchmark_id = "nasbench101"
    search_space_id = "nb101_dag"

    def __init__(self, path: str, *, version: str | None = None, model_builder=None) -> None:
        super().__init__(
            path,
            benchmark_id=self.benchmark_id,
            search_space_id=self.search_space_id,
            version=version,
            model_builder=model_builder or default_model_builder,
        )

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if not isinstance(specification, Mapping):
            raise TypeError("NAS-Bench-101 specification must be a mapping")
        matrix = specification.get("matrix")
        operations = specification.get("operations")
        if not isinstance(matrix, list) or not matrix or any(not isinstance(row, list) for row in matrix):
            raise ValueError("NAS-Bench-101 matrix must be a non-empty list of rows")
        if any(len(row) != len(matrix) for row in matrix):
            raise ValueError("NAS-Bench-101 matrix must be square")
        if not isinstance(operations, list) or len(operations) != len(matrix):
            raise ValueError("NAS-Bench-101 operations must match matrix dimensions")
        if operations[0] != "input" or operations[-1] != "output":
            raise ValueError("NAS-Bench-101 operations must start with input and end with output")
        allowed = {"input", "output", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"}
        if not set(operations).issubset(allowed):
            raise ValueError("NAS-Bench-101 specification contains an unsupported operation")
        return {"matrix": [[int(value) for value in row] for row in matrix], "operations": operations}
