from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from typing import Any

from zcp_test.benchmarks.jsonl_adapter import JsonlBenchmarkAdapter
from zcp_test.benchmarks.model_builders import model_builder as default_model_builder
from zcp_test.types import Architecture


TRANSNAS_SPACES = {"micro": "transnas_micro", "macro": "transnas_macro"}


class _ArchitectureIterator(Iterator[Architecture]):
    def __init__(
        self,
        architectures: Iterator[Architecture],
        sampling_provenance: Mapping[str, str | None],
    ) -> None:
        self._architectures = architectures
        self.sampling_provenance = dict(sampling_provenance)

    def __next__(self) -> Architecture:
        return next(self._architectures)


class TransNasBench101Adapter(JsonlBenchmarkAdapter):
    benchmark_id = "transnasbench101"

    def __init__(self, path: str, *, space: str, version: str | None = None, model_builder=None) -> None:
        try:
            self.search_space_id = TRANSNAS_SPACES[space]
        except KeyError as error:
            raise ValueError(f"Unknown TransNAS space {space!r}: {sorted(TRANSNAS_SPACES)}") from error
        self.space = space
        expected_protocol = f"transnasbench101-{space}-final"
        super().__init__(
            path,
            benchmark_id=self.benchmark_id,
            search_space_id=self.search_space_id,
            version=version,
            model_builder=model_builder or default_model_builder,
        )
        protocols = {record.get("protocol") for record in self._records}
        legacy_protocol_fixture = protocols == {None} and version is None
        if protocols != {expected_protocol} and not legacy_protocol_fixture:
            raise ValueError(
                f"Expected protocol {expected_protocol!r}, found {sorted(map(str, protocols))}"
            )
        self.protocol = expected_protocol if protocols == {expected_protocol} else None
        source_hashes = {
            str(record["source_sha256"])
            for record in self._records
            if record.get("source_sha256")
        }
        if len(source_hashes) > 1:
            raise ValueError("TransNAS records contain multiple source_sha256 values")
        self.source_sha256 = next(iter(source_hashes), None)
        self.converted_file_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def canonicalize(self, specification: Any) -> Mapping[str, Any]:
        if isinstance(specification, str):
            specification = {"architecture": specification}
        if not isinstance(specification, Mapping) or not isinstance(
            specification.get("architecture"), str
        ):
            raise ValueError("TransNAS specification must contain an architecture string")
        return {"architecture": specification["architecture"].strip()}

    def metadata(self) -> Mapping[str, Any]:
        return {
            **super().metadata(),
            "space": self.space,
            "protocol": self.protocol,
            "model_fidelity": "reference_topology_pytorch_port",
            "model_protocol": "official-encoder-and-task-head-pytorch-port",
            "implementation_source": "https://github.com/yawen-d/TransNASBench",
            "implementation_commit": "6d4231b1eb04e95750a5b2b6cf391db770bc25d6",
            "source_sha256": self.source_sha256,
            "converted_file_sha256": self.converted_file_sha256,
        }

    def iter_architectures(
        self, start: int = 0, end: int | None = None
    ) -> Iterator[Architecture]:
        return _ArchitectureIterator(
            super().iter_architectures(start, end),
            {
                "search_space_id": self.search_space_id,
                "benchmark_variant": self.space,
                "source_sha256": self.source_sha256,
                "converted_file_sha256": self.converted_file_sha256,
            },
        )
