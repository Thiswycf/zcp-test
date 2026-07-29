from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from zcp_test.types import Architecture


class SearchSpace(ABC):
    search_space_id: str
    model_family: str

    @abstractmethod
    def sample(self, seed: int | None = None) -> Architecture: ...

    @abstractmethod
    def mutate(self, architecture: Architecture, seed: int | None = None) -> Architecture: ...

    @abstractmethod
    def crossover(self, left: Architecture, right: Architecture, seed: int | None = None) -> Architecture: ...

    @abstractmethod
    def canonicalize(self, specification: Mapping[str, Any]) -> Architecture: ...

    @abstractmethod
    def build_model(self, architecture: Architecture, num_classes: int) -> Any: ...

