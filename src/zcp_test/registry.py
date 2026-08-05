from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}
        self._retired: dict[str, str] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._entries:
            raise KeyError(f"Duplicate {self.kind}: {name}")
        self._entries[name] = factory

    def create(self, name: str, **kwargs: Any) -> T:
        if name in self._retired:
            raise KeyError(f"Retired {self.kind} {name!r}: {self._retired[name]}")
        try:
            factory = self._entries[name]
        except KeyError as error:
            raise KeyError(f"Unknown {self.kind} {name!r}; available: {self.names()}") from error
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def items(self) -> list[tuple[str, Callable[..., T]]]:
        return sorted(self._entries.items())

    def retire(self, name: str, reason: str) -> None:
        if name in self._entries:
            raise KeyError(f"Cannot retire registered {self.kind}: {name}")
        self._retired[name] = reason
