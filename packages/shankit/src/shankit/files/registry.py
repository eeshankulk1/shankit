"""Explicit registry for live objects referenced from agent files (design §5).

Anything importable is referenced from a file by its ordinary
``module:attribute`` path and never needs this. The registry exists only for
things that genuinely can't be reached by import — live, configured objects
like connectors — and registration is always explicit: no auto-scanning, no
guessing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

from ..exceptions import AgentFileError

__all__ = ["Registry", "default_registry", "register"]


class Registry:
    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        """Register a live object under a short name for use in agent files."""
        if ":" in name:
            raise ValueError(
                f"Registry names may not contain ':' ({name!r}); "
                "colons denote module:attribute import references."
            )
        self._objects[name] = obj

    def get(self, name: str) -> Any:
        try:
            return self._objects[name]
        except KeyError:
            known = ", ".join(sorted(self._objects)) or "(nothing registered)"
            raise AgentFileError(
                f"Nothing registered under {name!r}. Registered names: {known}. "
                "Register live objects with registry.register(name, obj); "
                "importable objects should be referenced as 'module:attribute' instead."
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._objects

    def __iter__(self) -> Iterator[str]:
        return iter(self._objects)


default_registry = Registry()


def register(name: str, obj: Any, *, registry: Optional[Registry] = None) -> None:
    """Register into the default registry (or an explicit one)."""
    (registry or default_registry).register(name, obj)
