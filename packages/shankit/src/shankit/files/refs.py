"""Reference resolution for agent files (design §5).

Two reference styles, disambiguated by the presence of a colon:

- ``"module:attribute"`` — standard Python import reference (the same
  convention uvicorn and entry points use). Dotted attribute paths after the
  colon are supported (``"pkg.mod:Class.attr"``).
- ``"short_name"`` — a live object explicitly registered in a
  :class:`Registry` (connectors and other non-importables).
"""

from __future__ import annotations

import importlib
from typing import Any

from ..exceptions import AgentFileError
from .registry import Registry

__all__ = ["resolve_ref"]


def resolve_ref(ref: str, registry: Registry) -> Any:
    ref = ref.strip()
    if not ref:
        raise AgentFileError("Empty reference in agent file.")
    if ":" not in ref:
        return registry.get(ref)
    module_path, _, attr_path = ref.partition(":")
    if not module_path or not attr_path:
        raise AgentFileError(
            f"Malformed reference {ref!r}: expected 'module:attribute' or a registered name."
        )
    try:
        obj: Any = importlib.import_module(module_path)
    except ImportError as exc:
        raise AgentFileError(f"Could not import module {module_path!r} for reference {ref!r}: {exc}") from exc
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise AgentFileError(
                f"Module {module_path!r} has no attribute {attr_path!r} (reference {ref!r})."
            ) from None
    return obj
