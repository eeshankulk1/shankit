"""Combining several tool sources into one (design §3.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..exceptions import ToolNotFoundError
from .base import ToolDef, ToolResult, ToolSource

__all__ = ["CompositeToolSource"]


class CompositeToolSource(ToolSource):
    """Aggregates several tool sources behind the same two-capability seam.

    Tool names must be unique across sources; a collision raises at
    ``list_tools`` time so it surfaces before the model ever runs.
    """

    def __init__(self, sources: Sequence[ToolSource]) -> None:
        self.sources = list(sources)

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        seen: dict[str, ToolDef] = {}
        for source in self.sources:
            for tool_def in await source.list_tools(context):
                if tool_def.name in seen:
                    raise ValueError(
                        f"Tool name collision across sources: {tool_def.name!r}. "
                        "Tool names must be unique within an agent."
                    )
                seen[tool_def.name] = tool_def
        return list(seen.values())

    async def execute(self, name: str, arguments: dict[str, Any], context: Any = None) -> ToolResult:
        for source in self.sources:
            names = {t.name for t in await source.list_tools(context)}
            if name in names:
                return await source.execute(name, arguments, context)
        raise ToolNotFoundError(name)
