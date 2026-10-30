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

    Routing only remembers which *source* owns a name — per-run visibility
    is the agent loop's job (it restricts calls to what this run listed),
    and per-caller authorization is the owning source's job: a source whose
    catalog varies by context must authorize in ``execute``, never assume
    ``list_tools`` gated access.
    """

    def __init__(self, sources: Sequence[ToolSource]) -> None:
        self.sources = list(sources)
        self._routes: dict[str, ToolSource] = {}

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        seen: dict[str, ToolDef] = {}
        routes: dict[str, ToolSource] = {}
        for source in self.sources:
            for tool_def in await source.list_tools(context):
                if tool_def.name in seen:
                    raise ValueError(
                        f"Tool name collision across sources: {tool_def.name!r}. "
                        "Tool names must be unique within an agent."
                    )
                seen[tool_def.name] = tool_def
                routes[tool_def.name] = source
        # Merge, don't replace: concurrent runs share this instance, and a
        # run listing with context B must not evict the routes a run with
        # context A is about to execute against. Name→source ownership is
        # stable for a fixed source set, so merging is safe.
        self._routes.update(routes)
        return list(seen.values())

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        # Route on the name→source map captured by the last list_tools call
        # (the loop always lists before executing); re-listing every source
        # per call would cost a round-trip per network-backed source.
        source = self._routes.get(name)
        if source is None:
            await self.list_tools(context)
            source = self._routes.get(name)
        if source is None:
            raise ToolNotFoundError(name)
        return await source.execute(name, arguments, context)
