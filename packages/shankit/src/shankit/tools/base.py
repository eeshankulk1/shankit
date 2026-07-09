"""The tool seam (design §3.1).

The single abstraction the agent loop knows about for tools. It exposes
exactly two capabilities — list the tools it offers and execute one of them —
and both receive the opaque per-run context and nothing more. Tool
definitions are in Anthropic tool-use shape at the boundary.

This contract is deliberately too small to over-abstract. Everything else
(local functions, MCP servers, connectors, sub-agents) is an implementation
of it.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..events import Source
from ..usage import Usage

__all__ = ["ToolDef", "ToolResult", "ToolSource", "is_tool_source"]


class ToolDef(BaseModel):
    """A tool definition in Anthropic tool-use shape."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolResult(BaseModel):
    """What executing a tool returns to the loop.

    ``content`` is what the model sees. ``sources`` surface citations to the
    event stream. ``usage`` lets a tool that itself spends tokens (e.g. a
    sub-agent) report them for accounting in the parent run. ``truncated``
    marks ``content`` as cut off (e.g. a sub-agent run that hit its token
    limit); the parent run's sticky ``truncated`` flag picks it up.
    """

    content: str
    is_error: bool = False
    sources: list[Source] = Field(default_factory=list)
    usage: Optional[Usage] = None
    truncated: bool = False


class ToolSource(abc.ABC):
    """Anything that can offer tools to an agent.

    Both methods receive the opaque per-run context (design §3.3). The
    framework never reads inside it; a source may use it to resolve *whose*
    credentials to use without the loop knowing what a "user" is.

    Error contract: raise :class:`shankit.ToolError` from ``execute`` (or
    return ``ToolResult(is_error=True)``) for a controlled, model-visible
    failure. Any other exception is sanitized by the loop before the model
    sees it.
    """

    @abc.abstractmethod
    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        """The tools this source currently offers."""

    @abc.abstractmethod
    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        """Execute one tool by name.

        Must raise :class:`shankit.ToolNotFoundError` for unknown names.
        """


def is_tool_source(obj: Any) -> bool:
    """True for ``ToolSource`` subclasses and duck-typed equivalents."""
    if isinstance(obj, ToolSource):
        return True
    return callable(getattr(obj, "list_tools", None)) and callable(getattr(obj, "execute", None))
