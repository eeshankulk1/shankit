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
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from ..events import Artifact, Source
from ..messages import ToolResultContent
from ..usage import Usage

__all__ = ["NOT_EXECUTED", "ToolDef", "ToolResult", "ToolSource", "Toolset", "is_tool_source"]

#: What an ordered toolset's calls get after an earlier call in the same
#: turn failed (Anthropic's text for its native toolsets).
NOT_EXECUTED = "Not executed: an earlier action in this turn failed."


class Toolset(BaseModel):
    """A group of tools a provider may offer natively as one entry.

    Each member is an ordinary :class:`ToolDef` (name, description, schema:
    the provider-neutral mirror) carrying the same ``Toolset``. A model
    client with a ``native`` entry for its provider sends that entry once in
    place of the members (e.g. Anthropic's ``{"type":
    "browser_toolset_20260801"}``), and round-trips ``name`` on the calls and
    their results; every other client sends the members as plain tools.

    ``ordered``: calls to this toolset within one model turn run one at a
    time, in the order the model emitted them, and stop at the first
    failure - the rest get ``not_executed`` as an error result. For tools
    whose calls act on shared state in sequence (a browser: click, type,
    press Enter), where running them concurrently or past a failure would
    act on the wrong page.
    """

    name: str
    native: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ordered: bool = False
    not_executed: str = NOT_EXECUTED


class ToolDef(BaseModel):
    """A tool definition in Anthropic tool-use shape (plus, for a member of
    a native toolset, the :class:`Toolset` it belongs to)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    toolset: Optional[Toolset] = None


class ToolResult(BaseModel):
    """What executing a tool returns to the loop.

    ``content`` is what the model sees: text, or a list of text, image and
    provider blocks (a screenshot with a caption, a provider's opaque state
    block). ``sources`` surface citations to the
    event stream; ``artifacts`` surface typed structured payloads the same
    way (the model never sees either — they are for the caller). ``usage``
    lets a tool that itself spends tokens (e.g. a sub-agent) report them for
    accounting in the parent run. ``truncated`` marks ``content`` as cut off
    (e.g. a sub-agent run that hit its token limit); the parent run's sticky
    ``truncated`` flag picks it up. ``end_run`` ends the run once this tool
    batch is processed, instead of going back to the model: for a tool whose
    result the model has nothing to add to, like a question put to a human
    whose answer arrives later as a new message.
    """

    content: Union[str, list[ToolResultContent]]
    is_error: bool = False
    sources: list[Source] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    usage: Optional[Usage] = None
    truncated: bool = False
    end_run: bool = False


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


def single_task_tool_def(name: str, description: str) -> ToolDef:
    """The one-tool schema shared by delegation-style sources (a sub-agent,
    a network as a tool): a single required ``task`` string."""
    return ToolDef(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for this agent, in plain language.",
                }
            },
            "required": ["task"],
        },
    )
