"""The provider-neutral model-client contract (design §7.3).

The agent loop talks to models only through :class:`ModelClient`. Requests
carry messages/tools in the framework's one boundary shape (Anthropic
tool-use shape); each client converts to its provider's wire format
internally.

Streaming contract: ``stream()`` yields zero or more ``ModelTextDelta``
events followed by exactly one ``ModelResponseComplete`` carrying the full
response (including any tool-use blocks), so the loop can stream text while
still handling tool calls off the complete response.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..messages import ContentBlock, Message
from ..tools.base import ToolDef
from ..usage import Usage

__all__ = [
    "ForcedTool",
    "ModelRequest",
    "ModelResponse",
    "ModelTextDelta",
    "ModelResponseComplete",
    "ModelStreamEvent",
    "ModelClient",
]


class ForcedTool(BaseModel):
    """Tool-choice that forces one specific tool by name."""

    name: str


ToolChoice = Union[Literal["auto", "required", "none"], ForcedTool]


class ModelRequest(BaseModel):
    model: str
    system: Optional[str] = None
    messages: list[Message]
    tools: list[ToolDef] = Field(default_factory=list)
    tool_choice: ToolChoice = "auto"
    max_tokens: int = 4096
    temperature: Optional[float] = None


class ModelResponse(BaseModel):
    content: list[ContentBlock]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"] = "end_turn"
    usage: Usage = Field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if b.type == "text")


class ModelTextDelta(BaseModel):
    text: str


class ModelResponseComplete(BaseModel):
    response: ModelResponse


ModelStreamEvent = Union[ModelTextDelta, ModelResponseComplete]


class ModelClient(abc.ABC):
    """Implement this to plug in a model provider."""

    @abc.abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """One non-streaming model call."""

    @abc.abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """One streaming model call (see module docstring for the contract)."""

    async def aclose(self) -> None:  # noqa: B027  (optional hook, no-op by default)
        """Release underlying transport resources, if any."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def default_stream_from_complete(
    client: ModelClient, request: ModelRequest
) -> AsyncIterator[ModelStreamEvent]:
    """Fallback streaming built on ``complete`` for clients without native streaming."""

    async def _gen() -> AsyncIterator[ModelStreamEvent]:
        response = await client.complete(request)
        text = response.text
        if text:
            yield ModelTextDelta(text=text)
        yield ModelResponseComplete(response=response)

    return _gen()


def _unused(*_: Any) -> None:  # keep pydantic import shape stable for type checkers
    return None
