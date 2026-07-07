"""The provider-neutral model-client contract (design §7.3).

The agent loop talks to models only through :class:`ModelClient`. Requests
carry messages/tools in the framework's one boundary shape (Anthropic
tool-use shape); each client converts to its provider's wire format
internally. Provider-specific response blocks with no place in that shape
(e.g. Anthropic thinking or server-tool blocks) are dropped by the client
during conversion — the boundary carries text and tool use only.

Streaming contract: ``stream()`` yields zero or more ``ModelTextDelta``
events followed by exactly one ``ModelResponseComplete`` carrying the full
response (including any tool-use blocks), so the loop can stream text while
still handling tool calls off the complete response.

Error contract: a failed provider call surfaces as
:class:`shankit.ModelError` (``model_error_for_status`` builds the common
HTTP-status case). A client that raises anything else is tolerated — the
loop wraps unknown exceptions as a fallback — but mapping in the client is
better, because the client knows which of its SDK's failures are transient.
"""

from __future__ import annotations

import abc
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..exceptions import ModelError
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
    "map_sdk_error",
    "model_error_for_status",
    "wrap_sdk_errors",
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


def model_error_for_status(provider: str, status: int) -> ModelError:
    """The :class:`ModelError` for an HTTP status from a provider API.

    Retryability follows what the provider SDKs themselves auto-retry:
    408/409/429 and every 5xx (which covers Anthropic's 529 overloaded).
    Messages are human-safe — no request ids, no response bodies.
    """
    retryable = status in (408, 409, 429) or status >= 500
    if status == 429:
        message = "The model provider is rate-limiting requests; retry shortly."
    elif status >= 500:
        message = "The model provider is temporarily unavailable; retry shortly."
    elif status in (401, 403):
        message = "The model provider rejected the request credentials."
    else:
        message = f"The model provider rejected the request (status {status})."
    return ModelError(message, provider=provider, status=status, retryable=retryable)


def map_sdk_error(
    exc: Exception,
    *,
    provider: str,
    status_error: type[Exception],
    connection_error: type[Exception],
    base_error: type[Exception],
) -> Optional[ModelError]:
    """Map a provider SDK exception onto the neutral error contract.

    The three modern provider SDKs share this exception taxonomy (a status
    error carrying ``status_code``, a connection/timeout error, a catch-all
    base); a client passes its own classes so the mapping — retryability
    policy and human-safe messages — cannot drift between providers.
    Returns ``None`` for exceptions that are not the SDK's.
    """
    if isinstance(exc, status_error):
        return model_error_for_status(provider, exc.status_code)  # type: ignore[attr-defined]
    if isinstance(exc, connection_error):
        return ModelError(
            "Could not reach the model provider; retry shortly.",
            provider=provider,
            retryable=True,
        )
    if isinstance(exc, base_error):
        return ModelError("The model provider call failed.", provider=provider)
    return None


@contextlib.contextmanager
def wrap_sdk_errors(map_error: Callable[[Exception], Optional[ModelError]]) -> Iterator[None]:
    """Re-raise SDK exceptions from the wrapped block as ``ModelError``.

    Exceptions ``map_error`` does not claim (returns ``None``) propagate
    unchanged; the original exception always stays chained as ``__cause__``.
    """
    try:
        yield
    except Exception as exc:
        mapped = map_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def _unused(*_: Any) -> None:  # keep pydantic import shape stable for type checkers
    return None
