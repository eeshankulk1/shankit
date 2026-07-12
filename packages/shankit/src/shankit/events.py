"""The typed event stream (design §6).

This is the single source of truth for what a streaming run emits. The
TypeScript consumer SDK's event types are generated from these models
(``scripts/generate_ts_events.py``), so the two cannot drift.

A stream ends with exactly one terminal event: ``done`` on success or
``error`` on failure.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .usage import Usage

__all__ = [
    "AgentEvent",
    "DoneEvent",
    "ErrorEvent",
    "Source",
    "SourceEvent",
    "StepEvent",
    "TextDeltaEvent",
    "UsageEvent",
    "agent_event_adapter",
    "error_event_for",
]


class Source(BaseModel):
    """A citation/source surfaced by a tool result.

    Extra fields are allowed so tools can attach vendor-specific metadata
    without the framework needing to know about it.
    """

    model_config = ConfigDict(extra="allow")

    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None


class TextDeltaEvent(BaseModel):
    """A chunk of assistant text as it is generated."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class StepEvent(BaseModel):
    """A user-facing narration step, produced by the step-describer.

    Each tool call yields a ``running`` event followed by a ``done`` or
    ``error`` event with the same ``id``.
    """

    type: Literal["step"] = "step"
    id: str
    title: str
    detail: Optional[str] = None
    phase: Optional[str] = None
    status: Literal["running", "done", "error"] = "running"


class SourceEvent(BaseModel):
    """A source/citation surfaced by a tool during the run."""

    type: Literal["source"] = "source"
    source: Source


class UsageEvent(BaseModel):
    """One usage increment within the run: a model call, or usage a tool
    reported (e.g. a sub-agent's spend surfacing through the tool seam).

    Invariant: the UsageEvents of a stream sum to ``DoneEvent.usage``, so a
    consumer can meter cost live without waiting for the terminal event.
    """

    type: Literal["usage"] = "usage"
    usage: Usage


class ErrorEvent(BaseModel):
    """Terminal event: the run failed.

    ``message`` is human-safe and may be shown to end users. ``code`` says
    *what kind* of failure without parsing the message; the codes emitted
    today are ``model_error`` (the model provider call failed),
    ``max_iterations``, ``output_validation``, ``error`` (other framework
    errors), and ``unexpected`` — the field stays an open string so new
    codes are not a breaking change. ``retryable`` is true when retrying
    the run shortly is reasonable (rate limits, provider overloads).
    """

    type: Literal["error"] = "error"
    message: str
    code: str = "unexpected"
    retryable: bool = False


class DoneEvent(BaseModel):
    """Terminal event: the run finished.

    ``text`` is the transcript: every assistant text pass of the run — the
    same content the ``text_delta`` events streamed — joined with blank
    lines (the deltas themselves carry no separator between passes).
    ``output`` is set only when the run produced a deliverable (for
    ``output_type=str`` runs, the final pass's text); for a plain streamed
    conversation it is ``None``. ``truncated`` is true if any model pass of
    the run (or of a sub-agent run reporting through the tool seam) stopped
    at the token limit, meaning the result may be incomplete.
    """

    type: Literal["done"] = "done"
    text: str = ""
    output: Any = None
    usage: Usage = Field(default_factory=Usage)
    truncated: bool = False


AgentEvent = Annotated[
    Union[TextDeltaEvent, StepEvent, SourceEvent, UsageEvent, ErrorEvent, DoneEvent],
    Field(discriminator="type"),
]


def error_event_for(exc: BaseException) -> ErrorEvent:
    """The one place an exception becomes a terminal ``error`` event, so the
    stream and SSE paths cannot drift: framework errors keep their (already
    human-safe) message, anything else is sanitized, and code/retryable come
    from the exception's type."""
    from .exceptions import ModelError, ShankitError, error_code

    message = str(exc) if isinstance(exc, ShankitError) else "The run failed unexpectedly."
    retryable = exc.retryable if isinstance(exc, ModelError) else False
    return ErrorEvent(message=message, code=error_code(exc), retryable=retryable)


def agent_event_adapter() -> TypeAdapter[Any]:
    """A ``TypeAdapter`` for parsing serialized events back into models."""
    return TypeAdapter(AgentEvent)
