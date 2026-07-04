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
    "Source",
    "TextDeltaEvent",
    "StepEvent",
    "SourceEvent",
    "UsageEvent",
    "ErrorEvent",
    "DoneEvent",
    "AgentEvent",
    "agent_event_adapter",
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
    """Incremental usage for one model call within the run."""

    type: Literal["usage"] = "usage"
    usage: Usage


class ErrorEvent(BaseModel):
    """Terminal event: the run failed."""

    type: Literal["error"] = "error"
    message: str


class DoneEvent(BaseModel):
    """Terminal event: the run finished.

    ``output`` is set only when the run produced a structured deliverable;
    for a plain streamed conversation it is ``None``.
    """

    type: Literal["done"] = "done"
    text: str = ""
    output: Any = None
    usage: Usage = Field(default_factory=Usage)


AgentEvent = Annotated[
    Union[TextDeltaEvent, StepEvent, SourceEvent, UsageEvent, ErrorEvent, DoneEvent],
    Field(discriminator="type"),
]


def agent_event_adapter() -> TypeAdapter[Any]:
    """A ``TypeAdapter`` for parsing serialized events back into models."""
    return TypeAdapter(AgentEvent)
