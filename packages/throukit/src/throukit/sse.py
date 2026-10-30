"""Server-sent events serialization for the event stream (design §8).

This is the wire format the TypeScript consumer SDK parses. Each event is
one SSE message whose ``event`` field is the event type and whose ``data``
is the JSON-serialized event (which also carries ``type``, so clients that
only read ``data`` lines still work).

Example with Starlette/FastAPI::

    @app.post("/agents/support/stream")
    async def stream(request: Request):
        events = agent.stream(prompt, context=ctx)
        return StreamingResponse(sse_stream(events), media_type="text/event-stream")
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Union

from pydantic import BaseModel

from .events import error_event_for

__all__ = ["format_sse", "sse_stream"]


def format_sse(event: BaseModel) -> str:
    """Serialize one event as an SSE message."""
    event_type = getattr(event, "type", "message")
    data = event.model_dump_json()
    return f"event: {event_type}\ndata: {data}\n\n"


async def sse_stream(
    events: AsyncIterator[BaseModel], *, errors: str = "emit"
) -> AsyncIterator[Union[str, bytes]]:
    """Map an agent event stream onto SSE lines.

    With ``errors="emit"`` (default), an unexpected exception is emitted as a
    terminal ``error`` event with a sanitized message instead of tearing the
    HTTP response mid-stream; framework errors already surface as ``error``
    events from ``Agent.stream``. Pass ``errors="raise"`` to propagate.
    """
    errored = False
    try:
        async for event in events:
            errored = getattr(event, "type", None) == "error"
            yield format_sse(event)
    except Exception as exc:
        if errors == "raise":
            raise
        if not errored:
            # Agent.stream yields its own terminal error event before
            # re-raising an unexpected exception; don't emit a second one.
            yield format_sse(error_event_for(exc))
