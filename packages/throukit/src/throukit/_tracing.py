"""Optional OpenTelemetry tracing (design §6).

Dev tracing is orthogonal to the user-facing event stream. If
``opentelemetry-api`` is installed (``pip install throukit[otel]``), the agent
loop emits spans; otherwise every span here is a no-op.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - exercised only when otel is installed
    from opentelemetry import context as _context
    from opentelemetry import trace as _trace

    _tracer = _trace.get_tracer("throukit")

    @contextmanager
    def span(name: str, **attributes: Any) -> Iterator[None]:
        with _tracer.start_as_current_span(name) as s:
            for key, value in attributes.items():
                if value is not None:
                    s.set_attribute(key, value)
            yield

    async def traced(  # pyright: ignore[reportRedeclaration]
        name: str, events: AsyncIterator[T], **attributes: Any
    ) -> AsyncGenerator[T, None]:
        """Yield ``events`` inside a span. The span is the current one only
        while ``events`` works, never across a ``yield``: a consumer that
        abandons the stream closes it from its own context, where a context
        attached here could not be detached."""
        s = _tracer.start_span(
            name, attributes={k: v for k, v in attributes.items() if v is not None}
        )
        ctx = _trace.set_span_in_context(s)
        try:
            while True:
                token = _context.attach(ctx)
                try:
                    event = await events.__anext__()
                except StopAsyncIteration:
                    return
                except BaseException as exc:
                    s.record_exception(exc)
                    s.set_status(_trace.Status(_trace.StatusCode.ERROR))
                    raise
                finally:
                    _context.detach(token)
                yield event
        finally:
            s.end()
            await _aclose(events)

except ImportError:

    @contextmanager
    def span(name: str, **attributes: Any) -> Iterator[None]:
        yield

    async def traced(
        name: str, events: AsyncIterator[T], **attributes: Any
    ) -> AsyncGenerator[T, None]:
        try:
            async for event in events:
                yield event
        finally:
            await _aclose(events)


async def _aclose(events: AsyncIterator[Any]) -> None:
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        await aclose()
