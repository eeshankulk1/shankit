"""Optional OpenTelemetry tracing (design §6).

Dev tracing is orthogonal to the user-facing event stream. If
``opentelemetry-api`` is installed (``pip install shankit[otel]``), the agent
loop emits spans; otherwise every span here is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - exercised only when otel is installed
    from opentelemetry import trace as _trace

    _tracer = _trace.get_tracer("shankit")

    @contextmanager
    def span(name: str, **attributes: Any) -> Iterator[None]:
        with _tracer.start_as_current_span(name) as s:
            for key, value in attributes.items():
                if value is not None:
                    s.set_attribute(key, value)
            yield

except ImportError:

    @contextmanager
    def span(name: str, **attributes: Any) -> Iterator[None]:
        yield
