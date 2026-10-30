"""A model client that reports each call it makes.

Wrap a provider's client in :class:`ObservedModel` to meter, log or cap
every model call without touching the agents that make them: register the
wrapped client's factory under the provider's name and every ``Agent``
resolving ``"provider:model"`` goes through it.

``before`` runs ahead of each call and may raise to refuse it (a spend
cap); ``after`` gets one :class:`ModelCall` per call once it has ended,
however it ended. An exception from ``after`` is logged, never raised into
the call it describes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
)

__all__ = ["ModelCall", "ObservedModel"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCall:
    """One finished model call.

    Attributes:
        request: What was sent (``request.agent`` names the calling agent).
        streamed: ``stream()`` made the call, not ``complete()``.
        started_at: When the call began (UTC).
        duration: Seconds from the start to the end of the call.
        first_text: Seconds from the start to the first text delta of a
            streamed call; ``None`` for ``complete()`` or a stream with no text.
        response: The response, or ``None`` if the call did not finish.
        error: Why it did not finish: the exception it raised, or ``None``
            with no response when a stream ended without one.
    """

    request: ModelRequest
    streamed: bool
    started_at: datetime
    duration: float
    first_text: Optional[float]
    response: Optional[ModelResponse]
    error: Optional[BaseException]

    @property
    def ok(self) -> bool:
        return self.response is not None

    @property
    def cancelled(self) -> bool:
        """The caller stopped the call (a cancelled task, a stream closed
        before its response), rather than the call failing."""
        return isinstance(self.error, (asyncio.CancelledError, GeneratorExit))


class ObservedModel(ModelClient):
    """``inner``, with ``before`` run ahead of each call and ``after`` told
    how each call went."""

    def __init__(
        self,
        inner: ModelClient,
        *,
        before: Optional[Callable[[ModelRequest], None]] = None,
        after: Optional[Callable[[ModelCall], None]] = None,
    ) -> None:
        self.inner = inner
        self._before = before
        self._after = after

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._before is not None:
            self._before(request)
        clock = _Clock()
        try:
            response = await self.inner.complete(request)
        except BaseException as exc:
            self._report(clock.call(request, streamed=False, error=exc))
            raise
        self._report(clock.call(request, streamed=False, response=response))
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if self._before is not None:
            self._before(request)
        clock = _Clock()
        response: Optional[ModelResponse] = None
        try:
            async for event in self.inner.stream(request):
                if isinstance(event, ModelTextDelta):
                    clock.text()
                elif isinstance(event, ModelResponseComplete):
                    response = event.response
                yield event
        except BaseException as exc:
            # Once the response is in, the call itself succeeded (it was
            # paid for), however the stream ended after it.
            done = response is not None
            self._report(
                clock.call(
                    request,
                    streamed=True,
                    response=response if done else None,
                    error=None if done else exc,
                )
            )
            raise
        self._report(clock.call(request, streamed=True, response=response))

    def _report(self, call: ModelCall) -> None:
        if self._after is None:
            return
        try:
            self._after(call)
        except Exception:  # observing never fails the call
            logger.warning(
                "model call observer failed on a %s call", call.request.model, exc_info=True
            )

    async def aclose(self) -> None:
        await self.inner.aclose()

    def __repr__(self) -> str:
        return f"ObservedModel({self.inner!r})"


class _Clock:
    def __init__(self) -> None:
        self._wall = datetime.now(timezone.utc)
        self._start = time.perf_counter()
        self._text: Optional[float] = None

    def text(self) -> None:
        if self._text is None:
            self._text = time.perf_counter() - self._start

    def call(
        self,
        request: ModelRequest,
        *,
        streamed: bool,
        response: Optional[ModelResponse] = None,
        error: Optional[BaseException] = None,
    ) -> ModelCall:
        return ModelCall(
            request=request,
            streamed=streamed,
            started_at=self._wall,
            duration=time.perf_counter() - self._start,
            first_text=self._text,
            response=response,
            error=error,
        )
