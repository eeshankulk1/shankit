"""Tracing around a stream: a stream closed from another task must not try
to detach an OpenTelemetry context it attached in the first one."""

from __future__ import annotations

import asyncio
import logging

import pytest
from conftest import text_response, tool_call_response
from shankit import StepEvent, tool

pytest.importorskip("opentelemetry")


async def test_stream_closed_from_another_task_detaches_cleanly(make_agent, caplog):
    @tool
    async def slow() -> str:
        """Takes a while."""
        await asyncio.sleep(10)
        return "never"

    agent, _ = make_agent([tool_call_response("slow", {}), text_response("done")], tools=[slow])
    stream = agent.stream("go")

    async def read_to_step() -> None:
        async for event in stream:
            if isinstance(event, StepEvent):
                return

    # The stream is read in one task until its tool starts...
    await asyncio.create_task(read_to_step())
    # ...and the abandoned stream is closed from another, as a server does
    # when the client goes away.
    with caplog.at_level(logging.ERROR):
        await asyncio.create_task(stream.aclose())

    assert not [r for r in caplog.records if "detach" in r.getMessage().lower()]


async def test_tools_run_inside_the_stream_span(make_agent):
    from opentelemetry import context as otel_context
    from opentelemetry import trace

    seen: list[object] = []

    @tool
    async def probe() -> str:
        """Records the current span."""
        seen.append(trace.get_current_span())
        return "ok"

    outside = otel_context.get_current()
    agent, _ = make_agent([tool_call_response("probe", {}), text_response("done")], tools=[probe])
    async for _ in agent.stream("go"):
        # Between events the caller's own context is current.
        assert otel_context.get_current() == outside
    assert len(seen) == 1
