"""Tests for the additions harvested from throu, the first real consumer:
conversation history, cache-read accounting, prompt caching, concurrent
tool execution, and OpenAI reasoning-model compatibility."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from conftest import text_response
from shankit import Message, TextBlock, ToolDef, ToolResult, ToolSource, Usage, coerce_message
from shankit.models.anthropic import build_kwargs as anthropic_kwargs
from shankit.models.base import ModelRequest
from shankit.models.openai import build_kwargs as openai_kwargs

# ---------------------------------------------------------------- history


async def test_run_with_history_prepends_turns(make_agent):
    agent, fake = make_agent([text_response("hi again")])
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = await agent.run("hello again", history=history, output_type=str)
    assert result.output == "hi again"

    sent = fake.requests[0].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert sent[0].content[0].text == "hello"
    assert sent[1].content[0].text == "hi there"
    assert sent[2].content[0].text == "hello again"


async def test_stream_with_history_message_objects(make_agent):
    agent, fake = make_agent([text_response("done")])
    history = [Message(role="user", content=[TextBlock(text="earlier")])]
    async for _ in agent.stream("now", history=history):
        pass
    assert [m.role for m in fake.requests[0].messages] == ["user", "user"]


def test_coerce_message_rejects_garbage():
    import pytest

    with pytest.raises(TypeError):
        coerce_message(42)


# ------------------------------------------------------- usage accounting


def test_usage_cache_read_is_additive():
    total = Usage()
    total.add(Usage(input_tokens=10, output_tokens=5, cache_read_tokens=100, requests=1))
    total.add(Usage(input_tokens=1, output_tokens=1, cache_read_tokens=50, requests=1))
    assert total.cache_read_tokens == 150
    summed = Usage(cache_read_tokens=1) + Usage(cache_read_tokens=2)
    assert summed.cache_read_tokens == 3


def test_usage_cache_write_is_additive():
    total = Usage()
    total.add(Usage(cache_write_tokens=1200, requests=1))
    total.add(Usage(cache_write_tokens=300, requests=1))
    assert total.cache_write_tokens == 1500
    summed = Usage(cache_write_tokens=4) + Usage(cache_write_tokens=5)
    assert summed.cache_write_tokens == 9


# ------------------------------------------------- provider kwargs shape


def _request() -> ModelRequest:
    return ModelRequest(
        model="m",
        system="sys",
        messages=[Message(role="user", content=[TextBlock(text="q")])],
        tools=[ToolDef(name="t", description="d")],
        max_tokens=123,
    )


def test_anthropic_cache_control_flag():
    assert anthropic_kwargs(_request(), cache_system_and_tools=True)["cache_control"] == {
        "type": "ephemeral"
    }
    assert "cache_control" not in anthropic_kwargs(_request())


def test_openai_uses_max_completion_tokens():
    kwargs = openai_kwargs(_request())
    assert kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in kwargs


# ------------------------------------------------ concurrent tool dispatch


class SlowSource(ToolSource):
    """Two tools that each sleep; records concurrency."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.finished: list[str] = []

    async def list_tools(self, context: Any = None):
        return [ToolDef(name="slow_a"), ToolDef(name="slow_b")]

    async def execute(self, name: str, arguments: dict[str, Any], context: Any = None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        # slow_a takes longer, so completion order is b-then-a while
        # emission order is a-then-b.
        await asyncio.sleep(0.05 if name == "slow_a" else 0.01)
        self.active -= 1
        self.finished.append(name)
        return ToolResult(content=f"{name} ok")


async def test_parallel_tool_calls_execute_concurrently(make_agent):
    from shankit.messages import ToolUseBlock
    from shankit.models.base import ModelResponse

    turn = ModelResponse(
        content=[
            ToolUseBlock(id="a", name="slow_a", input={}),
            ToolUseBlock(id="b", name="slow_b", input={}),
        ],
        stop_reason="tool_use",
    )
    source = SlowSource()
    agent, fake = make_agent([turn, text_response("done")], tools=[source])

    start = time.perf_counter()
    result = await agent.run("go", output_type=str)
    elapsed = time.perf_counter() - start

    assert result.output == "done"
    assert source.max_active == 2, "tool calls in one turn should overlap"
    assert source.finished == ["slow_b", "slow_a"]
    assert elapsed < 0.09, "wall clock should be ~max, not sum, of tool latencies"

    # Trajectory and tool_result blocks stay in the model's emission order.
    assert [r.tool for r in result.trajectory] == ["slow_a", "slow_b"]
    follow_up = fake.requests[1].messages[-1]
    assert [b.tool_use_id for b in follow_up.content] == ["a", "b"]


async def test_parallel_step_events_stay_ordered(make_agent):
    from shankit.events import DoneEvent, StepEvent
    from shankit.messages import ToolUseBlock
    from shankit.models.base import ModelResponse

    turn = ModelResponse(
        content=[
            ToolUseBlock(id="a", name="slow_a", input={}),
            ToolUseBlock(id="b", name="slow_b", input={}),
        ],
        stop_reason="tool_use",
    )
    agent, _ = make_agent([turn, text_response("done")], tools=[SlowSource()])

    events = [e async for e in agent.stream("go")]
    steps = [e for e in events if isinstance(e, StepEvent)]
    running = [s.id for s in steps if s.status == "running"]
    finished = [s.id for s in steps if s.status == "done"]
    assert running == ["s1", "s2"], "running steps emit upfront in emission order"
    assert finished == ["s1", "s2"], "completion steps report in emission order"
    assert isinstance(events[-1], DoneEvent)
