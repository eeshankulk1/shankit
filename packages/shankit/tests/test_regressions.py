"""Regression tests for fixed defects; each test names the failure it pins."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import text_response, tool_call_response
from pydantic import BaseModel
from shankit import Agent, CompositeToolSource, ShankitError, ToolDef, ToolResult, tool
from shankit.models.base import ModelClient, ModelRequest, ModelResponse
from shankit.models.registry import register_provider, resolve_model


class _Answer(BaseModel):
    value: int


async def test_user_tool_named_final_result_is_rejected_up_front(make_agent):
    """A user tool colliding with the synthetic output tool used to produce
    a provider 400 (duplicate tool name) or silently shadow the user tool."""

    @tool
    def final_result(ctx: Any) -> str:
        """Unluckily named user tool."""
        return "x"

    agent, _ = make_agent([], tools=[final_result], output_type=_Answer)
    with pytest.raises(ShankitError, match="final_result"):
        await agent.run("go")


async def test_str_output_is_the_final_pass_even_when_empty(make_agent):
    """An empty final pass must not promote the previous pass's interim
    narration to 'the answer'."""

    @tool
    def noop(ctx: Any) -> str:
        """Do nothing."""
        return "done"

    agent, _ = make_agent(
        [
            tool_call_response("noop", {}, text="Let me do that."),
            ModelResponse(content=[], stop_reason="end_turn"),
        ],
        tools=[noop],
    )
    result = await agent.run("go", output_type=str)
    assert result.output == ""
    assert result.text == "Let me do that."  # the transcript still has it


def test_client_cache_is_per_event_loop():
    """A cached provider client is bound to the loop it first ran on; a
    second asyncio.run() in the same process must get a fresh client."""
    instances: list[object] = []

    class _Client(ModelClient):
        def __init__(self) -> None:
            instances.append(self)

        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise NotImplementedError

    register_provider("percall", _Client)

    async def use() -> object:
        client, _ = resolve_model("percall:m")
        again, _ = resolve_model("percall:m")
        assert again is client  # cached within one loop
        return client

    first = asyncio.run(use())
    second = asyncio.run(use())
    assert first is not second  # never reused across loops


async def test_composite_listing_does_not_evict_other_contexts_routes():
    """Two runs sharing one CompositeToolSource: run B listing (with a
    context that sees different tools) must not break run A's routing."""

    class PerContextSource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def list_tools(self, context: Any = None) -> list[ToolDef]:
            if context == self.name:
                return [ToolDef(name=f"{self.name}_tool", description="")]
            return []

        async def execute(self, name: str, arguments: dict, context: Any = None) -> ToolResult:
            return ToolResult(content=f"ran {name}")

    composite = CompositeToolSource([PerContextSource("a"), PerContextSource("b")])
    await composite.list_tools("a")  # run A lists
    await composite.list_tools("b")  # run B lists (previously evicted A's routes)
    result = await composite.execute("a_tool", {}, context="a")
    assert result.content == "ran a_tool"


async def test_custom_client_needs_only_complete(make_agent):
    """ModelClient.stream defaults to the complete()-based fallback, so a
    minimal client works in both run modes."""

    class MinimalClient(ModelClient):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return text_response("hi there")

    agent = Agent(name="minimal", model="anything", model_client=MinimalClient())
    texts = [e.text async for e in agent.stream("hello") if e.type == "text_delta"]
    assert "".join(texts) == "hi there"


def test_evals_namespace_is_importable_off_the_package():
    import shankit

    assert shankit.evals.Dataset is not None


async def test_frontmatter_value_containing_dashes(tmp_path):
    """A '----' divider (or '---' inside a value) must not terminate the
    frontmatter block early."""
    path = tmp_path / "agent.md"
    path.write_text(
        "---\n"
        "name: divider\n"
        'description: "uses ---- dividers"\n'
        "model: fake:m\n"
        "---\n"
        "Intro.\n"
        "----\n"
        "Outro with --- inline.\n",
        encoding="utf-8",
    )
    from shankit import load_agent

    agent = load_agent(path)
    assert agent.name == "divider"
    assert agent.description == "uses ---- dividers"
    assert "----" in agent.instructions
    assert "--- inline" in agent.instructions


async def test_sqlite_checkpointer_releases_connections(tmp_path):
    """Every operation must release its sqlite connection (fd + WAL lock);
    an exclusive lock is obtainable right after heavy use."""
    import sqlite3

    from shankit import Checkpoint, SqliteCheckpointer

    db = tmp_path / "threads.db"
    store = SqliteCheckpointer(db)
    for i in range(20):
        await store.save(f"t{i}", Checkpoint(state={"i": i}))
        await store.load(f"t{i}")
    await store.delete("t0")

    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")  # fails if another handle holds a lock
        conn.rollback()
    finally:
        conn.close()
