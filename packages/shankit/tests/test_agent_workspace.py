"""Loop features for long, workspace-backed runs: spill-to-file, the run
transcript, reasoning round-trip, context clearing, and the wall-clock bound."""

from __future__ import annotations

import asyncio

import pytest
from conftest import text_response, tool_call_response, usage
from shankit import (
    DoneEvent,
    ErrorEvent,
    LocalWorkspace,
    Reasoning,
    ReasoningBlock,
    RunTimeoutError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    strip_reasoning,
    tool,
)
from shankit.messages import Message
from shankit.models.base import ModelResponse


async def collect(stream):
    return [event async for event in stream]


# ---------------------------------------------------------------- spill


async def test_oversized_result_spills_to_a_workspace_file(make_agent, tmp_path):
    big = "HEAD" + "x" * 50_000 + "TAIL"

    @tool
    def dump() -> str:
        return big

    ws = LocalWorkspace(tmp_path)
    agent, _ = make_agent(
        [tool_call_response("dump", {}), text_response("done")],
        tools=[dump],
        workspace=ws,
        spill_threshold_chars=1000,
    )
    result = await agent.run("go", output_type=str)
    seen = result.trajectory[0].content
    assert len(seen) < 5000
    assert seen.startswith("HEAD")
    assert "TAIL" in seen
    path = seen.rsplit("saved to ", 1)[1].split(".txt", 1)[0] + ".txt"
    assert path.startswith(ws.tmp + "/tool-output/dump-")
    assert (await ws.read_file(path)).decode() == big
    assert result.truncated is False  # lossless: nothing was cut


async def test_workspace_resolves_from_context_for_spill(make_agent, tmp_path):
    @tool
    def dump() -> str:
        return "y" * 3000

    spaces: dict[str, LocalWorkspace] = {}

    def resolve(ctx):
        return spaces.setdefault(ctx["user"], LocalWorkspace(tmp_path / ctx["user"]))

    agent, _ = make_agent(
        [tool_call_response("dump", {}), text_response("ok")],
        tools=[dump],
        workspace=resolve,
        spill_threshold_chars=100,
    )
    await agent.run("go", context={"user": "alice"}, output_type=str)
    assert list((tmp_path / "alice" / "tmp" / "tool-output").iterdir())


async def test_without_a_workspace_the_cap_still_applies(make_agent):
    @tool
    def dump() -> str:
        return "z" * 5000

    agent, _ = make_agent(
        [tool_call_response("dump", {}), text_response("ok")],
        tools=[dump],
        spill_threshold_chars=100,
        max_tool_result_chars=200,
    )
    result = await agent.run("go", output_type=str)
    assert result.trajectory[0].content.endswith("[truncated: tool result exceeded 200 characters]")
    assert result.truncated is True


# ----------------------------------------------------------- transcript


async def test_done_event_carries_the_run_transcript_not_the_history(make_agent):
    @tool
    def check() -> str:
        return "42"

    agent, _ = make_agent(
        [tool_call_response("check", {}, text="Looking."), text_response("It's 42.")],
        tools=[check],
    )
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    events = await collect(agent.stream("what is it?", history=history))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    roles = [m.role for m in done.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert done.messages[0].content[0].text == "what is it?"
    assert isinstance(done.messages[1].content[1], ToolUseBlock)
    assert isinstance(done.messages[2].content[0], ToolResultBlock)
    assert done.messages[2].content[0].content == "42"
    # never on the wire
    assert "messages" not in done.model_dump_json()


async def test_run_result_carries_the_transcript_and_it_replays(make_agent):
    agent, fake = make_agent([text_response("first"), text_response("second")])
    first = await agent.run("one", output_type=str)
    assert [m.role for m in first.messages] == ["user", "assistant"]
    await agent.run("two", output_type=str, history=first.messages)
    sent = fake.requests[1].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"]


# ------------------------------------------------------------ reasoning


def reasoning_block(sig: str = "sig") -> ReasoningBlock:
    return ReasoningBlock(
        provider="anthropic", data={"type": "thinking", "thinking": "hmm", "signature": sig}
    )


async def test_reasoning_config_reaches_the_request_and_blocks_round_trip(make_agent):
    @tool
    def check() -> str:
        return "ok"

    first = ModelResponse(
        content=[reasoning_block(), ToolUseBlock(id="t1", name="check", input={})],
        stop_reason="tool_use",
        usage=usage(),
    )
    agent, fake = make_agent([first, text_response("done")], tools=[check], reasoning="high")
    await agent.run("go", output_type=str)
    assert fake.requests[0].reasoning == Reasoning(effort="high")
    # the assistant pass is re-sent with its reasoning block intact
    replayed = fake.requests[1].messages[1].content
    assert isinstance(replayed[0], ReasoningBlock)
    assert replayed[0].data["signature"] == "sig"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, None),
        (True, Reasoning()),
        (False, Reasoning(enabled=False)),
        ("low", Reasoning(effort="low")),
        (Reasoning(budget_tokens=2048), Reasoning(budget_tokens=2048)),
    ],
)
def test_reasoning_spec_normalizes(make_agent, spec, expected):
    agent, _ = make_agent([], reasoning=spec)
    assert agent.reasoning == expected


def test_strip_reasoning_drops_blocks_and_empty_messages():
    messages = [
        Message(role="assistant", content=[reasoning_block()]),
        Message(role="assistant", content=[reasoning_block(), TextBlock(text="hi")]),
    ]
    out = strip_reasoning(messages)
    assert len(out) == 1
    assert out[0].content == [TextBlock(text="hi")]
    assert isinstance(messages[1].content[0], ReasoningBlock)  # originals untouched


# ------------------------------------------------------ context clearing


def big_usage(tokens: int):
    u = usage()
    u.input_tokens = tokens
    return u


async def test_context_clearing_stubs_old_results_and_keeps_recent(make_agent, tmp_path):
    @tool
    def fetch(n: int) -> str:
        return f"result-{n}-" + "d" * 3000

    responses = []
    for i in range(3):
        r = tool_call_response("fetch", {"n": i}, call_id=f"t{i}")
        r.content.insert(0, reasoning_block(f"s{i}"))
        r.usage = big_usage(10_000 if i < 2 else 90_000)
        responses.append(r)
    responses.append(text_response("final"))
    ws = LocalWorkspace(tmp_path)
    agent, fake = make_agent(
        responses,
        tools=[fetch],
        workspace=ws,
        reasoning=True,
        context_clear_threshold_tokens=50_000,
        context_keep_recent_results=1,
    )
    history = [Message(role="user", content=[TextBlock(text="hi")])]
    await agent.run("go", output_type=str, history=history)

    last = fake.requests[-1]
    results = [b for m in last.messages for b in m.content if isinstance(b, ToolResultBlock)]
    assert "cleared to save context" in results[0].content
    assert "cleared to save context" in results[1].content
    assert results[2].content.startswith("result-2-")
    assert "cleared" not in results[2].content
    # the cleared text is recoverable from the workspace
    path = results[0].content.split("Full output: ", 1)[1].rstrip(".]")
    assert (await ws.read_file(path)).decode().startswith("result-0-")
    # reasoning dropped everywhere, and the pass right after the clear runs without it
    assert not any(isinstance(b, ReasoningBlock) for m in last.messages for b in m.content)
    assert last.reasoning == Reasoning(enabled=False)
    # earlier passes weren't affected
    assert fake.requests[1].reasoning == Reasoning()


async def test_no_clearing_below_threshold(make_agent):
    @tool
    def fetch() -> str:
        return "d" * 3000

    agent, fake = make_agent(
        [tool_call_response("fetch", {}), text_response("x")],
        tools=[fetch],
        context_clear_threshold_tokens=1_000_000,
    )
    await agent.run("go", output_type=str)
    result = fake.requests[-1].messages[-1].content[0]
    assert "cleared" not in result.content


# -------------------------------------------------------------- timeout


async def test_timeout_stops_the_run_between_passes(make_agent):
    @tool
    async def slow() -> str:
        await asyncio.sleep(0.2)
        return "ok"

    agent, _ = make_agent(
        [tool_call_response("slow", {}), text_response("never")],
        tools=[slow],
        timeout_s=0.05,
    )
    with pytest.raises(RunTimeoutError):
        await agent.run("go", output_type=str)
    agent, _ = make_agent(
        [tool_call_response("slow", {}), text_response("never")],
        tools=[slow],
        timeout_s=0.05,
    )
    events = await collect(agent.stream("go"))
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "timeout"
