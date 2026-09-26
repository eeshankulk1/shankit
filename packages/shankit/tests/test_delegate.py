"""Delegation: one `delegate` tool over a sub-agent roster, and live tool events."""

from __future__ import annotations

import asyncio
import json

from conftest import FakeModel, text_response, tool_call_response
from shankit import (
    Agent,
    AgentDelegate,
    Delegate,
    DelegateToolSource,
    LocalWorkspace,
    StepEvent,
    ToolResult,
    Usage,
    current_tool_call,
    tool,
)
from shankit.observe import StepInfo


def sub_agent(name: str, responses, **kwargs) -> tuple[Agent, FakeModel]:
    fake = FakeModel(responses)
    return (
        Agent(
            name=name,
            model="fake",
            model_client=fake,
            description=f"{name} specialist",
            **kwargs,
        ),
        fake,
    )


def parent(responses, tools, **kwargs) -> tuple[Agent, FakeModel]:
    fake = FakeModel(responses)
    return Agent(name="main", model="fake", model_client=fake, tools=tools, **kwargs), fake


async def collect(stream):
    return [event async for event in stream]


async def test_one_tool_lists_the_roster():
    a, _ = sub_agent("mail", [])
    b, _ = sub_agent("calendar", [])
    source = DelegateToolSource([a, b], extra_properties={"mode": {"type": "string"}})
    [tool_def] = await source.list_tools()
    assert tool_def.name == "delegate"
    assert "- mail: mail specialist" in tool_def.description
    assert tool_def.input_schema["properties"]["agent"]["enum"] == ["mail", "calendar"]
    assert "mode" in tool_def.input_schema["properties"]


async def test_delegate_runs_the_sub_agent_and_rolls_up_usage():
    sub, _ = sub_agent("mail", [text_response("3 unread from Sam")])
    main, _ = parent(
        [
            tool_call_response("delegate", {"agent": "mail", "task": "unread from Sam?"}),
            text_response("You have 3."),
        ],
        [DelegateToolSource([sub])],
    )
    result = await main.run("any mail from sam", output_type=str)
    assert result.trajectory[0].content == "3 unread from Sam"
    assert result.usage.requests == 3  # 2 parent passes + 1 sub-agent pass


async def test_sub_agent_steps_stream_live_with_prefixed_ids():
    @tool
    async def search_inbox(q: str) -> str:
        await asyncio.sleep(0)
        return "found"

    sub, _ = sub_agent(
        "mail",
        [tool_call_response("search_inbox", {"q": "sam"}), text_response("done")],
        tools=[search_inbox],
        describe_step=lambda n, a, c: StepInfo(title="Searched your inbox"),
    )
    main, _ = parent(
        [
            tool_call_response("delegate", {"agent": "mail", "task": "t"}),
            text_response("ok"),
        ],
        [DelegateToolSource([sub])],
        describe_step=None,  # the delegate call itself is hidden
    )
    events = await collect(main.stream("go"))
    steps = [e for e in events if isinstance(e, StepEvent)]
    assert [(s.id, s.status, s.agent) for s in steps] == [
        ("s1.s1", "running", "mail"),
        ("s1.s1", "done", "mail"),
    ]


async def test_budget_and_dedup_return_notices_not_errors():
    sub, _ = sub_agent("mail", [text_response("a"), text_response("b")])
    source = DelegateToolSource([sub], max_calls_per_delegate=2)
    main, _ = parent(
        [
            tool_call_response("delegate", {"agent": "mail", "task": "Find X"}, call_id="1"),
            tool_call_response("delegate", {"agent": "mail", "task": "find  x"}, call_id="2"),
            tool_call_response("delegate", {"agent": "mail", "task": "Find Y"}, call_id="3"),
            tool_call_response("delegate", {"agent": "mail", "task": "Find Z"}, call_id="4"),
            text_response("ok"),
        ],
        [source],
    )
    result = await main.run("go", output_type=str)
    contents = [t.content for t in result.trajectory]
    assert contents[0] == "a"
    assert "already given exactly this task" in json.loads(contents[1])["notice"]
    assert contents[2] == "b"
    assert "already run 2 times" in json.loads(contents[3])["notice"]
    assert not any(t.is_error for t in result.trajectory)


async def test_budget_state_is_per_run():
    sub, _ = sub_agent("mail", [text_response("a"), text_response("b")])
    source = DelegateToolSource([sub], max_calls_per_delegate=1)
    for answer in ("a", "b"):
        main, _ = parent(
            [
                tool_call_response("delegate", {"agent": "mail", "task": "same"}),
                text_response("ok"),
            ],
            [source],
        )
        result = await main.run("go", output_type=str)
        assert result.trajectory[0].content == answer


async def test_failures_are_sanitized():
    class Broken(Delegate):
        name = "broken"
        description = "always fails"

        async def run(self, task, *, context, arguments, on_event):
            raise RuntimeError("api key sk-123 rejected")

    main, _ = parent(
        [tool_call_response("delegate", {"agent": "broken", "task": "t"}), text_response("x")],
        [DelegateToolSource([Broken()])],
    )
    result = await main.run("go", output_type=str)
    record = result.trajectory[0]
    assert record.is_error
    assert "sk-123" not in record.content
    assert "couldn't complete" in record.content


async def test_unknown_agent_is_a_model_visible_error():
    sub, _ = sub_agent("mail", [])
    main, _ = parent(
        [tool_call_response("delegate", {"agent": "nope", "task": "t"}), text_response("x")],
        [DelegateToolSource([sub])],
    )
    result = await main.run("go", output_type=str)
    assert result.trajectory[0].is_error
    assert "Choose one of: mail" in result.trajectory[0].content


async def test_custom_delegate_gets_arguments_and_context():
    seen = {}

    class Custom(Delegate):
        name = "custom"
        description = "custom"

        async def run(self, task, *, context, arguments, on_event):
            seen.update(task=task, context=context, mode=arguments.get("mode"))
            return ToolResult(content="ok", usage=Usage(input_tokens=7, requests=1))

    source = DelegateToolSource([Custom()], extra_properties={"mode": {"type": "string"}})
    main, _ = parent(
        [
            tool_call_response("delegate", {"agent": "custom", "task": "t", "mode": "write"}),
            text_response("x"),
        ],
        [source],
    )
    result = await main.run("go", context={"user": "u1"}, output_type=str)
    assert seen == {"task": "t", "context": {"user": "u1"}, "mode": "write"}
    assert result.usage.input_tokens == 7 + 20


async def test_from_directory_shares_the_workspace(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "notes.md").write_text(
        "---\nname: notes\ndescription: Keeps notes.\nmodel: fake:m\n---\nYou keep notes.\n"
    )
    ws = LocalWorkspace(tmp_path / "ws")
    source = DelegateToolSource.from_directory(agents_dir, agent_kwargs={"workspace": ws})
    [delegate] = source.delegates.values()
    assert isinstance(delegate, AgentDelegate)
    assert delegate.agent.workspace is ws
    assert delegate.description == "Keeps notes."


async def test_tools_can_emit_events_while_running():
    @tool
    async def long_job() -> str:
        call = current_tool_call()
        assert call is not None
        assert call.tool_name == "long_job"
        call.emit(StepEvent(id=f"{call.step_id}.p1", title="Halfway", status="running"))
        await asyncio.sleep(0)
        return "ok"

    main, _ = parent(
        [tool_call_response("long_job", {}), text_response("x")],
        [long_job],
    )
    events = await collect(main.stream("go"))
    titles = [e.title for e in events if isinstance(e, StepEvent)]
    # running step, the live progress step, then the done step
    assert titles == ["Long job", "Halfway", "Long job"]
    assert current_tool_call() is None


async def test_as_tool_forwards_sub_steps_too():
    @tool
    def lookup() -> str:
        return "x"

    sub, _ = sub_agent(
        "helper", [tool_call_response("lookup", {}), text_response("found")], tools=[lookup]
    )
    main, _ = parent(
        [tool_call_response("helper", {"task": "t"}), text_response("ok")],
        [sub.as_tool()],
        describe_step=None,
    )
    events = await collect(main.stream("go"))
    steps = [e for e in events if isinstance(e, StepEvent)]
    assert steps
    assert all(s.agent == "helper" for s in steps)
    assert all(s.id.startswith("s1.") for s in steps)
