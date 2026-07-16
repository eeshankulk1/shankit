from conftest import FakeModel, text_response, tool_call_response
from shankit import (
    Agent,
    ArtifactEvent,
    DoneEvent,
    ErrorEvent,
    SourceEvent,
    StepEvent,
    TextDeltaEvent,
    ToolResult,
    Usage,
    UsageEvent,
    tool,
)
from shankit.events import Artifact, Source
from shankit.observe import StepInfo


async def collect(stream):
    return [event async for event in stream]


async def test_stream_text_and_done(make_agent):
    agent, _ = make_agent([text_response("hello world")])
    events = await collect(agent.stream("hi"))

    deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert "".join(d.text for d in deltas) == "hello world"
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].text == "hello world"
    assert events[-1].output is None  # streaming mode: no structured deliverable
    assert any(isinstance(e, UsageEvent) for e in events)


async def test_interim_text_joins_into_done_text(make_agent):
    """A pass that says something before calling tools keeps that text in the
    final result — the streamed transcript and the persisted text agree."""

    @tool
    def check() -> str:
        return "ok"

    agent, _ = make_agent(
        [
            tool_call_response("check", {}, text="Let me check that."),
            text_response("The answer is 5."),
        ],
        tools=[check],
    )
    events = await collect(agent.stream("go"))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.text == "Let me check that.\n\nThe answer is 5."
    streamed = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert streamed == "Let me check that." + "The answer is 5."


async def test_usage_events_sum_to_done_usage(make_agent):
    """Sub-agent usage arrives through the tool seam; it must be part of the
    stream too, so summing UsageEvents always matches DoneEvent.usage."""
    sub = Agent(name="helper", model="fake", model_client=FakeModel([text_response("sub answer")]))
    agent, _ = make_agent(
        [tool_call_response("helper", {"task": "t"}), text_response("done")],
        tools=[sub.as_tool()],
    )
    events = await collect(agent.stream("go"))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    streamed = Usage()
    for event in events:
        if isinstance(event, UsageEvent):
            streamed.add(event.usage)
    assert streamed == done.usage
    assert streamed.requests == 3  # two parent passes + one sub-agent pass


async def test_truncated_pass_marks_done_event(make_agent):
    """Truncation is sticky: a cut-off intermediate pass taints the run even
    when the final pass ends cleanly."""

    @tool
    def check() -> str:
        return "ok"

    first = tool_call_response("check", {})
    first.stop_reason = "max_tokens"
    agent, _ = make_agent([first, text_response("fine")], tools=[check])
    events = await collect(agent.stream("go"))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.truncated


async def test_stream_narrates_steps(make_agent):
    @tool
    def search(q: str) -> str:
        return "found it"

    def describe(name, args, ctx):
        return StepInfo(title=f"Searched for {args['q']}", phase="research")

    agent, _ = make_agent(
        [tool_call_response("search", {"q": "cats"}), text_response("done")],
        tools=[search],
        describe_step=describe,
    )
    events = await collect(agent.stream("go"))
    steps = [e for e in events if isinstance(e, StepEvent)]
    assert [s.status for s in steps] == ["running", "done"]
    assert steps[0].title == "Searched for cats"
    assert steps[0].phase == "research"
    assert steps[0].id == steps[1].id


async def test_describer_none_hides_all_steps(make_agent):
    @tool
    def quiet() -> str:
        return "ok"

    agent, _ = make_agent(
        [tool_call_response("quiet", {}), text_response("done")],
        tools=[quiet],
        describe_step=None,
    )
    events = await collect(agent.stream("go"))
    assert not [e for e in events if isinstance(e, StepEvent)]


async def test_describer_can_hide_single_call(make_agent):
    @tool
    def a() -> str:
        return "1"

    def describe(name, args, ctx):
        return None  # hidden

    agent, _ = make_agent(
        [tool_call_response("a", {}), text_response("done")], tools=[a], describe_step=describe
    )
    events = await collect(agent.stream("go"))
    assert not [e for e in events if isinstance(e, StepEvent)]


async def test_failed_step_status_error(make_agent):
    @tool
    def broken() -> str:
        raise RuntimeError("boom")

    agent, _ = make_agent(
        [tool_call_response("broken", {}), text_response("recovered")], tools=[broken]
    )
    events = await collect(agent.stream("go"))
    steps = [e for e in events if isinstance(e, StepEvent)]
    assert steps[-1].status == "error"


async def test_sources_surface_as_events(make_agent):
    @tool
    def cite() -> ToolResult:
        return ToolResult(content="text", sources=[Source(title="Doc", url="https://x")])

    agent, _ = make_agent([tool_call_response("cite", {}), text_response("done")], tools=[cite])
    events = await collect(agent.stream("go"))
    sources = [e for e in events if isinstance(e, SourceEvent)]
    assert sources[0].source.url == "https://x"


async def test_artifacts_surface_as_events(make_agent):
    @tool
    def fetch() -> ToolResult:
        return ToolResult(
            content="text",
            artifacts=[Artifact(type="email_card", data={"subject": "Hi", "sender": "a@b.c"})],
        )

    agent, _ = make_agent([tool_call_response("fetch", {}), text_response("done")], tools=[fetch])
    events = await collect(agent.stream("go"))
    artifacts = [e for e in events if isinstance(e, ArtifactEvent)]
    assert artifacts[0].artifact.type == "email_card"
    assert artifacts[0].artifact.data == {"subject": "Hi", "sender": "a@b.c"}


async def test_framework_error_becomes_error_event(make_agent):
    @tool
    def spin() -> str:
        return "again"

    responses = [tool_call_response("spin", {}, call_id=f"t{i}") for i in range(2)]
    agent, _ = make_agent(responses, tools=[spin], max_iterations=2)
    events = await collect(agent.stream("go"))
    assert isinstance(events[-1], ErrorEvent)
    assert "max_iterations" in events[-1].message
    assert events[-1].code == "max_iterations"
    assert not events[-1].retryable


async def test_cancelled_stream_cancels_inflight_tools(make_agent):
    """Abandoning a stream mid-turn must cancel in-flight tool tasks, not
    orphan them to finish (and side-effect) in the background."""
    import asyncio

    state = {"started": False, "cancelled": False, "finished": False}

    @tool
    async def slow() -> str:
        state["started"] = True
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        state["finished"] = True
        return "done"

    agent, _ = make_agent(
        [tool_call_response("slow", {}), text_response("never reached")], tools=[slow]
    )

    async def consume():
        async for _ in agent.stream("go"):
            pass

    task = asyncio.create_task(consume())
    while not state["started"]:
        await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.01)
    assert state["cancelled"]
    assert not state["finished"]
