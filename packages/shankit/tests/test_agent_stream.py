from conftest import text_response, tool_call_response
from shankit import (
    DoneEvent,
    ErrorEvent,
    SourceEvent,
    StepEvent,
    TextDeltaEvent,
    ToolResult,
    UsageEvent,
    tool,
)
from shankit.events import Source
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


async def test_framework_error_becomes_error_event(make_agent):
    @tool
    def spin() -> str:
        return "again"

    responses = [tool_call_response("spin", {}, call_id=f"t{i}") for i in range(2)]
    agent, _ = make_agent(responses, tools=[spin], max_iterations=2)
    events = await collect(agent.stream("go"))
    assert isinstance(events[-1], ErrorEvent)
    assert "max_iterations" in events[-1].message
