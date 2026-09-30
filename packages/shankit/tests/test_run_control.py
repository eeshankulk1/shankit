"""RunControl: cancel, messages, budgets, refusals, checkpoints, resume."""

from __future__ import annotations

import pytest
from conftest import FakeModel, text_response, tool_call_response, usage
from shankit import (
    Agent,
    Budget,
    Message,
    RunControl,
    StepEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    tool,
)
from shankit.models.base import ModelResponse

ran: list[str] = []


@tool
def lookup(q: str) -> str:
    """Look something up."""
    ran.append(q)
    return f"found {q}"


def make(responses, **kwargs) -> tuple[Agent, FakeModel]:
    fake = FakeModel(responses)
    return Agent(name="a", model="fake", model_client=fake, tools=[lookup], **kwargs), fake


def last_user_texts(request) -> list[str]:
    return [b.text for b in request.messages[-1].content if isinstance(b, TextBlock)]


async def test_cancel_before_the_first_pass_calls_no_model():
    agent, fake = make([])
    control = RunControl()
    control.cancel()
    result = await agent.run("hi", output_type=str, control=control)
    assert result.stopped == "cancelled"
    assert result.output is None
    assert fake.requests == []


async def test_cancel_after_the_answer_skips_its_tools_and_keeps_the_transcript_valid():
    ran.clear()
    control = RunControl()
    agent, _fake = make([tool_call_response("lookup", {"q": "x"})])

    def on_event(event):
        if event.type == "usage":
            control.cancel()

    result = await agent.run("hi", output_type=str, control=control, on_event=on_event)
    assert result.stopped == "cancelled"
    assert ran == []
    last = result.messages[-1]
    assert last.role == "user"
    assert isinstance(last.content[0], ToolResultBlock)
    assert last.content[0].content == "Not run: stopped."


async def test_a_message_rides_the_next_tool_results():
    control = RunControl()
    agent, fake = make([tool_call_response("lookup", {"q": "x"}), text_response("ok")])

    def on_event(event):
        if isinstance(event, StepEvent) and event.status == "running":
            control.send("Make it 6:30 instead.")

    await agent.run("hi", output_type=str, control=control, on_event=on_event)
    second = fake.requests[1]
    assert isinstance(second.messages[-1].content[0], ToolResultBlock)
    assert last_user_texts(second) == ["Make it 6:30 instead."]


async def test_a_message_during_the_final_answer_gets_one_more_pass():
    control = RunControl()
    agent, fake = make([text_response("done"), text_response("switched to 6:30")])

    def on_event(event):
        if event.type == "usage" and len(fake.requests) == 1:
            control.send("Make it 6:30.")

    result = await agent.run("hi", output_type=str, control=control, on_event=on_event)
    assert result.output == "switched to 6:30"
    assert fake.requests[1].messages[-1].role == "user"
    assert last_user_texts(fake.requests[1]) == ["Make it 6:30."]


async def test_a_pass_budget_winds_down_with_tools_off():
    agent, fake = make(
        [
            tool_call_response("lookup", {"q": "a"}, call_id="t1"),
            tool_call_response("lookup", {"q": "b"}, call_id="t2"),
            text_response("Found a and b; c is unfinished."),
        ]
    )
    control = RunControl(budget=Budget(max_passes=3))
    result = await agent.run("find a, b, c", output_type=str, control=control)
    assert result.stopped == "budget"
    assert result.output == "Found a and b; c is unfinished."
    last = fake.requests[-1]
    assert last.tool_choice == "none"
    # Same tool list on the wind-down pass: the prompt prefix is unchanged.
    assert [t.name for t in last.tools] == [t.name for t in fake.requests[0].tools]
    assert "Budget reached" in last_user_texts(last)[-1]


async def test_a_cost_budget_is_priced_from_the_runs_usage():
    agent, fake = make([tool_call_response("lookup", {"q": "a"}), text_response("partial report")])
    budget = Budget(max_cost=0.5, cost=lambda u: u.input_tokens * 0.1)  # 10 tokens -> 1.0
    result = await agent.run("go", output_type=str, control=RunControl(budget=budget))
    assert result.stopped == "budget"
    assert fake.requests[1].tool_choice == "none"


async def test_a_tool_call_in_the_wind_down_pass_is_closed_not_run():
    ran.clear()
    agent, _ = make([tool_call_response("lookup", {"q": "late"})])
    result = await agent.run("go", output_type=str, control=RunControl(budget=Budget(max_passes=1)))
    assert result.stopped == "budget"
    assert ran == []
    assert result.messages[-1].content[0].content == "Not run: out of budget."


async def test_refuse_vetoes_at_call_time_and_keeps_the_tool_list():
    ran.clear()
    agent, fake = make([tool_call_response("lookup", {"q": "x"}), text_response("ok")])
    control = RunControl(refuse=lambda name, args: "Not for you." if name == "lookup" else None)
    await agent.run("go", output_type=str, control=control)
    assert ran == []
    result_block = fake.requests[1].messages[-1].content[0]
    assert result_block.is_error
    assert result_block.content == "Not for you."
    assert [t.name for t in fake.requests[0].tools] == ["lookup"]


async def test_checkpoints_then_resume_closes_the_call_in_flight():
    saved: list[list[Message]] = []

    async def checkpoint(transcript):
        saved.append(transcript)

    agent, _ = make([tool_call_response("lookup", {"q": "a"}, call_id="t1"), text_response("a")])
    await agent.run("go", output_type=str, control=RunControl(checkpoint=checkpoint))
    # One checkpoint with the call in flight, one after its result.
    assert saved[0][-1].role == "assistant"
    assert saved[1][-1].role == "user"

    # A crash while t1 ran: resume from the in-flight checkpoint.
    resumed, fake = make(
        [tool_call_response("lookup", {"q": "b"}, call_id="t2"), text_response("a and b")]
    )
    events = []
    result = await resumed.resume(saved[0], output_type=str, on_event=events.append)
    assert result.output == "a and b"
    first = fake.requests[0]
    interrupted = first.messages[-1].content[0]
    assert interrupted.is_error
    assert "Interrupted by a restart" in interrupted.content
    assert first.messages[0].content[0].text == "go"
    # Step ids carry on from the original run (s1 was t1).
    assert {e.id for e in events if isinstance(e, StepEvent)} == {"s2"}
    assert result.messages[0].content[0].text == "go"


async def test_resuming_a_finished_transcript_calls_no_model():
    agent, fake = make([])
    transcript = [
        Message(role="user", content=[TextBlock(text="go")]),
        Message(role="assistant", content=[TextBlock(text="all done")]),
    ]
    result = await agent.resume(transcript, output_type=str)
    assert result.output == "all done"
    assert fake.requests == []


async def test_resume_counts_prior_passes_against_the_budget():
    transcript = [
        Message(role="user", content=[TextBlock(text="go")]),
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="lookup", input={"q": "a"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="t1", content="found a")]),
    ]
    agent, fake = make([text_response("report")])
    result = await agent.resume(
        transcript, output_type=str, control=RunControl(budget=Budget(max_passes=2))
    )
    assert result.stopped == "budget"
    assert fake.requests[0].tool_choice == "none"


async def test_copy_changes_only_what_it_names():
    agent, _ = make([], timeout_s=270.0, max_iterations=40)
    copy = agent.copy(timeout_s=None, max_iterations=60)
    assert copy.timeout_s is None
    assert copy.max_iterations == 60
    assert agent.timeout_s == 270.0
    assert agent.max_iterations == 40
    assert copy.tool_source is agent.tool_source
    assert copy.instructions is agent.instructions
    with pytest.raises(TypeError, match="bogus"):
        agent.copy(bogus=1)


async def test_usage_of_a_cancelled_run_is_kept():
    control = RunControl()
    agent, _ = make(
        [
            ModelResponse(
                content=[ToolUseBlock(id="t1", name="lookup", input={"q": "a"})],
                stop_reason="tool_use",
                usage=usage(100, 20),
            )
        ]
    )

    def on_event(event):
        if event.type == "usage":
            control.cancel()

    result = await agent.run("go", output_type=str, control=control, on_event=on_event)
    assert result.usage.input_tokens == 100
    assert isinstance(result.usage, Usage)


def test_child_control_is_cancelled_with_its_parent():
    parent = RunControl()
    child = RunControl(parent=parent)
    assert not child.cancelled
    parent.cancel()
    assert child.cancelled


async def test_resume_of_a_structured_run_that_recorded_its_result_makes_no_call():
    from pydantic import BaseModel

    class Out(BaseModel):
        n: int

    agent, fake = make([tool_call_response("final_result", {"n": 3})])
    first = await agent.run("hi", output_type=Out)
    assert first.output == Out(n=3)
    fake.requests.clear()
    again = await agent.resume(first.messages, output_type=Out)
    assert again.output == Out(n=3)
    assert fake.requests == []
