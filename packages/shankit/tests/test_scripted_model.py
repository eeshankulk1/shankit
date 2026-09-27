"""ScriptedModel: the public fake model client for testing code built on
shankit. Every scripted behavior is exercised through a real Agent, in the
run mode that exposes it."""

import pytest
from pydantic import BaseModel
from shankit import (
    Agent,
    DoneEvent,
    ErrorEvent,
    ModelError,
    ReasoningBlock,
    TextBlock,
    TextDeltaEvent,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UsageEvent,
    register_provider,
    tool,
)
from shankit.agent import OUTPUT_TOOL_NAME
from shankit.models.base import ForcedTool, ModelResponse, model_error_for_status
from shankit.testing import ScriptedModel, ScriptFailure, ToolCall, Turn


@tool
def order_status(order_id: int) -> str:
    """Look up an order's shipping status."""
    return f"order {order_id}: shipped"


class Answer(BaseModel):
    status: str
    days: int


def make(script, **kwargs) -> tuple[Agent, ScriptedModel]:
    model = ScriptedModel(script)
    agent = Agent(name="support", model="scripted-1", model_client=model, **kwargs)
    return agent, model


async def collect(stream):
    return [event async for event in stream]


# ------------------------------------------------------------------ text


async def test_text_streams_as_deltas():
    agent, model = make(["Your order is on its way."])
    events = await collect(agent.stream("Where is my order?"))

    deltas = [e.text for e in events if isinstance(e, TextDeltaEvent)]
    assert len(deltas) > 1  # word-sized chunks, like a real stream
    assert "".join(deltas) == "Your order is on its way."
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.text == "Your order is on its way."
    model.assert_exhausted()


async def test_text_through_complete():
    agent, _ = make([Turn("All good.")])
    result = await agent.run("status?", output_type=str)
    assert result.output == "All good."


# ------------------------------------------------------------- tool calls


async def test_tool_call_round_trip():
    agent, model = make(
        [
            Turn("Let me check.", tool_calls=[ToolCall("order_status", {"order_id": 7})]),
            "Order 7 has shipped.",
        ],
        tools=[order_status],
    )
    result = await agent.run("Where is order 7?", output_type=str)

    assert result.output == "Order 7 has shipped."
    assert [(r.tool, r.arguments, r.content) for r in result.trajectory] == [
        ("order_status", {"order_id": 7}, "order 7: shipped")
    ]
    # The second request carries the call and its real result back.
    call, returned = model.requests[1].messages[-2:]
    assert call.content[-1] == ToolUseBlock(id="call_1", name="order_status", input={"order_id": 7})
    assert returned.content == [ToolResultBlock(tool_use_id="call_1", content="order 7: shipped")]
    model.assert_exhausted()


async def test_tool_call_ids_are_sequential_unless_given():
    agent, model = make(
        [
            Turn(tool_calls=[ToolCall("order_status", {"order_id": 1})]),
            Turn(
                tool_calls=[
                    ToolCall("order_status", {"order_id": 2}),
                    ToolCall("order_status", {"order_id": 3}, id="mine"),
                ]
            ),
            "done",
        ],
        tools=[order_status],
    )
    await agent.run("go", output_type=str)
    ids = [
        b.id for m in model.requests[-1].messages for b in m.content if isinstance(b, ToolUseBlock)
    ]
    assert ids == ["call_1", "call_2", "mine"]


async def test_tool_call_turn_streams_its_text_and_steps():
    agent, _ = make(
        [
            Turn("Checking.", tool_calls=[ToolCall("order_status", {"order_id": 7})]),
            "Shipped.",
        ],
        tools=[order_status],
    )
    events = await collect(agent.stream("go"))
    streamed = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert streamed == "Checking." + "Shipped."
    assert isinstance(events[-1], DoneEvent)


# ------------------------------------------------------ structured output


async def test_structured_output_from_a_model_instance():
    agent, model = make([Turn(output=Answer(status="shipped", days=2))], output_type=Answer)
    result = await agent.run("status?")

    assert result.output == Answer(status="shipped", days=2)
    # No real tools: the loop forces the output tool, and the request shows it.
    assert model.requests[0].tool_choice == ForcedTool(name=OUTPUT_TOOL_NAME)


async def test_structured_output_accepts_a_dict_and_wraps_non_object_types():
    agent, _ = make([Turn(output={"status": "late", "days": 5})], output_type=Answer)
    assert (await agent.run("status?")).output == Answer(status="late", days=5)

    agent, model = make([Turn(output=[3, 1, 2])], output_type=list[int])
    assert (await agent.run("ids?")).output == [3, 1, 2]
    call = model.requests[0].tools[-1]
    assert call.name == OUTPUT_TOOL_NAME
    assert call.input_schema["required"] == ["value"]


async def test_structured_output_after_a_tool_call():
    agent, _ = make(
        [
            Turn(tool_calls=[ToolCall("order_status", {"order_id": 7})]),
            Turn(output=Answer(status="shipped", days=0)),
        ],
        tools=[order_status],
        output_type=Answer,
    )
    result = await agent.run("status?")
    assert result.output.status == "shipped"
    assert [r.tool for r in result.trajectory] == ["order_status"]


async def test_output_without_an_output_tool_fails_the_script():
    agent, _ = make([Turn(output={"status": "x", "days": 1})])
    with pytest.raises(ScriptFailure, match="offers no `final_result` tool"):
        await agent.run("status?", output_type=str)


# -------------------------------------------------------------- reasoning


async def test_reasoning_round_trips_into_the_next_request():
    agent, model = make(
        [
            Turn(
                reasoning="The user wants order 7.",
                tool_calls=[ToolCall("order_status", {"order_id": 7})],
            ),
            "Shipped.",
        ],
        tools=[order_status],
        reasoning="low",
    )
    result = await agent.run("Where is order 7?", output_type=str)

    assert model.requests[0].reasoning is not None
    assert model.requests[0].reasoning.effort == "low"
    assistant = model.requests[1].messages[1]
    assert assistant.role == "assistant"
    reasoning = assistant.content[0]
    assert isinstance(reasoning, ReasoningBlock)
    assert reasoning.provider == "scripted"
    assert reasoning.text == "The user wants order 7."
    # ... and it is in the run's own transcript.
    assert any(isinstance(b, ReasoningBlock) for m in result.messages for b in m.content)


async def test_reasoning_block_is_sent_as_is():
    block = ReasoningBlock(provider="anthropic", data={"type": "thinking", "signature": "sig"})
    agent, _ = make([Turn("Hi.", reasoning=block)])
    result = await agent.run("hi", output_type=str)
    assert result.messages[1].content[0] == block


# ------------------------------------------------------------------ usage


async def test_usage_accounting():
    agent, _ = make(
        [
            Turn(
                tool_calls=[ToolCall("order_status", {"order_id": 7})],
                usage=Usage(input_tokens=100, output_tokens=20, cache_read_tokens=50, requests=1),
            ),
            "Shipped.",  # default usage: one request, no tokens
        ],
        tools=[order_status],
    )
    events = await collect(agent.stream("go"))

    usages = [e.usage for e in events if isinstance(e, UsageEvent)]
    assert usages == [
        Usage(input_tokens=100, output_tokens=20, cache_read_tokens=50, requests=1),
        Usage(requests=1),
    ]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.usage == Usage(input_tokens=100, output_tokens=20, cache_read_tokens=50, requests=2)


async def test_scripted_max_tokens_marks_the_run_truncated():
    agent, _ = make([Turn("Cut off mid", stop_reason="max_tokens")])
    result = await agent.run("go", output_type=str)
    assert result.truncated


# ----------------------------------------------------------------- errors


async def test_scripted_provider_error_then_retry_succeeds():
    rate_limited = model_error_for_status("anthropic", 429)
    agent, model = make([rate_limited, "Recovered."])

    with pytest.raises(ModelError) as excinfo:
        await agent.run("go", output_type=str)
    assert excinfo.value is rate_limited
    assert excinfo.value.retryable

    # The caller's retry is the script's next turn.
    result = await agent.run("go", output_type=str)
    assert result.output == "Recovered."
    model.assert_exhausted()


async def test_scripted_error_becomes_an_error_event_in_stream():
    agent, _ = make([Turn(error=model_error_for_status("anthropic", 529))])
    events = await collect(agent.stream("go"))
    error = events[-1]
    assert isinstance(error, ErrorEvent)
    assert error.code == "model_error"
    assert error.retryable


async def test_error_with_text_fails_the_stream_midway():
    agent, _ = make([Turn("Partial answer", error=model_error_for_status("anthropic", 529))])
    events = await collect(agent.stream("go"))
    assert "".join(e.text for e in events if isinstance(e, TextDeltaEvent)) == "Partial answer"
    assert isinstance(events[-1], ErrorEvent)


async def test_non_model_error_goes_through_the_loops_fallback():
    agent, _ = make([RuntimeError("socket exploded")])
    with pytest.raises(ModelError) as excinfo:
        await agent.run("go", output_type=str)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert not excinfo.value.retryable


# -------------------------------------------------------- script exhaustion


async def test_exhausted_script_fails_loudly_in_run():
    agent, _ = make(["only turn"])
    await agent.run("first", output_type=str)
    with pytest.raises(ScriptFailure, match=r"script exhausted: requests\[1\] has no turn") as exc:
        await agent.run("second", output_type=str)
    assert "'second'" in str(exc.value)  # the unscripted request is summarized


async def test_exhausted_script_is_not_swallowed_by_stream():
    """The loop turns client exceptions into a calm error event; a script
    failure must escape that and fail the test instead."""
    agent, _ = make([])
    with pytest.raises(ScriptFailure):
        await collect(agent.stream("go"))


async def test_exhausted_script_escapes_sub_agent_sanitizing():
    sub, _ = make([])
    sub.name = "researcher"
    parent, _ = make(
        [Turn(tool_calls=[ToolCall("researcher", {"task": "look"})]), "done"],
        tools=[sub.as_tool()],
    )
    with pytest.raises(ScriptFailure):
        await parent.run("go", output_type=str)


def test_assert_exhausted_names_the_unplayed_turns():
    model = ScriptedModel(["a", "b"])
    with pytest.raises(AssertionError, match=r"played 0 of 2 .* script\[0:\]"):
        model.assert_exhausted()


def test_invalid_script_items_are_rejected_up_front():
    with pytest.raises(TypeError, match=r"script\[1\] is a int"):
        ScriptedModel(["fine", 42])  # type: ignore[list-item]


# ------------------------------------------------------ request assertions


async def test_requests_record_what_was_sent():
    agent, model = make(
        ["Shipped."],
        instructions="You are a support agent.",
        tools=[order_status],
        max_tokens=512,
        temperature=0.2,
    )
    await agent.run("Where is order 7?", output_type=str)

    [request] = model.requests
    assert request.model == "scripted-1"
    assert request.system == "You are a support agent."
    assert [t.name for t in request.tools] == ["order_status"]
    assert request.tool_choice == "auto"
    assert (request.max_tokens, request.temperature) == (512, 0.2)
    assert request.messages[0].content == [TextBlock(text="Where is order 7?")]


async def test_requests_are_snapshots():
    agent, model = make(
        [Turn(tool_calls=[ToolCall("order_status", {"order_id": 7})]), "Shipped."],
        tools=[order_status],
    )
    await agent.run("go", output_type=str)
    # The loop kept appending to its history; the first request did not grow.
    assert len(model.requests[0].messages) == 1
    assert len(model.requests[1].messages) == 3


async def test_expect_passes_and_fails_with_a_request_summary():
    def saw_the_tool_result(request):
        assert request.messages[-1].content[0].content == "order 7: shipped"

    agent, _ = make(
        [
            Turn(
                tool_calls=[ToolCall("order_status", {"order_id": 7})],
                expect=lambda r: r.messages[0].content[0].text == "Where is order 7?",
            ),
            Turn("Shipped.", expect=saw_the_tool_result),
        ],
        tools=[order_status],
    )
    assert (await agent.run("Where is order 7?", output_type=str)).output == "Shipped."

    agent, _ = make([Turn("x", expect=lambda r: "refund" in (r.system or ""))])
    with pytest.raises(ScriptFailure, match=r"script\[0\] expect returned False") as exc:
        await agent.run("hi", output_type=str)
    assert "messages[0] user: text 'hi'" in str(exc.value)


async def test_failed_expect_assertion_is_not_swallowed_by_stream():
    def no_tools(request):
        assert request.tools == [], "the agent should offer no tools"

    agent, _ = make([Turn("x", expect=no_tools)], tools=[order_status])
    with pytest.raises(ScriptFailure, match="the agent should offer no tools") as exc:
        await collect(agent.stream("hi"))
    assert "tools=[order_status]" in str(exc.value)


# ---------------------------------------------------------- the seam itself


async def test_works_as_a_registered_provider():
    model = ScriptedModel(["From the registry."])
    register_provider("scripted-test", lambda: model)
    agent = Agent(name="support", model="scripted-test:any-model-id")

    result = await agent.run("hi", output_type=str)
    assert result.output == "From the registry."
    assert model.requests[0].model == "any-model-id"


async def test_model_response_items_and_extra_blocks_pass_through():
    raw = ModelResponse(content=[TextBlock(text="verbatim")], usage=Usage(requests=1))
    extra = ToolUseBlock(id="srv_1", name="order_status", input={"order_id": 9})
    agent, _ = make([Turn("Checking.", blocks=[extra]), raw], tools=[order_status])

    result = await agent.run("go", output_type=str)
    assert result.messages[1].content == [TextBlock(text="Checking."), extra]
    assert result.trajectory[0].arguments == {"order_id": 9}  # blocks are live content
    assert result.messages[3].content == [TextBlock(text="verbatim")]
    assert result.output == "verbatim"


def test_describe_request_handles_multimodal_tool_results() -> None:
    from shankit.messages import ImageBlock, Message, TextBlock, ToolResultBlock
    from shankit.models.base import ModelRequest
    from shankit.testing import _describe_request

    request = ModelRequest(
        model="scripted",
        messages=[
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="t1",
                        content=[
                            TextBlock(text="page text"),
                            ImageBlock(media_type="image/png", data="AAAA"),
                        ],
                    )
                ],
            )
        ],
    )
    summary = _describe_request(request)
    assert "tool_result t1 'page text\\n[image]'" in summary
