import pytest
from conftest import final_result_response, text_response, tool_call_response
from pydantic import BaseModel
from shankit import (
    MaxIterationsError,
    OutputValidationError,
    ShankitError,
    StepEvent,
    ToolError,
    tool,
)
from shankit.agent import OUTPUT_TOOL_NAME
from shankit.models.base import ForcedTool


class Answer(BaseModel):
    value: int
    note: str


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


async def test_structured_run_with_tool(make_agent):
    agent, fake = make_agent(
        [
            tool_call_response("add", {"a": 2, "b": 3}),
            final_result_response({"value": 5, "note": "sum"}),
        ],
        tools=[add],
        output_type=Answer,
    )
    result = await agent.run("what is 2+3?")

    assert result.output == Answer(value=5, note="sum")
    assert [r.tool for r in result.trajectory] == ["add"]
    assert result.trajectory[0].content == "5"
    assert result.usage.requests == 2
    assert result.usage.input_tokens == 20

    # the model saw both the real tool and the synthetic output tool
    first_request = fake.requests[0]
    names = [t.name for t in first_request.tools]
    assert names == ["add", OUTPUT_TOOL_NAME]
    # and the tool result came back on the transcript
    followup = fake.requests[1].messages[-1]
    assert followup.role == "user"
    assert followup.content[0].content == "5"


async def test_output_validation_retry(make_agent):
    agent, fake = make_agent(
        [
            final_result_response({"value": "not an int", "note": 1}),
            final_result_response({"value": 7, "note": "ok"}),
        ],
        output_type=Answer,
    )
    result = await agent.run("go")
    assert result.output.value == 7
    # the retry message described the validation failure
    retry_block = fake.requests[1].messages[-1].content[0]
    assert retry_block.is_error
    assert OUTPUT_TOOL_NAME in retry_block.content


async def test_output_validation_exhausts_retries(make_agent):
    bad = final_result_response({"value": "nope", "note": 1})
    agent, _ = make_agent([bad, bad.model_copy(deep=True)], output_type=Answer, output_retries=1)
    with pytest.raises(OutputValidationError):
        await agent.run("go")


async def test_nudge_when_model_stops_without_output(make_agent):
    agent, fake = make_agent(
        [
            text_response("I think the answer is five."),
            final_result_response({"value": 5, "note": "after nudge"}),
        ],
        output_type=Answer,
    )
    result = await agent.run("go")
    assert result.output.value == 5
    # second request nudged and forced the output tool
    assert isinstance(fake.requests[1].tool_choice, ForcedTool)
    assert fake.requests[1].tool_choice.name == OUTPUT_TOOL_NAME


async def test_str_output_type(make_agent):
    agent, _ = make_agent([text_response("plain answer")])
    result = await agent.run("go", output_type=str)
    assert result.output == "plain answer"
    assert result.text == "plain answer"


async def test_run_text_keeps_interim_passes(make_agent):
    agent, _ = make_agent(
        [
            tool_call_response("add", {"a": 2, "b": 3}, text="Let me add those."),
            text_response("The sum is 5."),
        ],
        tools=[add],
    )
    result = await agent.run("go", output_type=str)
    # text is the transcript; the str deliverable is the answer alone
    assert result.text == "Let me add those.\n\nThe sum is 5."
    assert result.output == "The sum is 5."


async def test_run_reports_truncation(make_agent):
    async def run_with_stop(stop_reason):
        response = text_response("an answer")
        response.stop_reason = stop_reason
        agent, _ = make_agent([response])
        return await agent.run("go", output_type=str)

    assert not (await run_with_stop("end_turn")).truncated
    assert (await run_with_stop("max_tokens")).truncated


async def test_run_without_schema_rejected(make_agent):
    agent, _ = make_agent([text_response("x")])
    with pytest.raises(ShankitError, match="output schema"):
        await agent.run("go")


async def test_default_schema_used(make_agent):
    agent, _ = make_agent([final_result_response({"value": 1, "note": "n"})], output_type=Answer)
    result = await agent.run("go")
    assert isinstance(result.output, Answer)


async def test_unknown_tool_is_error_result(make_agent):
    agent, _fake = make_agent(
        [
            tool_call_response("hallucinated", {}),
            final_result_response({"value": 0, "note": "recovered"}),
        ],
        tools=[add],
        output_type=Answer,
    )
    result = await agent.run("go")
    assert result.trajectory[0].is_error
    assert "hallucinated" in result.trajectory[0].content


async def test_tool_error_message_reaches_model(make_agent):
    @tool
    def fails() -> str:
        raise ToolError("The order id does not exist; ask the user for a valid one.")

    agent, fake = make_agent(
        [tool_call_response("fails", {}), final_result_response({"value": 0, "note": "n"})],
        tools=[fails],
        output_type=Answer,
    )
    await agent.run("go")
    block = fake.requests[1].messages[-1].content[0]
    assert block.is_error
    assert "order id" in block.content


async def test_unexpected_exception_sanitized(make_agent):
    @tool
    def explodes() -> str:
        raise RuntimeError("secret internals: db password is hunter2")

    agent, fake = make_agent(
        [tool_call_response("explodes", {}), final_result_response({"value": 0, "note": "n"})],
        tools=[explodes],
        output_type=Answer,
    )
    await agent.run("go")
    block = fake.requests[1].messages[-1].content[0]
    assert block.is_error
    assert "hunter2" not in block.content
    assert "failed unexpectedly" in block.content


async def test_max_iterations(make_agent):
    responses = [tool_call_response("add", {"a": 1, "b": 1}, call_id=f"t{i}") for i in range(3)]
    agent, _ = make_agent(responses, tools=[add], output_type=Answer, max_iterations=3)
    with pytest.raises(MaxIterationsError):
        await agent.run("go")


async def test_non_object_output_schema_wrapped(make_agent):
    agent, fake = make_agent([final_result_response({"value": ["a", "b"]})])
    result = await agent.run("go", output_type=list[str])
    assert result.output == ["a", "b"]
    schema = fake.requests[0].tools[-1].input_schema
    assert schema["properties"]["value"]["type"] == "array"


async def test_on_event_observes_structured_run(make_agent):
    agent, _ = make_agent(
        [
            tool_call_response("add", {"a": 1, "b": 2}),
            final_result_response({"value": 3, "note": "n"}),
        ],
        tools=[add],
        output_type=Answer,
    )
    seen: list = []
    await agent.run("go", on_event=seen.append)
    steps = [e for e in seen if isinstance(e, StepEvent)]
    assert [s.status for s in steps] == ["running", "done"]
    assert steps[0].title == "Add"


async def test_dynamic_instructions_receive_context(make_agent):
    agent, fake = make_agent(
        [final_result_response({"value": 1, "note": "n"})],
        instructions=lambda ctx: f"You serve {ctx['user']}.",
        output_type=Answer,
    )
    await agent.run("go", context={"user": "ada"})
    assert fake.requests[0].system == "You serve ada."


async def test_async_instructions(make_agent):
    async def instructions(ctx):
        return "async prose"

    agent, fake = make_agent(
        [final_result_response({"value": 1, "note": "n"})],
        instructions=instructions,
        output_type=Answer,
    )
    await agent.run("go")
    assert fake.requests[0].system == "async prose"


async def test_agent_object_in_tools_rejected(make_agent):
    from shankit import Agent

    sub = Agent(name="sub", model="anthropic:x")
    with pytest.raises(TypeError, match="as_tool"):
        Agent(name="parent", model="anthropic:x", tools=[sub])


def test_output_spec_hoists_defs_for_wrapped_schemas():
    """Non-object output types (list[Model], unions) get nested under
    properties.value — pydantic's "#/$defs/..." refs must stay resolvable
    from the schema root or the model receives dangling references."""
    from shankit.agent import _OutputSpec

    spec = _OutputSpec(list[Answer])
    schema = spec.tool_def.input_schema
    assert spec.wrapped
    assert "$defs" in schema
    assert "$defs" not in schema["properties"]["value"]
    ref = schema["properties"]["value"]["items"]["$ref"]
    assert ref.rsplit("/", 1)[-1] in schema["$defs"]
    validated = spec.validate({"value": [{"value": 1, "note": "n"}]})
    assert validated == [Answer(value=1, note="n")]


async def test_run_with_list_output_type(make_agent):
    agent, _fake = make_agent(
        [final_result_response({"value": [{"value": 2, "note": "x"}]})],
    )
    result = await agent.run("go", output_type=list[Answer])
    assert result.output == [Answer(value=2, note="x")]
