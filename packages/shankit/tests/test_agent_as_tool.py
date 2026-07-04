import pytest
from conftest import FakeModel, final_result_response, text_response, tool_call_response
from pydantic import BaseModel
from shankit import Agent, ShankitError, ToolResult, tool
from shankit.events import Source


class Findings(BaseModel):
    summary: str


def make_sub(responses, **kwargs):
    return Agent(
        name="researcher",
        model="fake",
        model_client=FakeModel(responses),
        description="Research things.",
        **kwargs,
    )


async def test_parent_delegates_to_sub_agent(make_agent):
    @tool
    def lookup() -> ToolResult:
        return ToolResult(content="raw", sources=[Source(title="wiki")])

    sub = make_sub([tool_call_response("lookup", {}), text_response("sub-answer")], tools=[lookup])
    parent, fake = make_agent(
        [
            tool_call_response("researcher", {"task": "find facts"}),
            text_response("parent-answer"),
        ],
        tools=[sub.as_tool()],
    )
    result = await parent.run("go", output_type=str)

    assert result.text == "parent-answer"
    # sub-agent transcript flowed back as the tool result
    assert result.trajectory[0].tool == "researcher"
    assert result.trajectory[0].content == "sub-answer"
    # token accounting: parent 2 calls + sub 2 calls
    assert result.usage.requests == 4
    # sub-agent sources propagate to the parent run
    assert [s.title for s in result.sources] == ["wiki"]


async def test_sub_agent_failure_sanitized(make_agent):
    class ExplodingModel(FakeModel):
        async def complete(self, request):
            raise RuntimeError("provider blew up: secret_key=abc")

    sub = Agent(name="fragile", model="fake", model_client=ExplodingModel([]))
    parent, fake = make_agent(
        [tool_call_response("fragile", {"task": "x"}), text_response("recovered")],
        tools=[sub.as_tool()],
    )
    result = await parent.run("go", output_type=str)
    assert result.trajectory[0].is_error
    assert "secret_key" not in result.trajectory[0].content
    assert "could not complete" in result.trajectory[0].content


async def test_as_tool_exposes_task_schema():
    sub = make_sub([])
    source = sub.as_tool()
    defs = await source.list_tools()
    assert defs[0].name == "researcher"
    assert defs[0].description == "Research things."
    assert defs[0].input_schema["required"] == ["task"]


async def test_as_tool_structured_output():
    sub = make_sub([final_result_response({"summary": "s"})], output_type=Findings)
    source = sub.as_tool(output="structured")
    result = await source.execute("researcher", {"task": "x"})
    assert result.content == '{"summary":"s"}'


async def test_as_tool_structured_requires_schema():
    sub = make_sub([])
    with pytest.raises(ShankitError, match="output schema"):
        sub.as_tool(output="structured")


async def test_as_tool_name_sanitized():
    sub = Agent(name="my agent!", model="fake", model_client=FakeModel([]))
    defs = await sub.as_tool().list_tools()
    assert defs[0].name == "my_agent_"


async def test_empty_task_is_tool_error():
    sub = make_sub([])
    source = sub.as_tool()
    from shankit import ToolError

    with pytest.raises(ToolError):
        await source.execute("researcher", {"task": "  "})
