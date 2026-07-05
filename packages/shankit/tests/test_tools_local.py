from typing import Optional

import pytest
from shankit import FunctionToolSource, ToolError, ToolNotFoundError, ToolResult, tool
from shankit.events import Source


@tool
def add(a: int, b: int = 1) -> str:
    """Add two integers."""
    return str(a + b)


@tool
async def whoami(ctx) -> str:
    """Report the acting user."""
    return f"user={ctx['user_id']}"


@tool(name="renamed", description="Custom description.")
def original_name(x: str) -> str:
    return x


def test_schema_from_signature():
    d = add.definition
    assert d.name == "add"
    assert d.description == "Add two integers."
    assert d.input_schema["type"] == "object"
    assert d.input_schema["properties"]["a"]["type"] == "integer"
    assert d.input_schema["required"] == ["a"]  # b has a default
    assert d.input_schema["properties"]["b"]["default"] == 1


def test_ctx_param_excluded_from_schema():
    d = whoami.definition
    assert "ctx" not in d.input_schema.get("properties", {})


def test_decorator_with_options():
    assert original_name.definition.name == "renamed"
    assert original_name.definition.description == "Custom description."
    # decorated function still directly callable
    assert original_name("hi") == "hi"


async def test_execute_sync_and_async():
    source = FunctionToolSource([add, whoami])
    result = await source.execute("add", {"a": 2, "b": 3})
    assert result.content == "5"
    result = await source.execute("whoami", {}, context={"user_id": "u1"})
    assert result.content == "user=u1"


async def test_argument_validation_raises_tool_error():
    source = FunctionToolSource([add])
    with pytest.raises(ToolError):
        await source.execute("add", {"a": "not an int at all"})


async def test_unknown_tool():
    source = FunctionToolSource([add])
    with pytest.raises(ToolNotFoundError):
        await source.execute("nope", {})


async def test_return_normalization():
    @tool
    def gives_dict() -> dict:
        return {"k": [1, 2]}

    @tool
    def gives_none() -> None:
        return None

    @tool
    def gives_result() -> ToolResult:
        return ToolResult(content="x", sources=[Source(title="doc")])

    source = FunctionToolSource([gives_dict, gives_none, gives_result])
    assert (await source.execute("gives_dict", {})).content == '{"k": [1, 2]}'
    assert (await source.execute("gives_none", {})).content == ""
    r = await source.execute("gives_result", {})
    assert r.sources[0].title == "doc"


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        FunctionToolSource([add, add])


def test_var_args_rejected():
    with pytest.raises(TypeError, match="args"):

        @tool
        def bad(**kwargs) -> str:
            return ""


def test_optional_annotation():
    @tool
    def opt(q: Optional[str] = None) -> str:
        return q or ""

    props = opt.definition.input_schema["properties"]
    assert "q" in props
