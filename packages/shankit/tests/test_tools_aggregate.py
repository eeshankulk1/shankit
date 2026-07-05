import pytest
from shankit import CompositeToolSource, FunctionToolSource, ToolNotFoundError, tool


@tool
def alpha() -> str:
    return "a"


@tool
def beta() -> str:
    return "b"


async def test_composite_lists_and_routes():
    composite = CompositeToolSource(
        [FunctionToolSource([alpha]), FunctionToolSource([beta])]
    )
    names = [t.name for t in await composite.list_tools()]
    assert names == ["alpha", "beta"]
    assert (await composite.execute("beta", {})).content == "b"


async def test_composite_collision():
    composite = CompositeToolSource(
        [FunctionToolSource([alpha]), FunctionToolSource([alpha])]
    )
    with pytest.raises(ValueError, match="collision"):
        await composite.list_tools()


async def test_composite_unknown():
    composite = CompositeToolSource([FunctionToolSource([alpha])])
    with pytest.raises(ToolNotFoundError):
        await composite.execute("nope", {})
