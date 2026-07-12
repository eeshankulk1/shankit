import pytest
from shankit import CompositeToolSource, FunctionToolSource, ToolNotFoundError, tool


@tool
def alpha() -> str:
    return "a"


@tool
def beta() -> str:
    return "b"


class _CountingSource(FunctionToolSource):
    def __init__(self, functions):
        super().__init__(functions)
        self.list_calls = 0

    async def list_tools(self, context=None):
        self.list_calls += 1
        return await super().list_tools(context)


async def test_composite_lists_and_routes():
    composite = CompositeToolSource([FunctionToolSource([alpha]), FunctionToolSource([beta])])
    names = [t.name for t in await composite.list_tools()]
    assert names == ["alpha", "beta"]
    assert (await composite.execute("beta", {})).content == "b"


async def test_composite_execute_routes_without_relisting():
    first = _CountingSource([alpha])
    second = _CountingSource([beta])
    composite = CompositeToolSource([first, second])
    await composite.list_tools()
    await composite.execute("beta", {})
    await composite.execute("beta", {})
    assert (first.list_calls, second.list_calls) == (1, 1)


async def test_composite_execute_rebuilds_routes_when_cold():
    # execute before any list_tools call still routes correctly
    composite = CompositeToolSource([_CountingSource([alpha])])
    assert (await composite.execute("alpha", {})).content == "a"


async def test_composite_collision():
    composite = CompositeToolSource([FunctionToolSource([alpha]), FunctionToolSource([alpha])])
    with pytest.raises(ValueError, match="collision"):
        await composite.list_tools()


async def test_composite_unknown():
    composite = CompositeToolSource([FunctionToolSource([alpha])])
    with pytest.raises(ToolNotFoundError):
        await composite.execute("nope", {})
