"""MCPToolSource against a faked mcp SDK (no server, no subprocess)."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any, ClassVar, Optional

import pytest
from shankit import MCPToolSource

pytest.importorskip("mcp")


class _FakeSession:
    """Stands in for mcp.ClientSession."""

    # Class-level script shared by every instance (reset per test in fake_mcp).
    pages: ClassVar[list[Any]] = []
    call_results: ClassVar[dict[str, Any]] = {}

    def __init__(self, read: Any, write: Any) -> None:
        self.initialized = False
        self.closed = False

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    async def initialize(self) -> None:
        await asyncio.sleep(0)  # yield, so concurrent connectors interleave
        self.initialized = True

    async def list_tools(self, cursor: Optional[str] = None) -> Any:
        pages = {p.cursor: p for p in type(self).pages}
        return pages[cursor]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return type(self).call_results[name]


def _page(cursor: Optional[str], tools: list[Any], next_cursor: Optional[str] = None) -> Any:
    return SimpleNamespace(cursor=cursor, tools=tools, nextCursor=next_cursor)


def _tool(name: str) -> Any:
    return SimpleNamespace(name=name, description=f"{name} tool", inputSchema=None)


@pytest.fixture
def fake_mcp(monkeypatch):
    """Patch the mcp names MCPToolSource imports at connect time."""
    connects = {"count": 0}

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params):
        connects["count"] += 1
        await asyncio.sleep(0)  # widen the connect window for the race test
        yield ("read", "write")

    monkeypatch.setattr("mcp.ClientSession", _FakeSession)
    monkeypatch.setattr("mcp.client.stdio.stdio_client", fake_stdio_client)
    _FakeSession.pages = [
        _page(None, [_tool("alpha")], next_cursor="p2"),
        _page("p2", [_tool("beta")]),
    ]
    _FakeSession.call_results = {}
    return connects


async def test_list_tools_follows_pagination(fake_mcp):
    async with MCPToolSource.stdio("server-cmd") as source:
        defs = await source.list_tools()
    assert [d.name for d in defs] == ["alpha", "beta"]
    assert defs[0].description == "alpha tool"
    # A missing inputSchema falls back to an empty object schema.
    assert defs[0].input_schema == {"type": "object", "properties": {}}


async def test_execute_joins_text_parts_and_maps_is_error(fake_mcp):
    _FakeSession.call_results["alpha"] = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="line 1"),
            SimpleNamespace(type="image", data=b".."),  # non-text parts skipped
            SimpleNamespace(type="text", text="line 2"),
        ],
        isError=False,
    )
    _FakeSession.call_results["boom"] = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="it broke")], isError=True
    )
    async with MCPToolSource.stdio("server-cmd") as source:
        ok = await source.execute("alpha", {})
        failed = await source.execute("boom", {})
    assert ok.content == "line 1\nline 2"
    assert ok.is_error is False
    assert failed.content == "it broke"
    assert failed.is_error is True


async def test_concurrent_first_use_connects_exactly_once(fake_mcp):
    """The loop executes parallel tool calls concurrently; two first uses
    racing into connect() must not create two sessions (the loser's
    subprocess would be orphaned)."""
    _FakeSession.call_results["alpha"] = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], isError=False
    )
    source = MCPToolSource.stdio("server-cmd")
    try:
        results = await asyncio.gather(
            source.execute("alpha", {}),
            source.execute("alpha", {}),
            source.list_tools(),
        )
        assert results[0].content == "ok"
    finally:
        await source.aclose()
    assert fake_mcp["count"] == 1


async def test_aclose_is_idempotent_and_reconnectable(fake_mcp):
    source = MCPToolSource.stdio("server-cmd")
    await source.connect()
    await source.aclose()
    await source.aclose()  # second close is a no-op
    assert fake_mcp["count"] == 1
    # Lazy reconnect on next use.
    defs = await source.list_tools()
    assert [d.name for d in defs] == ["alpha", "beta"]
    assert fake_mcp["count"] == 2
    await source.aclose()


def test_requires_exactly_one_transport():
    with pytest.raises(ValueError, match="exactly one"):
        MCPToolSource()
    with pytest.raises(ValueError, match="exactly one"):
        MCPToolSource(command="x", url="https://y")
