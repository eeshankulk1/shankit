"""Any MCP server as a tool source (design §3.1).

Requires the optional ``mcp`` extra: ``pip install shankit[mcp]``.

Preferred usage is as an async context manager so the session lifecycle is
explicit::

    async with MCPToolSource.stdio("uvx", ["mcp-server-fetch"]) as fetch:
        agent = Agent(name="reader", model="anthropic:...", tools=[fetch])
        ...

The source also lazily connects on first use, but then you own calling
``aclose()`` from the same task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any, Optional

from ..exceptions import ShankitError
from .base import ToolDef, ToolResult, ToolSource

__all__ = ["MCPToolSource"]


class MCPToolSource(ToolSource):
    def __init__(
        self,
        *,
        command: Optional[str] = None,
        args: Sequence[str] = (),
        env: Optional[Mapping[str, str]] = None,
        url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if (command is None) == (url is None):
            raise ValueError("Provide exactly one of 'command' (stdio) or 'url' (streamable HTTP).")
        self._command = command
        self._args = list(args)
        self._env = dict(env) if env else None
        self._url = url
        self._headers = dict(headers) if headers else None
        self._stack: Optional[AsyncExitStack] = None
        self._session: Any = None

    @classmethod
    def stdio(
        cls,
        command: str,
        args: Sequence[str] = (),
        env: Optional[Mapping[str, str]] = None,
    ) -> MCPToolSource:
        """An MCP server run as a subprocess over stdio."""
        return cls(command=command, args=args, env=env)

    @classmethod
    def http(cls, url: str, headers: Optional[Mapping[str, str]] = None) -> MCPToolSource:
        """A remote MCP server over streamable HTTP."""
        return cls(url=url, headers=headers)

    async def connect(self) -> MCPToolSource:
        if self._session is not None:
            return self
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover
            raise ShankitError(
                "MCPToolSource requires the 'mcp' package. Install with: pip install shankit[mcp]"
            ) from exc

        stack = AsyncExitStack()
        try:
            if self._url is not None:
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self._url, headers=self._headers)
                )
            else:
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(
                    command=self._command, args=self._args, env=self._env
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        return self

    async def aclose(self) -> None:
        if self._stack is not None:
            stack, self._stack, self._session = self._stack, None, None
            await stack.aclose()

    async def __aenter__(self) -> MCPToolSource:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _ensure_connected(self) -> Any:
        if self._session is None:
            await self.connect()
        return self._session

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        session = await self._ensure_connected()
        defs: list[ToolDef] = []
        cursor: Optional[str] = None
        while True:
            page = await session.list_tools(cursor=cursor) if cursor else await session.list_tools()
            for t in page.tools:
                defs.append(
                    ToolDef(
                        name=t.name,
                        description=t.description or "",
                        input_schema=t.inputSchema or {"type": "object", "properties": {}},
                    )
                )
            cursor = getattr(page, "nextCursor", None)
            if not cursor:
                return defs

    async def execute(self, name: str, arguments: dict[str, Any], context: Any = None) -> ToolResult:
        session = await self._ensure_connected()
        result = await session.call_tool(name, arguments or {})
        parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        return ToolResult(content="\n".join(parts), is_error=bool(getattr(result, "isError", False)))
