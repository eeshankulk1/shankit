"""Anthropic model client. Requires ``pip install shankit[anthropic]``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional

from ..exceptions import ShankitError
from ..messages import ContentBlock, TextBlock, ToolUseBlock
from ..usage import Usage
from .base import (
    ForcedTool,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
)

__all__ = ["AnthropicModel"]

_STOP_REASONS = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}


class AnthropicModel(ModelClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ShankitError(
                    "AnthropicModel requires the 'anthropic' package. "
                    "Install with: pip install shankit[anthropic]"
                ) from exc
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        message = await client.messages.create(**build_kwargs(request))
        return parse_message(message)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        client = self._get_client()
        async with client.messages.stream(**build_kwargs(request)) as stream:
            async for text in stream.text_stream:
                if text:
                    yield ModelTextDelta(text=text)
            final = await stream.get_final_message()
        yield ModelResponseComplete(response=parse_message(final))

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()


def build_kwargs(request: ModelRequest) -> dict[str, Any]:
    """Convert a neutral request into ``anthropic.messages.create`` kwargs."""
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [m.model_dump() for m in request.messages],
    }
    if request.system is not None:
        kwargs["system"] = request.system
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = [t.model_dump() for t in request.tools]
        choice = request.tool_choice
        if isinstance(choice, ForcedTool):
            kwargs["tool_choice"] = {"type": "tool", "name": choice.name}
        elif choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif choice == "none":
            kwargs["tool_choice"] = {"type": "none"}
    return kwargs


def parse_message(message: Any) -> ModelResponse:
    """Convert an Anthropic ``Message`` into the neutral response shape."""
    content: list[ContentBlock] = []
    for block in message.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content.append(TextBlock(text=block.text))
        elif block_type == "tool_use":
            content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input or {})))
        # Other block types (thinking, server tool use, ...) are not part of
        # the v1 boundary shape and are dropped here.
    usage = Usage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
        requests=1,
    )
    stop = _STOP_REASONS.get(getattr(message, "stop_reason", None) or "", "other")
    return ModelResponse(content=content, stop_reason=stop, usage=usage)  # type: ignore[arg-type]
