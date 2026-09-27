"""Anthropic model client. Requires ``pip install shankit[anthropic]``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional

from ..exceptions import ModelError, ShankitError
from ..messages import (
    ContentBlock,
    ImageBlock,
    Message,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from ..usage import Usage
from .base import (
    ForcedTool,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
    map_sdk_error,
    wrap_sdk_errors,
)

__all__ = ["AnthropicModel"]

_PROVIDER = "anthropic"

_STOP_REASONS = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}


class AnthropicModel(ModelClient):
    """Anthropic client.

    ``cache_system_and_tools`` (default on) sends top-level
    ``cache_control: {"type": "ephemeral"}``, which the API applies to the
    last cacheable block of the request — so the system prompt, tool
    definitions, *and the growing message prefix* are prompt-cached across
    the iterations of an agent loop (the cache marker advances each
    iteration, writing the new suffix). Cache-read tokens are reported on
    ``Usage.cache_read_tokens``; cache-written tokens (billed at a premium
    and excluded from ``input_tokens`` by the API) on
    ``Usage.cache_write_tokens``.

    Reasoning: ``ModelRequest.reasoning`` maps to ``thinking`` (adaptive,
    or a fixed ``budget_tokens``, or disabled) plus ``output_config.effort``.
    ``thinking`` and ``redacted_thinking`` blocks come back as
    :class:`ReasoningBlock` and are re-sent verbatim on later passes, which
    tool-use continuations with thinking on require. With reasoning on,
    ``temperature`` is not sent (thinking models reject sampling
    parameters), and a pass that forces a tool (structured output) turns
    thinking off for that one request, since forced ``tool_choice`` and
    thinking can't be combined.

    Native toolsets: tools whose :class:`~shankit.tools.base.Toolset` has
    an ``"anthropic"`` entry are sent as that entry, once, instead of as
    member tools (e.g. ``browser_toolset_20260801``). Claude's calls to a
    member come back carrying ``toolset_name``, kept on
    ``ToolUseBlock.toolset`` and echoed on the call and its result - the API
    rejects a result that drops it.

    Tool results may be text or a list of text, image, and provider blocks;
    a :class:`ProviderBlock` from this provider (e.g. ``browser_state``) is
    sent verbatim, one from any other provider as its text fallback.

    ``extra_request_kwargs`` is an escape hatch merged (last) into every
    ``messages.create``/``messages.stream`` call — for provider parameters
    the neutral :class:`ModelRequest` doesn't model. Keys collide with the
    generated ones at the caller's own risk.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        cache_system_and_tools: bool = True,
        extra_request_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._cache_system_and_tools = cache_system_and_tools
        self._extra_request_kwargs = extra_request_kwargs

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

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs = build_kwargs(request, cache_system_and_tools=self._cache_system_and_tools)
        if self._extra_request_kwargs:
            kwargs.update(self._extra_request_kwargs)
        return kwargs

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        with wrap_sdk_errors(_map_error):
            message = await client.messages.create(**self._build_kwargs(request))
        return parse_message(message)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        with wrap_sdk_errors(_map_error):
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield ModelTextDelta(text=text)
                final = await stream.get_final_message()
        yield ModelResponseComplete(response=parse_message(final))

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()


def _map_error(exc: Exception) -> Optional[ModelError]:
    import anthropic

    return map_sdk_error(
        exc,
        provider="anthropic",
        status_error=anthropic.APIStatusError,
        connection_error=anthropic.APIConnectionError,  # includes APITimeoutError
        base_error=anthropic.AnthropicError,
    )


def build_kwargs(request: ModelRequest, *, cache_system_and_tools: bool = False) -> dict[str, Any]:
    """Convert a neutral request into ``anthropic.messages.create`` kwargs."""
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [p for m in request.messages if (p := _message_param(m)) is not None],
    }
    if cache_system_and_tools:
        kwargs["cache_control"] = {"type": "ephemeral"}
    if request.system is not None:
        kwargs["system"] = request.system
    reasoning = request.reasoning
    forced = isinstance(request.tool_choice, ForcedTool) or request.tool_choice == "required"
    thinking_on = reasoning is not None and reasoning.enabled and not (forced and request.tools)
    if reasoning is not None:
        if not thinking_on:
            kwargs["thinking"] = {"type": "disabled"}
        elif reasoning.budget_tokens is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": reasoning.budget_tokens}
        else:
            kwargs["thinking"] = {"type": "adaptive"}
        if thinking_on and reasoning.effort is not None:
            kwargs["output_config"] = {"effort": reasoning.effort}
    if request.temperature is not None and not thinking_on:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = _tool_params(request.tools)
        choice = request.tool_choice
        if isinstance(choice, ForcedTool):
            kwargs["tool_choice"] = {"type": "tool", "name": choice.name}
        elif choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif choice == "none":
            kwargs["tool_choice"] = {"type": "none"}
    return kwargs


def _tool_params(tools: list[Any]) -> list[dict[str, Any]]:
    """Tool definitions, with each native toolset's members folded into its
    one native entry (at the position of its first member)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        toolset = tool.toolset
        native = toolset.native.get(_PROVIDER) if toolset is not None else None
        if toolset is not None and native is not None:
            if toolset.name not in seen:
                seen.add(toolset.name)
                out.append(dict(native))
            continue
        out.append(
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        )
    return out


def _message_param(message: Message) -> Optional[dict[str, Any]]:
    """One neutral message as an Anthropic message param. Reasoning from
    this provider is re-sent verbatim; reasoning from any other provider is
    dropped. A message left with no content is omitted (consecutive
    same-role messages are merged by the API)."""
    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ReasoningBlock):
            if block.provider == _PROVIDER and block.data:
                content.append(dict(block.data))
            continue
        if isinstance(block, ToolUseBlock):
            param: dict[str, Any] = {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
            if block.toolset:
                param["toolset_name"] = block.toolset
            content.append(param)
        elif isinstance(block, ToolResultBlock):
            content.append(_tool_result_param(block))
        else:
            content.append(block.model_dump())
    if not content:
        return None
    return {"role": message.role, "content": content}


def _tool_result_param(block: ToolResultBlock) -> dict[str, Any]:
    param: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": _result_content(block.content),
        "is_error": block.is_error,
    }
    if block.toolset:
        param["toolset_name"] = block.toolset
    return param


def _result_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    out: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextBlock):
            out.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageBlock):
            out.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": item.media_type, "data": item.data},
                }
            )
        elif isinstance(item, ProviderBlock):
            if item.provider == _PROVIDER and item.data:
                out.append(dict(item.data))
            elif item.text:
                out.append({"type": "text", "text": item.text})
    return out or ""


def parse_message(message: Any) -> ModelResponse:
    """Convert an Anthropic ``Message`` into the neutral response shape."""
    content: list[ContentBlock] = []
    for block in message.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content.append(TextBlock(text=block.text))
        elif block_type == "tool_use":
            content.append(
                ToolUseBlock(
                    id=block.id,
                    name=block.name,
                    input=dict(block.input or {}),
                    toolset=getattr(block, "toolset_name", None) or None,
                )
            )
        elif block_type == "thinking":
            thinking = getattr(block, "thinking", "") or ""
            content.append(
                ReasoningBlock(
                    provider=_PROVIDER,
                    data={
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": getattr(block, "signature", "") or "",
                    },
                    text=thinking,
                )
            )
        elif block_type == "redacted_thinking":
            content.append(
                ReasoningBlock(
                    provider=_PROVIDER,
                    data={"type": "redacted_thinking", "data": getattr(block, "data", "")},
                )
            )
        # Other block types (server tool use, ...) are not part of the
        # boundary shape and are dropped here.
    usage = Usage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        requests=1,
    )
    stop = _STOP_REASONS.get(getattr(message, "stop_reason", None) or "", "other")
    return ModelResponse(content=content, stop_reason=stop, usage=usage)  # type: ignore[arg-type]
