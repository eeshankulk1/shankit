"""Anthropic model client. Requires ``pip install shankit[anthropic]``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional

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
    Reasoning,
    map_sdk_error,
    wrap_sdk_errors,
)

__all__ = ["AnthropicModel", "CacheTTL", "ModelRules", "model_rules"]

_PROVIDER = "anthropic"

_STOP_REASONS = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}

#: How long a prompt-cache entry lives: Anthropic's ``ephemeral`` TTLs. An
#: hour costs twice the input price to write; five minutes, 1.25 times.
CacheTTL = Literal["5m", "1h"]


class AnthropicModel(ModelClient):
    """Anthropic client.

    ``prompt_cache`` (default ``"5m"``; ``None`` turns caching off) places
    cache breakpoints so a request reads back as much of the last one as
    stayed the same. The API reads an entry only within 20 blocks of one of
    the request's own breakpoints, so one mark at the end loses everything
    after a long tool turn. There are four, the API's limit:

    - the system prompt's last part (tools, which come first, ride along);
    - the newest two prompts (user messages that aren't only tool results),
      each on its last block before any ``volatile`` one: the next turn
      reads up to where this one started, however many tool calls it made;
    - the request's last block, unless it is ``volatile``: the next pass of
      the agent loop reads this whole one. It always lives five minutes,
      since only the same turn reads it;
    - with a slot left (a first message: a new chat, a sub-agent, a
      one-shot call), what stays the same across conversations: a system
      prompt's first part when there are several, else the tool list.

    ``"1h"`` keeps all but the last for an hour, for a chat whose user comes
    back after five minutes; writes cost more (``Usage.cache_write_1h_tokens``
    says how many). A ``volatile`` block (:class:`~shankit.messages.TextBlock`,
    :class:`~shankit.messages.ImageBlock`) is one the next request won't send
    in the same place - this turn's notes, a photo replayed for a few turns -
    so no breakpoint lands on it or after it in its message. Cache-read
    tokens are reported on ``Usage.cache_read_tokens``; cache-written tokens
    (billed at a premium and excluded from ``input_tokens`` by the API) on
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

    Newer models reject some of these shapes (``thinking: disabled``, a
    ``budget_tokens`` budget, forced ``tool_choice``, ``temperature``);
    :func:`model_rules` says what each model takes instead, so the same
    :class:`ModelRequest` works on every model: reasoning off is
    ``between_tools`` on Claude Sonnet 5.5, and low effort on the models
    that always think (Claude Opus 5.5, Fable 5.1); a budget becomes
    adaptive thinking; a forced tool goes out as ``auto`` where forcing is
    rejected (the reasoning then stays on, with nothing to conflict with).

    Native toolsets: tools whose :class:`~shankit.tools.base.Toolset` has
    an ``"anthropic"`` entry are sent as that entry, once, instead of as
    member tools (e.g. ``browser_toolset_20260801``). Claude's calls to a
    member come back carrying ``toolset_name``, kept on
    ``ToolUseBlock.toolset`` and echoed on the call and its result - the API
    rejects a result that drops it.

    Tool results may be text or a list of text, image, and provider blocks;
    a :class:`ProviderBlock` from this provider (e.g. ``browser_state``) is
    sent verbatim, one from any other provider as its text fallback.

    ``default_reasoning`` stands in for a request's ``reasoning`` when it
    has none: Claude thinks adaptively when the parameter is left out, so
    ``Reasoning(enabled=False)`` makes cheap passes cheap unless an agent
    asks for more.

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
        prompt_cache: Optional[CacheTTL] = "5m",
        default_reasoning: Optional[Reasoning] = None,
        extra_request_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._prompt_cache: Optional[CacheTTL] = prompt_cache
        self._default_reasoning = default_reasoning
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
        if request.reasoning is None and self._default_reasoning is not None:
            request = request.model_copy(update={"reasoning": self._default_reasoning})
        kwargs = build_kwargs(request, prompt_cache=self._prompt_cache)
        if self._extra_request_kwargs:
            extra = dict(self._extra_request_kwargs)
            if "extra_body" in kwargs and "extra_body" in extra:
                extra["extra_body"] = {**kwargs["extra_body"], **extra["extra_body"]}
            kwargs.update(extra)
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


def build_kwargs(
    request: ModelRequest, *, prompt_cache: Optional[CacheTTL] = None
) -> dict[str, Any]:
    """Convert a neutral request into ``anthropic.messages.create`` kwargs."""
    params = [p for m in request.messages if (p := _message_param(m)) is not None]
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [param for param, _ in params],
    }
    if isinstance(request.system, list):
        kwargs["system"] = [{"type": "text", "text": part} for part in request.system if part]
    elif request.system is not None:
        kwargs["system"] = request.system
    rules = model_rules(request.model)
    reasoning = request.reasoning
    choice = request.tool_choice
    forced = rules.forced_tool_choice and (isinstance(choice, ForcedTool) or choice == "required")
    thinking_on = reasoning is not None and reasoning.enabled and not (forced and request.tools)
    if reasoning is not None:
        if thinking_on:
            if reasoning.budget_tokens is not None and rules.thinking_budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": reasoning.budget_tokens}
            else:
                kwargs["thinking"] = {"type": "adaptive"}
            if reasoning.effort is not None:
                kwargs["output_config"] = {"effort": reasoning.effort}
        else:
            if rules.thinking_off is not None:
                kwargs["thinking"] = {"type": rules.thinking_off}
            if rules.off_effort is not None:
                kwargs["output_config"] = {"effort": rules.off_effort}
    if request.temperature is not None and not thinking_on and rules.sampling:
        # In the request body, not as a keyword: SDK 1.0 dropped the
        # sampling keywords, which older models still take.
        kwargs["extra_body"] = {"temperature": request.temperature}
    if request.tools:
        kwargs["tools"] = _tool_params(request.tools)
        if forced and isinstance(choice, ForcedTool):
            kwargs["tool_choice"] = {"type": "tool", "name": choice.name}
        elif forced:
            kwargs["tool_choice"] = {"type": "any"}
        elif choice == "none":
            kwargs["tool_choice"] = {"type": "none"}
    if prompt_cache is not None:
        _mark_cache(kwargs, [stable for _, stable in params], prompt_cache)
    return kwargs


def _mark_cache(kwargs: dict[str, Any], stable: list[int], ttl: CacheTTL) -> None:
    """The cache breakpoints :class:`AnthropicModel` describes. ``stable``
    is, per message, how many of its blocks come before its first volatile
    one."""
    mark = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": ttl}
    marked = 0
    system: Any = kwargs.get("system")
    if isinstance(system, str) and system:
        system = kwargs["system"] = [{"type": "text", "text": system}]
    if system:
        system[-1]["cache_control"] = dict(mark)
        marked += 1
    messages = kwargs["messages"]
    prompts = [i for i, m in enumerate(messages) if _is_prompt(m)]
    for i in prompts[-2:]:
        if stable[i]:
            messages[i]["content"][stable[i] - 1]["cache_control"] = dict(mark)
            marked += 1
    if marked < 3:
        tools = kwargs.get("tools")
        if system and len(system) > 1:
            system[0]["cache_control"] = dict(mark)
        elif tools and "input_schema" in tools[-1]:
            tools[-1] = {**tools[-1], "cache_control": dict(mark)}
    if messages and stable[-1] == len(messages[-1]["content"]):
        last = messages[-1]["content"][-1]
        if "cache_control" not in last and last["type"] not in _UNMARKABLE:
            last["cache_control"] = {"type": "ephemeral"}


# Blocks the API takes no cache_control on.
_UNMARKABLE = {"thinking", "redacted_thinking"}


def _is_prompt(message: dict[str, Any]) -> bool:
    """A user turn, as opposed to one that only carries tool results (a
    step inside a turn)."""
    return message["role"] == "user" and any(
        block["type"] != "tool_result" for block in message["content"]
    )


@dataclass(frozen=True)
class ModelRules:
    """What one Claude model accepts where request shapes differ by model.

    ``thinking_off`` is the ``thinking`` type ``Reasoning(enabled=False)``
    sends, or ``None`` for a model that can't turn thinking off: the
    parameter is left out and ``off_effort`` (if set) keeps the thinking
    short. ``forced_tool_choice``: whether ``tool_choice`` ``tool``/``any``
    is accepted; where it isn't, a forced request goes out as ``auto`` (the
    agent loop's structured-output nudge asks for the tool in words).
    ``sampling``: whether ``temperature`` is accepted. ``thinking_budget``:
    whether a fixed ``thinking.budget_tokens`` is accepted; where it isn't,
    a request for one gets adaptive thinking instead.
    """

    thinking_off: Optional[str] = "disabled"
    off_effort: Optional[str] = None
    forced_tool_choice: bool = True
    sampling: bool = True
    thinking_budget: bool = True


# Every model listed below takes adaptive thinking only (no ``budget_tokens``)
# and no sampling parameters. Some also can't turn thinking off, or can't be
# forced to call a tool.
_ALWAYS_THINKS = ModelRules(
    thinking_off=None, off_effort="low", sampling=False, thinking_budget=False
)
_ALWAYS_THINKS_NO_FORCING = replace(_ALWAYS_THINKS, forced_tool_choice=False)
_ADAPTIVE_ONLY = ModelRules(sampling=False, thinking_budget=False)

# Most specific prefix first: "claude-opus-5" also prefixes "claude-opus-5-5".
# A model not listed (Haiku 4.5, Sonnet 4.6 and older) takes the defaults.
_MODEL_RULES: tuple[tuple[str, ModelRules], ...] = (
    ("claude-fable-5-1", _ALWAYS_THINKS_NO_FORCING),
    ("claude-mythos-5-1", _ALWAYS_THINKS_NO_FORCING),
    ("claude-opus-5-5", _ALWAYS_THINKS_NO_FORCING),
    # `between_tools`: no extended thinking, only short notes between tools.
    (
        "claude-sonnet-5-5",
        ModelRules(
            thinking_off="between_tools",
            forced_tool_choice=False,
            sampling=False,
            thinking_budget=False,
        ),
    ),
    ("claude-fable-5", _ALWAYS_THINKS),
    ("claude-mythos-5", _ALWAYS_THINKS),
    ("claude-opus-5", _ADAPTIVE_ONLY),
    ("claude-sonnet-5", _ADAPTIVE_ONLY),
    ("claude-haiku-5-5", _ADAPTIVE_ONLY),
    ("claude-opus-4-8", _ADAPTIVE_ONLY),
    ("claude-opus-4-7", _ADAPTIVE_ONLY),
)


def model_rules(model: str) -> ModelRules:
    """The request rules for ``model`` (a bare id, or a platform id such as
    Bedrock's ``anthropic.claude-...``)."""
    at = model.find("claude-")
    name = model[at:] if at >= 0 else model
    for prefix, rules in _MODEL_RULES:
        if name.startswith(prefix):
            return rules
    return ModelRules()


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


def _message_param(message: Message) -> Optional[tuple[dict[str, Any], int]]:
    """One neutral message as an Anthropic message param, and how many of
    its blocks come before the first volatile one. Reasoning from this
    provider is re-sent verbatim; reasoning from any other provider is
    dropped. So is a text block with only whitespace in it: the API refuses
    the whole request over one, and a stored history goes out again on
    every later turn. A message left with no content is omitted
    (consecutive same-role messages are merged by the API)."""
    content: list[dict[str, Any]] = []
    stable: Optional[int] = None
    for block in message.content:
        if stable is None and getattr(block, "volatile", False):
            stable = len(content)
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
        elif isinstance(block, ImageBlock):
            content.append(_image_param(block))
        elif isinstance(block, TextBlock):
            if block.text.strip():
                content.append({"type": "text", "text": block.text})
        else:
            content.append(block.model_dump())
    if not content:
        return None
    return {"role": message.role, "content": content}, len(content) if stable is None else stable


def _image_param(block: ImageBlock) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
    }


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
            out.append(_image_param(item))
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
    creation = getattr(message.usage, "cache_creation", None)
    usage = Usage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        cache_write_1h_tokens=getattr(creation, "ephemeral_1h_input_tokens", 0) or 0,
        requests=1,
    )
    stop = _STOP_REASONS.get(getattr(message, "stop_reason", None) or "", "other")
    return ModelResponse(content=content, stop_reason=stop, usage=usage)  # type: ignore[arg-type]
