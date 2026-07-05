"""OpenAI model client (Chat Completions). Requires ``pip install shankit[openai]``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Optional

from ..exceptions import ShankitError
from ..messages import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
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

__all__ = ["OpenAIModel"]

_FINISH_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


class OpenAIModel(ModelClient):
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
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ShankitError(
                    "OpenAIModel requires the 'openai' package. "
                    "Install with: pip install shankit[openai]"
                ) from exc
            self._client = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        response = await client.chat.completions.create(**build_kwargs(request))
        return parse_completion(response)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        client = self._get_client()
        kwargs = build_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream = await client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        usage = Usage(requests=1)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage.input_tokens = chunk.usage.prompt_tokens or 0
                usage.output_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                text_parts.append(delta.content)
                yield ModelTextDelta(text=delta.content)
            for tc in delta.tool_calls or []:
                acc = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function and tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    acc["arguments"] += tc.function.arguments

        content: list[ContentBlock] = []
        text = "".join(text_parts)
        if text:
            content.append(TextBlock(text=text))
        for index in sorted(tool_calls):
            acc = tool_calls[index]
            content.append(
                ToolUseBlock(
                    id=acc["id"] or f"call_{index}",
                    name=acc["name"],
                    input=_parse_arguments(acc["arguments"]),
                )
            )
        stop = _FINISH_REASONS.get(finish_reason or "", "other")
        yield ModelResponseComplete(
            response=ModelResponse(content=content, stop_reason=stop, usage=usage)  # type: ignore[arg-type]
        )

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()


def build_kwargs(request: ModelRequest) -> dict[str, Any]:
    """Convert a neutral request into ``chat.completions.create`` kwargs."""
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": to_openai_messages(request.system, request.messages),
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]
        choice = request.tool_choice
        if isinstance(choice, ForcedTool):
            kwargs["tool_choice"] = {"type": "function", "function": {"name": choice.name}}
        elif choice in ("required", "none"):
            kwargs["tool_choice"] = choice
    return kwargs


def to_openai_messages(system: Optional[str], messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "user":
            texts: list[str] = []
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content
                    if block.is_error:
                        content = f"ERROR: {content}"
                    out.append(
                        {"role": "tool", "tool_call_id": block.tool_use_id, "content": content}
                    )
                elif isinstance(block, TextBlock):
                    texts.append(block.text)
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        else:
            text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
            calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input)},
                }
                for b in message.content
                if isinstance(b, ToolUseBlock)
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
    return out


def parse_completion(response: Any) -> ModelResponse:
    choice = response.choices[0]
    message = choice.message
    content: list[ContentBlock] = []
    if message.content:
        content.append(TextBlock(text=message.content))
    for tc in message.tool_calls or []:
        content.append(
            ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=_parse_arguments(tc.function.arguments),
            )
        )
    usage = Usage(requests=1)
    if getattr(response, "usage", None):
        usage.input_tokens = response.usage.prompt_tokens or 0
        usage.output_tokens = response.usage.completion_tokens or 0
    stop = _FINISH_REASONS.get(choice.finish_reason or "", "other")
    return ModelResponse(content=content, stop_reason=stop, usage=usage)  # type: ignore[arg-type]


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}
