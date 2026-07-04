"""Shared test helpers: a scripted fake model client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional

import pytest
from shankit.agent import OUTPUT_TOOL_NAME
from shankit.messages import TextBlock, ToolUseBlock
from shankit.models.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
)
from shankit.usage import Usage


class FakeModel(ModelClient):
    """Plays back a script of responses and records every request."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def _next(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeModel script exhausted")
        return self.responses.pop(0)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self._next(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        response = self._next(request)
        for block in response.content:
            if isinstance(block, TextBlock) and block.text:
                # split into two deltas to exercise accumulation
                mid = max(1, len(block.text) // 2)
                yield ModelTextDelta(text=block.text[:mid])
                if block.text[mid:]:
                    yield ModelTextDelta(text=block.text[mid:])
        yield ModelResponseComplete(response=response)


def usage(inp: int = 10, out: int = 5) -> Usage:
    return Usage(input_tokens=inp, output_tokens=out, requests=1)


def text_response(text: str) -> ModelResponse:
    return ModelResponse(content=[TextBlock(text=text)], stop_reason="end_turn", usage=usage())


def tool_call_response(
    name: str, args: dict[str, Any], *, call_id: str = "t1", text: Optional[str] = None
) -> ModelResponse:
    content: list[Any] = []
    if text:
        content.append(TextBlock(text=text))
    content.append(ToolUseBlock(id=call_id, name=name, input=args))
    return ModelResponse(content=content, stop_reason="tool_use", usage=usage())


def final_result_response(payload: dict[str, Any], *, call_id: str = "fr1") -> ModelResponse:
    return tool_call_response(OUTPUT_TOOL_NAME, payload, call_id=call_id)


@pytest.fixture
def make_agent():
    from shankit import Agent

    def factory(responses: list[ModelResponse], **kwargs: Any) -> tuple[Any, FakeModel]:
        fake = FakeModel(responses)
        agent = Agent(
            name=kwargs.pop("name", "test-agent"),
            model=kwargs.pop("model", "fake-model"),
            model_client=fake,
            **kwargs,
        )
        return agent, fake

    return factory
