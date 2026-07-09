"""The OpenAI streaming accumulator: delta merging, parallel calls, usage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from shankit.messages import TextBlock, ToolUseBlock
from shankit.models.base import ModelRequest, ModelResponseComplete, ModelTextDelta
from shankit.models.openai import OpenAIModel


def _chunk(
    *,
    content: Optional[str] = None,
    tool_calls: Optional[list[Any]] = None,
    finish_reason: Optional[str] = None,
    usage: Any = None,
    choices: Optional[list[Any]] = None,
) -> Any:
    if choices is None:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(choices=choices, usage=usage)


def _tc(
    index: int, *, id: Optional[str] = None, name: Optional[str] = None, args: Optional[str] = None
) -> Any:
    function = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=id, function=function)


class _FakeStreamClient:
    """Mimics AsyncOpenAI far enough for OpenAIModel.stream."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.kwargs: dict[str, Any] = {}

    async def _create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs

        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


async def _stream(chunks: list[Any]) -> tuple[list[str], Any, dict[str, Any]]:
    client = _FakeStreamClient(chunks)
    model = OpenAIModel(client=client)
    request = ModelRequest(model="gpt-test", messages=[])
    deltas: list[str] = []
    response = None
    async for event in model.stream(request):
        if isinstance(event, ModelTextDelta):
            deltas.append(event.text)
        elif isinstance(event, ModelResponseComplete):
            response = event.response
    assert response is not None
    return deltas, response, client.kwargs


async def test_stream_accumulates_text_and_requests_usage():
    deltas, response, kwargs = await _stream(
        [
            _chunk(content="hel"),
            _chunk(content="lo"),
            _chunk(finish_reason="stop"),
            _chunk(choices=[], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)),
        ]
    )
    assert deltas == ["hel", "lo"]
    assert response.text == "hello"
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


async def test_stream_merges_split_tool_call_arguments():
    _, response, _ = await _stream(
        [
            _chunk(tool_calls=[_tc(0, id="call_a", name="search", args='{"que')]),
            _chunk(tool_calls=[_tc(0, args='ry": "x"}')]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    (block,) = response.content
    assert isinstance(block, ToolUseBlock)
    assert block.id == "call_a"
    assert block.name == "search"
    assert block.input == {"query": "x"}
    assert response.stop_reason == "tool_use"


async def test_stream_keeps_parallel_tool_calls_separate_and_ordered():
    _, response, _ = await _stream(
        [
            _chunk(tool_calls=[_tc(0, id="a", name="first", args="{}")]),
            _chunk(tool_calls=[_tc(1, id="b", name="second", args='{"n": ')]),
            _chunk(tool_calls=[_tc(1, args="2}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    names = [b.name for b in response.content if isinstance(b, ToolUseBlock)]
    assert names == ["first", "second"]
    second = response.content[-1]
    assert isinstance(second, ToolUseBlock)
    assert second.input == {"n": 2}


async def test_stream_mixed_text_then_tool_call():
    deltas, response, _ = await _stream(
        [
            _chunk(content="Let me check."),
            _chunk(tool_calls=[_tc(0, id="c", name="lookup", args="{}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    assert deltas == ["Let me check."]
    assert isinstance(response.content[0], TextBlock)
    assert isinstance(response.content[1], ToolUseBlock)


async def test_stream_missing_tool_call_id_gets_synthetic_id():
    _, response, _ = await _stream(
        [
            _chunk(tool_calls=[_tc(0, name="t", args="{}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    (block,) = response.content
    assert isinstance(block, ToolUseBlock)
    assert block.id == "call_0"


async def test_stream_length_finish_reason_maps_to_max_tokens():
    _, response, _ = await _stream([_chunk(content="x", finish_reason="length")])
    assert response.stop_reason == "max_tokens"
