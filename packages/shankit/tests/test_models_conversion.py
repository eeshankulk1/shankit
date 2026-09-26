"""Provider conversion is pure-function tested with SDK-shaped objects."""

from types import SimpleNamespace

import pytest
from shankit import ShankitError
from shankit.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from shankit.models.anthropic import build_kwargs as anthropic_kwargs
from shankit.models.anthropic import parse_message
from shankit.models.base import ForcedTool, ModelRequest
from shankit.models.openai import build_kwargs as openai_kwargs
from shankit.models.openai import parse_completion, to_openai_messages
from shankit.models.registry import register_provider, resolve_model
from shankit.tools.base import ToolDef


def sample_request(**overrides):
    defaults = {
        "model": "m",
        "system": "be helpful",
        "messages": [Message(role="user", content=[TextBlock(text="hi")])],
        "tools": [
            ToolDef(name="t", description="d", input_schema={"type": "object", "properties": {}})
        ],
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)


# ---------------------------------------------------------------- anthropic


def test_anthropic_extra_request_kwargs_merge():
    from shankit.models.anthropic import AnthropicModel

    model = AnthropicModel(extra_request_kwargs={"thinking": {"type": "disabled"}})
    kwargs = model._build_kwargs(sample_request())
    assert kwargs["thinking"] == {"type": "disabled"}
    # merged last: an extra kwarg wins over a generated one
    override = AnthropicModel(extra_request_kwargs={"max_tokens": 99})
    assert override._build_kwargs(sample_request())["max_tokens"] == 99


def test_anthropic_kwargs_shape():
    kwargs = anthropic_kwargs(sample_request(temperature=0.2))
    assert kwargs["system"] == "be helpful"
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"][0]["content"][0] == {"type": "text", "text": "hi"}
    assert kwargs["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert "tool_choice" not in kwargs  # auto is the provider default


def test_anthropic_tool_choice_mapping():
    assert anthropic_kwargs(sample_request(tool_choice="required"))["tool_choice"] == {
        "type": "any"
    }
    forced = anthropic_kwargs(sample_request(tool_choice=ForcedTool(name="t")))["tool_choice"]
    assert forced == {"type": "tool", "name": "t"}


def test_anthropic_parse_message():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="tu1", name="search", input={"q": "x"}),
            SimpleNamespace(type="server_tool_use", id="st1"),  # dropped
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    response = parse_message(message)
    assert response.text == "hello"
    assert response.content[1] == ToolUseBlock(id="tu1", name="search", input={"q": "x"})
    assert len(response.content) == 2
    assert response.stop_reason == "tool_use"
    assert response.usage.input_tokens == 11
    assert response.usage.requests == 1


def test_anthropic_parse_message_cache_tokens():
    """Cache reads AND cache writes both land on Usage; with prompt caching
    on, the API excludes cache-written tokens from input_tokens, so dropping
    them would undercount nearly the whole prompt on cache-writing calls."""
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=7,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=2500,
        ),
    )
    usage = parse_message(message).usage
    assert usage.cache_read_tokens == 1000
    assert usage.cache_write_tokens == 2500


# ------------------------------------------------------------------ openai


def test_openai_message_conversion():
    messages = [
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(
            role="assistant",
            content=[
                TextBlock(text="let me check"),
                ToolUseBlock(id="c1", name="search", input={"q": "x"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="c1", content="result!", is_error=False)],
        ),
    ]
    out = to_openai_messages("sys", messages)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "search"
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "result!"}


def test_openai_error_results_prefixed():
    messages = [
        Message(
            role="user", content=[ToolResultBlock(tool_use_id="c1", content="bad", is_error=True)]
        )
    ]
    out = to_openai_messages(None, messages)
    assert out[0]["content"] == "ERROR: bad"


def test_openai_kwargs_tools():
    kwargs = openai_kwargs(sample_request(tool_choice=ForcedTool(name="t")))
    assert kwargs["tools"][0]["function"]["name"] == "t"
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "t"}}


def test_openai_parse_completion():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="thinking...",
                    tool_calls=[
                        SimpleNamespace(
                            id="c9",
                            function=SimpleNamespace(name="add", arguments='{"a": 1}'),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )
    parsed = parse_completion(response)
    assert parsed.stop_reason == "tool_use"
    assert parsed.content[0] == TextBlock(text="thinking...")
    assert parsed.content[1].input == {"a": 1}
    assert parsed.usage.output_tokens == 2


def test_openai_malformed_arguments_preserved():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1", function=SimpleNamespace(name="t", arguments="{oops")
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    parsed = parse_completion(response)
    assert parsed.content[0].input == {"_raw": "{oops"}


# ---------------------------------------------------------------- registry


def test_resolve_model_requires_prefix():
    with pytest.raises(ShankitError, match="provider prefix"):
        resolve_model("claude-sonnet-4-5")


def test_resolve_model_unknown_provider():
    with pytest.raises(ShankitError, match="Unknown model provider"):
        resolve_model("nonsense:model-1")


def test_resolve_model_custom_provider():
    from conftest import FakeModel

    register_provider("faketest", lambda: FakeModel([]))
    client, model_id = resolve_model("faketest:model-x")
    assert isinstance(client, FakeModel)
    assert model_id == "model-x"
    # provider clients are cached
    client2, _ = resolve_model("faketest:model-y")
    assert client2 is client


def test_resolve_model_with_explicit_client():
    from conftest import FakeModel

    fake = FakeModel([])
    client, model_id = resolve_model("anything-goes", fake)
    assert client is fake
    assert model_id == "anything-goes"


async def test_shutdown_closes_and_clears_cached_clients():
    from conftest import FakeModel
    from shankit.models.registry import shutdown

    class ClosableModel(FakeModel):
        closed = False

        async def aclose(self):
            self.closed = True

    register_provider("closetest", lambda: ClosableModel([]))
    client, _ = resolve_model("closetest:model-x")
    await shutdown()
    assert client.closed
    # the cache was cleared: the next resolve constructs a fresh client
    client2, _ = resolve_model("closetest:model-x")
    assert client2 is not client


async def test_shutdown_survives_a_failing_close():
    from conftest import FakeModel
    from shankit.models.registry import shutdown

    class ExplodingClose(FakeModel):
        async def aclose(self):
            raise RuntimeError("transport already gone")

    class ClosableModel(FakeModel):
        closed = False

        async def aclose(self):
            self.closed = True

    register_provider("badclose", lambda: ExplodingClose([]))
    register_provider("goodclose", lambda: ClosableModel([]))
    resolve_model("badclose:m")
    good, _ = resolve_model("goodclose:m")
    await shutdown()  # must not raise
    assert good.closed  # the healthy client still closed


# ---------------------------------------------------------------- reasoning

from shankit.messages import ReasoningBlock  # noqa: E402
from shankit.models.base import Reasoning  # noqa: E402


def test_anthropic_reasoning_maps_to_thinking_and_effort():
    kwargs = anthropic_kwargs(sample_request(reasoning=Reasoning(effort="high"), temperature=0.3))
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs  # thinking models reject sampling params

    budget = anthropic_kwargs(sample_request(reasoning=Reasoning(budget_tokens=2048)))
    assert budget["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in budget

    off = anthropic_kwargs(sample_request(reasoning=Reasoning(enabled=False), temperature=0.2))
    assert off["thinking"] == {"type": "disabled"}
    assert off["temperature"] == 0.2

    assert "thinking" not in anthropic_kwargs(sample_request())


def test_anthropic_forced_tool_turns_thinking_off_for_that_request():
    kwargs = anthropic_kwargs(
        sample_request(reasoning=Reasoning(effort="high"), tool_choice=ForcedTool(name="t"))
    )
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "output_config" not in kwargs


def test_anthropic_thinking_round_trips_and_foreign_reasoning_is_dropped():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="let me see", signature="sig1"),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="tool_use", id="tu1", name="t", input={}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    response = parse_message(message)
    thinking, redacted, _ = response.content
    assert isinstance(thinking, ReasoningBlock)
    assert thinking.text == "let me see"
    assert isinstance(redacted, ReasoningBlock)

    foreign = ReasoningBlock(provider="openai", data={"id": "rs_1"})
    request = sample_request(
        messages=[
            Message(role="user", content=[TextBlock(text="hi")]),
            Message(role="assistant", content=[*response.content]),
            Message(role="user", content=[ToolResultBlock(tool_use_id="tu1", content="ok")]),
            Message(role="assistant", content=[foreign]),  # nothing left -> omitted
        ]
    )
    sent = anthropic_kwargs(request)["messages"]
    assert len(sent) == 3
    assert sent[1]["content"][0] == {
        "type": "thinking",
        "thinking": "let me see",
        "signature": "sig1",
    }
    assert sent[1]["content"][1] == {"type": "redacted_thinking", "data": "opaque"}
    assert sent[1]["content"][2]["type"] == "tool_use"


def test_openai_reasoning_effort_and_reasoning_blocks_ignored():
    kwargs = openai_kwargs(sample_request(reasoning=Reasoning(effort="xhigh")))
    assert kwargs["reasoning_effort"] == "high"
    assert "reasoning_effort" not in openai_kwargs(sample_request(reasoning=Reasoning()))
    messages = to_openai_messages(
        None,
        [
            Message(
                role="assistant",
                content=[ReasoningBlock(provider="anthropic", data={}), TextBlock(text="hi")],
            )
        ],
    )
    assert messages == [{"role": "assistant", "content": "hi"}]
