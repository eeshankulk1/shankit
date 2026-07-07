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
    defaults = dict(
        model="m",
        system="be helpful",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        tools=[ToolDef(name="t", description="d", input_schema={"type": "object", "properties": {}})],
    )
    defaults.update(overrides)
    return ModelRequest(**defaults)


# ---------------------------------------------------------------- anthropic


def test_anthropic_kwargs_shape():
    kwargs = anthropic_kwargs(sample_request(temperature=0.2))
    assert kwargs["system"] == "be helpful"
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"][0]["content"][0] == {"type": "text", "text": "hi"}
    assert kwargs["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert "tool_choice" not in kwargs  # auto is the provider default


def test_anthropic_tool_choice_mapping():
    assert anthropic_kwargs(sample_request(tool_choice="required"))["tool_choice"] == {"type": "any"}
    forced = anthropic_kwargs(sample_request(tool_choice=ForcedTool(name="t")))["tool_choice"]
    assert forced == {"type": "tool", "name": "t"}


def test_anthropic_parse_message():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="tu1", name="search", input={"q": "x"}),
            SimpleNamespace(type="thinking", thinking="..."),  # dropped
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
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="bad", is_error=True)])
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
                        SimpleNamespace(id="c1", function=SimpleNamespace(name="t", arguments="{oops"))
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
