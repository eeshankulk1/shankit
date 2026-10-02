"""Provider conversion is pure-function tested with SDK-shaped objects."""

from types import SimpleNamespace

import pytest
from shankit import ShankitError
from shankit.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    coerce_message,
    user_message,
)
from shankit.models.anthropic import _MODEL_RULES, ModelRules, model_rules, parse_message
from shankit.models.anthropic import build_kwargs as anthropic_kwargs
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


def test_a_system_prompt_in_parts_is_one_block_each_or_joined():
    parts = ["ours\n\n", "the user's\n\n", "", "now"]
    kwargs = anthropic_kwargs(sample_request(system=parts))
    assert kwargs["system"] == [
        {"type": "text", "text": "ours\n\n"},
        {"type": "text", "text": "the user's\n\n"},
        {"type": "text", "text": "now"},
    ]
    messages = to_openai_messages(parts, [])
    assert messages == [{"role": "system", "content": "ours\n\nthe user's\n\nnow"}]


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


@pytest.mark.parametrize(
    ("model", "thinking", "effort"),
    [
        ("claude-sonnet-5-5", {"type": "between_tools"}, None),
        ("anthropic.claude-sonnet-5-5", {"type": "between_tools"}, None),  # Bedrock id
        ("claude-opus-5-5", None, "low"),
        ("claude-fable-5-1", None, "low"),
        ("claude-mythos-5-1", None, "low"),
        ("claude-fable-5", None, "low"),
        ("claude-mythos-5", None, "low"),
        ("us.anthropic.claude-fable-5-1-v1:0", None, "low"),  # cross-region Bedrock id
        ("claude-sonnet-5", {"type": "disabled"}, None),
        ("claude-opus-5", {"type": "disabled"}, None),
        ("claude-opus-4-8", {"type": "disabled"}, None),
        ("claude-haiku-4-5", {"type": "disabled"}, None),
    ],
)
def test_anthropic_reasoning_off_takes_each_models_lowest_setting(model, thinking, effort):
    kwargs = anthropic_kwargs(sample_request(model=model, reasoning=Reasoning(enabled=False)))
    assert kwargs.get("thinking") == thinking
    assert kwargs.get("output_config") == ({"effort": effort} if effort else None)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"])
def test_anthropic_forced_tool_goes_out_as_auto_where_forcing_is_rejected(model):
    for choice in (ForcedTool(name="t"), "required"):
        kwargs = anthropic_kwargs(
            sample_request(model=model, reasoning=Reasoning(effort="high"), tool_choice=choice)
        )
        assert "tool_choice" not in kwargs  # auto
        # Nothing to be incompatible with, so the pass keeps its reasoning.
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "high"}
    none = anthropic_kwargs(sample_request(model=model, tool_choice="none"))
    assert none["tool_choice"] == {"type": "none"}


@pytest.mark.parametrize(
    ("model", "sent"),
    [
        ("claude-sonnet-5-5", False),
        ("claude-sonnet-5", False),
        ("claude-opus-4-8", False),
        ("claude-haiku-4-5", True),
        ("claude-sonnet-4-6", True),
    ],
)
def test_anthropic_temperature_only_where_the_model_takes_it(model, sent):
    kwargs = anthropic_kwargs(
        sample_request(model=model, reasoning=Reasoning(enabled=False), temperature=0.0)
    )
    assert ("temperature" in kwargs) is sent


def test_model_rules_prefixes_are_listed_most_specific_first():
    # "claude-opus-5" prefixes "claude-opus-5-5": the longer one must win.
    prefixes = [prefix for prefix, _ in _MODEL_RULES]
    for i, prefix in enumerate(prefixes):
        assert not any(later.startswith(prefix) for later in prefixes[i + 1 :]), prefix
    assert model_rules("claude-fable-5-1").forced_tool_choice is False
    assert model_rules("claude-fable-5").forced_tool_choice is True
    assert model_rules("claude-mythos-5-1").forced_tool_choice is False
    assert model_rules("claude-mythos-5").forced_tool_choice is True
    assert model_rules("claude-opus-5-5").forced_tool_choice is False
    assert model_rules("claude-opus-5").forced_tool_choice is True
    assert model_rules("claude-sonnet-5-5").thinking_off == "between_tools"
    assert model_rules("claude-sonnet-5").thinking_off == "disabled"


def test_model_rules_read_platform_ids_and_default_for_the_rest():
    sonnet = model_rules("claude-sonnet-5-5")
    for platform_id in (
        "anthropic.claude-sonnet-5-5",  # Bedrock
        "global.anthropic.claude-sonnet-5-5-v1:0",  # Bedrock inference profile
        "anthropic/claude-sonnet-5-5",  # a gateway
    ):
        assert model_rules(platform_id) == sonnet
    for other in ("m", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5", ""):
        assert model_rules(other) == ModelRules()


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"])
def test_anthropic_forced_tool_with_no_reasoning_sends_neither(model):
    kwargs = anthropic_kwargs(sample_request(model=model, tool_choice=ForcedTool(name="t")))
    assert "tool_choice" not in kwargs
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


def test_anthropic_forced_tool_without_tools_keeps_reasoning():
    # No tools, so no tool_choice goes out and nothing conflicts with thinking.
    kwargs = anthropic_kwargs(
        sample_request(tools=[], reasoning=Reasoning(effort="high"), tool_choice="required")
    )
    assert "tool_choice" not in kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


@pytest.mark.parametrize(
    ("model", "thinking", "effort"),
    [
        ("claude-fable-5", None, "low"),  # always thinks, but takes a forced tool
        ("claude-mythos-5", None, "low"),
        ("claude-opus-5", {"type": "disabled"}, None),
        ("claude-opus-4-8", {"type": "disabled"}, None),
    ],
)
def test_anthropic_forced_tool_still_forced_where_accepted(model, thinking, effort):
    kwargs = anthropic_kwargs(
        sample_request(
            model=model, reasoning=Reasoning(effort="high"), tool_choice=ForcedTool(name="t")
        )
    )
    assert kwargs["tool_choice"] == {"type": "tool", "name": "t"}
    # Forcing and the caller's reasoning don't combine: that pass runs at the
    # model's lowest setting, whatever effort the caller asked for.
    assert kwargs.get("thinking") == thinking
    assert kwargs.get("output_config") == ({"effort": effort} if effort else None)


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
    ],
)
def test_anthropic_thinking_budget_becomes_adaptive_where_it_is_rejected(model):
    kwargs = anthropic_kwargs(sample_request(model=model, reasoning=Reasoning(budget_tokens=2048)))
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "output_config" not in kwargs
    with_effort = anthropic_kwargs(
        sample_request(model=model, reasoning=Reasoning(budget_tokens=2048, effort="low"))
    )
    assert with_effort["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"])
def test_anthropic_thinking_budget_kept_where_it_is_accepted(model):
    kwargs = anthropic_kwargs(sample_request(model=model, reasoning=Reasoning(budget_tokens=2048)))
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}


async def test_a_structured_run_on_a_model_that_rejects_forcing_is_nudged_in_words():
    """End to end through the Anthropic client: forcing goes out as auto, so a
    pass that answers in prose is asked for the result tool in words."""
    from pydantic import BaseModel
    from shankit import Agent
    from shankit.agent import OUTPUT_TOOL_NAME
    from shankit.models.anthropic import AnthropicModel

    class Answer(BaseModel):
        value: int

    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    replies = [
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="It is five.")],
            stop_reason="end_turn",
            usage=usage,
        ),
        SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", id="u1", name=OUTPUT_TOOL_NAME, input={"value": 5})
            ],
            stop_reason="tool_use",
            usage=usage,
        ),
    ]
    sent: list[dict] = []

    class Messages:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return replies.pop(0)

    agent = Agent(
        name="a",
        model="claude-sonnet-5-5",
        model_client=AnthropicModel(client=SimpleNamespace(messages=Messages())),
        output_type=Answer,  # no real tools: the loop would force the result tool
    )
    result = await agent.run("what is 2 + 3?")
    assert result.output == Answer(value=5)
    assert len(sent) == 2
    assert all("tool_choice" not in kwargs for kwargs in sent)  # auto, both passes
    nudge = sent[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert OUTPUT_TOOL_NAME in nudge["content"][0]["text"]


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


# ------------------------------------------------- images in a user turn


def _photo_turn():
    return Message(
        role="user",
        content=[
            TextBlock(text="what's on this receipt?"),
            ImageBlock(media_type="image/jpeg", data="AAAA"),
        ],
    )


def test_anthropic_sends_a_user_turns_image_as_an_image_block():
    kwargs = anthropic_kwargs(sample_request(messages=[_photo_turn()]))
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's on this receipt?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
                },
            ],
        }
    ]


def test_openai_sends_a_user_turns_image_in_order_with_its_text():
    out = to_openai_messages(None, [_photo_turn()])
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's on this receipt?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
        }
    ]


def test_openai_text_only_user_turn_stays_a_string():
    out = to_openai_messages(None, [Message(role="user", content=[TextBlock(text="hi")])])
    assert out == [{"role": "user", "content": "hi"}]


def test_a_message_dict_with_an_image_block_is_coerced():
    message = coerce_message(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "this one"},
                {"type": "image", "media_type": "image/png", "data": "QQ"},
            ],
        }
    )
    assert isinstance(message.content[1], ImageBlock)


def test_user_message_takes_text_or_blocks():
    assert user_message("hi").content == [TextBlock(text="hi")]
    blocks = [TextBlock(text="hi"), ImageBlock(data="QQ")]
    assert user_message(blocks).content == blocks
