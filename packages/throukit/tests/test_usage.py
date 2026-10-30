"""Usage split by model: a run that mixes models can be priced right."""

from __future__ import annotations

from conftest import FakeModel, text_response, tool_call_response
from throukit import Agent, Usage


def test_add_merges_the_split():
    a = Usage(input_tokens=10, requests=1).for_model("big")
    b = Usage(input_tokens=3, requests=1).for_model("small")
    total = a + b + Usage(input_tokens=5).for_model("big")
    assert total.input_tokens == 18
    assert total.per_model() == {
        "big": Usage(input_tokens=15, requests=1),
        "small": Usage(input_tokens=3, requests=1),
    }
    # The operands are untouched.
    assert a.by_model["big"].input_tokens == 10


def test_untagged_usage_is_reported_under_none():
    total = Usage(output_tokens=4).for_model("big") + Usage(output_tokens=6)
    assert total.output_tokens == 10
    assert total.per_model() == {"big": Usage(output_tokens=4), None: Usage(output_tokens=6)}


def test_for_model_keeps_an_existing_split():
    split = Usage(input_tokens=2).for_model("small")
    assert split.for_model("big").per_model() == {"small": Usage(input_tokens=2)}


async def test_sub_agent_on_another_model_keeps_its_own_rates():
    sub = Agent(
        name="lookup",
        model="small",
        model_client=FakeModel([text_response("found it")]),
    )
    parent = Agent(
        name="main",
        model="big",
        model_client=FakeModel(
            [tool_call_response("lookup", {"task": "find it"}), text_response("done")]
        ),
        tools=[sub.as_tool()],
    )
    result = await parent.run("go", output_type=str)

    parts = result.usage.per_model()
    assert set(parts) == {"big", "small"}
    assert parts["big"].requests == 2
    assert parts["small"].requests == 1
    assert result.usage.requests == 3
