"""Native toolsets, multimodal tool results, ordered fail-stop batches,
image-aware context clearing, and live artifacts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from conftest import text_response, usage
from shankit import Artifact, ArtifactEvent, ToolDef, ToolResult, ToolSource, current_tool_call
from shankit.exceptions import ToolError
from shankit.messages import (
    ImageBlock,
    Message,
    ProviderBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    content_text,
)
from shankit.models.anthropic import _message_param, build_kwargs, parse_message
from shankit.models.base import ModelRequest, ModelResponse
from shankit.models.openai import build_kwargs as openai_kwargs
from shankit.tools.base import NOT_EXECUTED, Toolset

BROWSER = Toolset(
    name="browser",
    native={"anthropic": {"type": "browser_toolset_20260801"}},
    ordered=True,
)


def member(name: str) -> ToolDef:
    return ToolDef(name=name, description=f"browser {name}", toolset=BROWSER)


class BrowserSource(ToolSource):
    """Members of an ordered toolset plus one ordinary tool; logs calls."""

    def __init__(self, fail: set[str] = frozenset()) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def list_tools(self, context: Any = None):
        return [member("click"), member("type"), member("screenshot"), ToolDef(name="lookup")]

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        self.calls.append(name)
        if name in self.fail:
            raise ToolError(f"Error: {name} failed")
        if name == "screenshot":
            return ToolResult(
                content=[
                    TextBlock(text="Screenshot"),
                    ImageBlock(media_type="image/jpeg", data="AAAA"),
                    ProviderBlock(
                        provider="anthropic",
                        data={"type": "browser_state", "tabs": []},
                        text="1 tab",
                    ),
                ]
            )
        return ToolResult(content="OK")


def calls(*names: str, toolset: str | None = "browser") -> ModelResponse:
    content = [
        ToolUseBlock(id=f"c{i}", name=n, input={}, toolset=toolset if n != "lookup" else None)
        for i, n in enumerate(names)
    ]
    return ModelResponse(content=content, stop_reason="tool_use", usage=usage())


# -- Anthropic wire shape -------------------------------------------------------


def test_native_toolset_is_sent_once_in_place_of_its_members():
    tools = [member("click"), ToolDef(name="lookup"), member("type")]
    kwargs = build_kwargs(ModelRequest(model="m", messages=[], tools=tools))
    assert kwargs["tools"] == [
        {"type": "browser_toolset_20260801"},
        {"name": "lookup", "description": "", "input_schema": {"type": "object", "properties": {}}},
    ]


def test_toolset_name_round_trips_on_calls_and_results():
    parsed = parse_message(
        SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use", id="t1", name="click", input={}, toolset_name="browser"
                )
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
    )
    assert parsed.content[0].toolset == "browser"
    call = _message_param(Message(role="assistant", content=parsed.content))
    assert call["content"][0]["toolset_name"] == "browser"
    result = _message_param(
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="OK", toolset="browser")],
        )
    )
    assert result["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "OK",
        "is_error": False,
        "toolset_name": "browser",
    }


def test_plain_calls_and_results_carry_no_toolset_key():
    msg = Message(role="user", content=[ToolResultBlock(tool_use_id="t1", content="x")])
    assert "toolset_name" not in _message_param(msg)["content"][0]


def test_multimodal_result_blocks_on_anthropic():
    content = [
        TextBlock(text="Shot"),
        ImageBlock(media_type="image/jpeg", data="AAAA"),
        ProviderBlock(
            provider="anthropic", data={"type": "browser_state", "tabs": []}, text="1 tab"
        ),
        ProviderBlock(provider="openai", data={"x": 1}, text="other provider"),
        ProviderBlock(provider="openai", data={"x": 1}),
    ]
    param = _message_param(
        Message(role="user", content=[ToolResultBlock(tool_use_id="t", content=content)])
    )
    assert param["content"][0]["content"] == [
        {"type": "text", "text": "Shot"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
        {"type": "browser_state", "tabs": []},
        {"type": "text", "text": "other provider"},
    ]


def test_content_text_renders_blocks():
    assert (
        content_text(
            [TextBlock(text="a"), ImageBlock(data="x"), ProviderBlock(provider="p", text="b")]
        )
        == "a\n[image]\nb"
    )


# -- OpenAI mirror --------------------------------------------------------------


def test_openai_gets_members_as_plain_tools_and_images_as_user_parts():
    tools = [member("click"), ToolDef(name="lookup")]
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="screenshot", input={})]),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="t1",
                    content=[
                        TextBlock(text="Shot"),
                        ImageBlock(media_type="image/png", data="QQ"),
                        ProviderBlock(
                            provider="anthropic", data={"type": "browser_state"}, text="1 tab"
                        ),
                    ],
                )
            ],
        ),
    ]
    kwargs = openai_kwargs(ModelRequest(model="m", messages=messages, tools=tools))
    assert [t["function"]["name"] for t in kwargs["tools"]] == ["click", "lookup"]
    tool_msg, image_msg = kwargs["messages"][1], kwargs["messages"][2]
    assert tool_msg == {"role": "tool", "tool_call_id": "t1", "content": "Shot\n[image]\n1 tab"}
    assert image_msg["role"] == "user"
    assert image_msg["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QQ"},
    }


# -- the loop -------------------------------------------------------------------


async def test_ordered_toolset_runs_in_order_and_stops_at_first_failure(make_agent):
    source = BrowserSource(fail={"type"})
    agent, fake = make_agent(
        [calls("click", "type", "screenshot", "lookup"), text_response("done")], tools=[source]
    )
    result = await agent.run("go", output_type=str)
    assert source.calls.count("screenshot") == 0
    assert source.calls[:2] == ["click", "type"] or source.calls[1:3] == ["click", "type"]
    results = {b.tool_use_id: b for b in fake.requests[1].messages[-1].content}
    assert results["c0"].content == "OK"
    assert results["c0"].toolset == "browser"
    assert results["c1"].is_error
    assert (results["c2"].content, results["c2"].is_error) == (NOT_EXECUTED, True)
    assert results["c3"].content == "OK"
    assert results["c3"].toolset is None  # not in the group
    assert [r.is_error for r in result.trajectory] == [False, True, True, False]


async def test_unordered_calls_still_run_concurrently(make_agent):
    started: list[str] = []
    gate = asyncio.Event()

    class Slow(ToolSource):
        async def list_tools(self, context: Any = None):
            return [ToolDef(name="a"), ToolDef(name="b")]

        async def execute(
            self, name: str, arguments: dict[str, Any], context: Any = None
        ) -> ToolResult:
            started.append(name)
            if len(started) == 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), 1)  # deadlocks if run one at a time
            return ToolResult(content=name)

    agent, _ = make_agent(
        [
            ModelResponse(
                content=[ToolUseBlock(id="1", name="a"), ToolUseBlock(id="2", name="b")],
                stop_reason="tool_use",
                usage=usage(),
            ),
            text_response("ok"),
        ],
        tools=[Slow()],
    )
    await agent.run("go", output_type=str)
    assert sorted(started) == ["a", "b"]


async def test_keep_recent_images_clears_older_screenshots_in_batches(make_agent):
    source = BrowserSource()
    script = [calls("screenshot") for _ in range(5)] + [text_response("done")]
    agent, fake = make_agent(script, tools=[source], keep_recent_images=2, reasoning="low")

    def images(request: ModelRequest) -> int:
        return sum(
            isinstance(item, ImageBlock)
            for m in request.messages
            for b in m.content
            if isinstance(b, ToolResultBlock) and not isinstance(b.content, str)
            for item in b.content
        )

    await agent.run("go", output_type=str)
    counts = [images(r) for r in fake.requests]
    # 0,1,2,3 images accumulate; at 4 (= 2 * keep) the older two are cleared.
    assert counts == [0, 1, 2, 3, 2, 3]
    assert fake.requests[4].reasoning is not None
    assert fake.requests[4].reasoning.enabled is False
    assert fake.requests[5].reasoning.enabled is True
    stubs = [
        item.text
        for m in fake.requests[4].messages
        for b in m.content
        if isinstance(b, ToolResultBlock) and not isinstance(b.content, str)
        for item in b.content
        if isinstance(item, TextBlock)
    ]
    assert stubs.count("[older image removed to save context]") == 2


async def test_live_emitted_artifact_lands_in_the_run_result(make_agent):
    class Card(ToolSource):
        async def list_tools(self, context: Any = None):
            return [ToolDef(name="start_task")]

        async def execute(
            self, name: str, arguments: dict[str, Any], context: Any = None
        ) -> ToolResult:
            call = current_tool_call()
            assert call is not None
            call.emit(ArtifactEvent(artifact=Artifact(type="task_card", data={"id": "t1"})))
            return ToolResult(content="started")

    def script() -> list[ModelResponse]:
        return [
            ModelResponse(
                content=[ToolUseBlock(id="1", name="start_task")],
                stop_reason="tool_use",
                usage=usage(),
            ),
            text_response("ok"),
        ]

    agent, _ = make_agent(script(), tools=[Card()])
    events = [e async for e in agent.stream("go")]
    assert [e.artifact.type for e in events if isinstance(e, ArtifactEvent)] == ["task_card"]
    agent, _ = make_agent(script(), tools=[Card()])
    result = await agent.run("go", output_type=str)
    assert [a.type for a in result.artifacts] == ["task_card"]
