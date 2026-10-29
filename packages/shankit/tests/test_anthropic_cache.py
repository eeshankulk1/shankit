"""Where AnthropicModel puts prompt-cache breakpoints (``prompt_cache``)."""

from __future__ import annotations

from typing import Any

from shankit.messages import ImageBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from shankit.models.anthropic import AnthropicModel, build_kwargs
from shankit.models.base import ModelRequest, Reasoning
from shankit.tools.base import ToolDef

FIVE = {"type": "ephemeral"}
HOUR = {"type": "ephemeral", "ttl": "1h"}


def _user(*blocks: Any) -> Message:
    return Message(
        role="user", content=[TextBlock(text=b) if isinstance(b, str) else b for b in blocks]
    )


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)])


def _tool_step(n: int) -> list[Message]:
    call = ToolUseBlock(id=f"c{n}", name="t", input={})
    return [
        Message(role="assistant", content=[call]),
        Message(role="user", content=[ToolResultBlock(tool_use_id=f"c{n}", content="r")]),
    ]


def _request(messages: list[Message], system: Any = "sys", tools: bool = True) -> ModelRequest:
    return ModelRequest(
        model="claude-sonnet-4-6",
        system=system,
        messages=messages,
        tools=[ToolDef(name="t", description="d")] if tools else [],
    )


def _marks(kwargs: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every breakpoint, as (where, cache_control)."""
    out: list[tuple[str, Any]] = []
    for i, tool in enumerate(kwargs.get("tools") or []):
        if "cache_control" in tool:
            out.append((f"tool{i}", tool["cache_control"]))
    system = kwargs.get("system")
    if isinstance(system, list):
        for i, part in enumerate(system):
            if "cache_control" in part:
                out.append((f"system{i}", part["cache_control"]))
    for m, message in enumerate(kwargs["messages"]):
        for b, block in enumerate(message["content"]):
            if "cache_control" in block:
                out.append((f"m{m}.{b}", block["cache_control"]))
    return out


def test_first_message_marks_system_tools_and_prompt():
    kwargs = build_kwargs(_request([_user("q")]), prompt_cache="5m")
    assert "cache_control" not in kwargs
    assert _marks(kwargs) == [("tool0", FIVE), ("system0", FIVE), ("m0.0", FIVE)]


def test_long_turn_keeps_marks_on_the_last_two_prompts():
    messages = [_user("q1"), _assistant("a1"), _user("q2")]
    for n in range(15):
        messages += _tool_step(n)
    kwargs = build_kwargs(_request(messages), prompt_cache="5m")
    last = len(messages) - 1
    assert _marks(kwargs) == [
        ("system0", FIVE),
        ("m0.0", FIVE),
        ("m2.0", FIVE),
        (f"m{last}.0", FIVE),
    ]


def test_system_in_parts_marks_the_last_and_with_a_slot_left_the_first():
    kwargs = build_kwargs(_request([_user("q")], system=["shared", "per user"]), prompt_cache="5m")
    assert _marks(kwargs) == [("system0", FIVE), ("system1", FIVE), ("m0.0", FIVE)]


def test_volatile_blocks_stay_after_the_breakpoint():
    note = TextBlock(text="<turn note>", volatile=True)
    messages = [_user("q1"), _assistant("a1"), _user("q2", note)]
    kwargs = build_kwargs(_request(messages), prompt_cache="5m")
    # The prompt's mark sits before the note, and the request's last block
    # (the note) gets none: the next request won't send it there.
    assert ("m2.0", FIVE) in _marks(kwargs)
    assert all(not where.startswith("m2.1") for where, _ in _marks(kwargs))
    assert "volatile" not in kwargs["messages"][2]["content"][1]


def test_volatile_photo_moves_the_mark_to_the_text_before_it():
    photo = ImageBlock(data="aGk=", volatile=True)
    kwargs = build_kwargs(_request([_user("look", photo)]), prompt_cache="5m")
    assert ("m0.0", FIVE) in _marks(kwargs)
    assert "cache_control" not in kwargs["messages"][0]["content"][1]


def test_one_hour_ttl_leaves_the_tail_at_five_minutes():
    messages = [_user("q1"), _assistant("a1"), _user("q2"), *_tool_step(0)]
    kwargs = build_kwargs(_request(messages), prompt_cache="1h")
    marks = _marks(kwargs)
    assert marks[-1] == ("m4.0", FIVE)
    assert all(mark == HOUR for _, mark in marks[:-1])


def test_at_most_four_breakpoints():
    messages = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2"), _user("q3")]
    for system in ("sys", ["a", "b", "c"]):
        for extra in ([], _tool_step(0)):
            kwargs = build_kwargs(_request(messages + extra, system=system), prompt_cache="5m")
            assert len(_marks(kwargs)) <= 4


def test_cache_off_sends_no_marks():
    assert _marks(build_kwargs(_request([_user("q")]))) == []
    assert (
        AnthropicModel(prompt_cache=None)._build_kwargs(_request([_user("q")]))["system"] == "sys"
    )


def test_default_reasoning_fills_in_only_when_the_request_has_none():
    model = AnthropicModel(default_reasoning=Reasoning(enabled=False))
    assert model._build_kwargs(_request([_user("q")]))["thinking"] == {"type": "disabled"}
    asked = _request([_user("q")]).model_copy(update={"reasoning": Reasoning(effort="high")})
    assert model._build_kwargs(asked)["thinking"] == {"type": "adaptive"}


def test_volatile_is_never_serialized():
    block = TextBlock(text="x", volatile=True)
    assert block.model_dump() == {"type": "text", "text": "x"}
