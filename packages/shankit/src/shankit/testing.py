"""Test doubles for code built on shankit.

:class:`ScriptedModel` is a provider-neutral fake :class:`ModelClient`. It
replays a script, one turn per model call - text (streamed as deltas), tool
calls, reasoning, structured output, usage, raw content blocks, or a raised
error - and records every request the agent loop sends, so a test can
assert on what the model was asked as well as on what the agent did with
the answers::

    model = ScriptedModel([
        Turn("Let me check.", tool_calls=[ToolCall("order_status", {"order_id": 7})]),
        "Order 7 has shipped.",
    ])
    agent = Agent(name="support", model="scripted", model_client=model, tools=[order_status])
    result = await agent.run("Where is order 7?", output_type=str)
    assert model.requests[1].messages[-1].content[0].content == "shipped"

A script that disagrees with the agent (a failed ``expect``, more calls than
turns) raises :class:`ScriptFailure`, which the agent loop cannot turn into
a ``ModelError`` or an ``error`` event.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from pydantic_core import to_jsonable_python

from .agent import OUTPUT_TOOL_NAME
from .messages import (
    ContentBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    content_text,
)
from .models.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
    system_text,
)
from .usage import Usage

__all__ = ["ScriptFailure", "ScriptItem", "ScriptedModel", "ToolCall", "Turn"]

StopReason = Literal["end_turn", "tool_use", "max_tokens", "other"]

_PROVIDER = "scripted"


class ScriptFailure(BaseException):
    """The script and the agent disagreed: a turn's ``expect`` failed, the
    agent made more model calls than the script has turns, or a turn could
    not be played against the request it answers.

    Derives from ``BaseException``, as pytest's own outcome exceptions do,
    so nothing between the model and the test can swallow it: the agent
    loop turns any ``Exception`` a model client raises into a ``ModelError``
    (an ``error`` event in ``stream()``), sub-agent tools sanitize their
    failures, and application code often catches ``Exception``. Any of those
    would let a broken script pass as a provider failure.
    """


@dataclass(frozen=True)
class ToolCall:
    """One scripted tool call. ``id`` defaults to ``call_<n>``, numbered in
    order across the model's calls, so ids are the same on every run."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass(frozen=True)
class Turn:
    """What the model does on one call. Every field is optional and they
    compose; the response content is laid out as a provider's would be:
    reasoning, text, tool calls, the structured-output call, then ``blocks``.

    Args:
        text: Assistant text. ``stream()`` yields it as word-sized deltas.
        tool_calls: Tool calls, executed by the agent loop like real ones.
        reasoning: Reasoning text (sent as a ``ReasoningBlock`` from provider
            ``"scripted"``), or a ``ReasoningBlock`` to send as-is.
        output: The value ``run(output_type=...)`` should return - a model
            instance, a dict, a list, a scalar. The turn calls the synthetic
            output tool with it, wrapping non-object types the way the
            agent's output schema does. ``None`` (default): no output call.
        blocks: Content blocks appended verbatim - the escape hatch for any
            block shape the other fields don't cover.
        usage: This call's usage, as-is. Defaults to ``Usage(requests=1)``;
            real clients report ``requests=1`` per call, so keep it when you
            script token counts.
        stop_reason: Defaults to ``"tool_use"`` when the turn calls a tool,
            else ``"end_turn"``. Script ``"max_tokens"`` to test truncation.
        error: Raised instead of responding: a :class:`~shankit.ModelError`
            (``model_error_for_status("anthropic", 429)`` is exactly what the
            shipped clients raise for a rate limit, ``529`` for overloaded)
            or any exception. With ``text`` too, ``stream()`` yields the text
            before raising - a stream that fails midway.
        expect: Called with this call's :class:`ModelRequest` before the turn
            plays. Assert inside it, or return ``False``; either fails the
            test with :class:`ScriptFailure` and a summary of the request.
    """

    text: str = ""
    tool_calls: Sequence[ToolCall] = ()
    reasoning: Union[str, ReasoningBlock, None] = None
    output: Any = None
    blocks: Sequence[ContentBlock] = ()
    usage: Optional[Usage] = None
    stop_reason: Optional[StopReason] = None
    error: Optional[BaseException] = None
    expect: Optional[Callable[[ModelRequest], Any]] = None


#: One script entry: a :class:`Turn`, or a shorthand - a ``str`` (a
#: text-only turn), an exception (``Turn(error=...)``), or a
#: :class:`ModelResponse` returned verbatim.
ScriptItem = Union[Turn, str, BaseException, ModelResponse]


class ScriptedModel(ModelClient):
    """A fake model client that plays a script, one item per model call.

    Works for both run modes (``complete`` and ``stream``), as an agent's
    ``model_client=``, or behind a spec via
    ``register_provider("scripted", lambda: model)``. Every request is
    recorded on :attr:`requests` as a snapshot of what was sent. A call past
    the end of the script raises :class:`ScriptFailure`.

    One model plays one script in call order, so give agents that run
    concurrently (parallel sub-agents) their own ``ScriptedModel``.
    """

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        for index, item in enumerate(script):
            if not isinstance(item, (Turn, str, BaseException, ModelResponse)):
                raise TypeError(
                    f"ScriptedModel script[{index}] is a {type(item).__name__}; expected a "
                    "Turn, a str, an exception, or a ModelResponse."
                )
        self.script: list[ScriptItem] = list(script)
        #: Every request received, in order: ``requests[i]`` was answered by
        #: ``script[i]``.
        self.requests: list[ModelRequest] = []
        self._call_ids = 0

    def assert_exhausted(self) -> None:
        """Fail unless every scripted turn was played: a run that stopped
        early can pass its other assertions by accident."""
        unused = len(self.script) - len(self.requests)
        if unused > 0:
            raise AssertionError(
                f"ScriptedModel played {len(self.requests)} of {len(self.script)} "
                f"scripted turns; script[{len(self.requests)}:] never ran."
            )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response, error = self._play(request)
        if error is not None:
            raise error
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        response, error = self._play(request)
        for chunk in _deltas(response.text):
            yield ModelTextDelta(text=chunk)
        if error is not None:
            raise error
        yield ModelResponseComplete(response=response)

    def __repr__(self) -> str:
        return f"ScriptedModel(played={len(self.requests)}, turns={len(self.script)})"

    # ------------------------------------------------------------ playing

    def _play(self, request: ModelRequest) -> tuple[ModelResponse, Optional[BaseException]]:
        index = len(self.requests)
        # A snapshot: the loop keeps building on the objects it sent.
        self.requests.append(request.model_copy(deep=True))
        if index >= len(self.script):
            raise ScriptFailure(
                f"ScriptedModel script exhausted: requests[{index}] has no turn "
                f"(the script has {len(self.script)}).\n{_describe_request(request)}"
            )
        item = self.script[index]
        if isinstance(item, ModelResponse):
            return item.model_copy(deep=True), None
        if isinstance(item, str):
            turn = Turn(text=item)
        elif isinstance(item, BaseException):
            turn = Turn(error=item)
        else:
            turn = item
        if turn.expect is not None:
            _check(turn.expect, request, index)
        return self._respond(turn, request, index), turn.error

    def _respond(self, turn: Turn, request: ModelRequest, index: int) -> ModelResponse:
        content: list[ContentBlock] = []
        if isinstance(turn.reasoning, ReasoningBlock):
            content.append(turn.reasoning)
        elif turn.reasoning:
            content.append(ReasoningBlock(provider=_PROVIDER, text=turn.reasoning))
        if turn.text:
            content.append(TextBlock(text=turn.text))
        for call in turn.tool_calls:
            content.append(
                ToolUseBlock(id=call.id or self._call_id(), name=call.name, input=dict(call.input))
            )
        if turn.output is not None:
            content.append(
                ToolUseBlock(
                    id=self._call_id(),
                    name=OUTPUT_TOOL_NAME,
                    input=_output_input(turn.output, request, index),
                )
            )
        content.extend(turn.blocks)
        stop_reason = turn.stop_reason or (
            "tool_use" if any(isinstance(b, ToolUseBlock) for b in content) else "end_turn"
        )
        usage = turn.usage.model_copy() if turn.usage is not None else Usage(requests=1)
        return ModelResponse(content=content, stop_reason=stop_reason, usage=usage)

    def _call_id(self) -> str:
        self._call_ids += 1
        return f"call_{self._call_ids}"


def _check(expect: Callable[[ModelRequest], Any], request: ModelRequest, index: int) -> None:
    try:
        verdict = expect(request)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, AssertionError) else f"{type(exc).__name__}: {exc}"
        raise ScriptFailure(
            f"ScriptedModel script[{index}] expect failed: {reason}\n{_describe_request(request)}"
        ) from exc
    if verdict is False:
        raise ScriptFailure(
            f"ScriptedModel script[{index}] expect returned False.\n{_describe_request(request)}"
        )


def _output_input(value: Any, request: ModelRequest, index: int) -> dict[str, Any]:
    """The output tool's input for a scripted ``output=`` value."""
    tool = next((t for t in request.tools if t.name == OUTPUT_TOOL_NAME), None)
    if tool is None:
        raise ScriptFailure(
            f"ScriptedModel script[{index}] scripts output=, but the request offers no "
            f"`{OUTPUT_TOOL_NAME}` tool; run the agent with a non-str output_type.\n"
            f"{_describe_request(request)}"
        )
    payload = to_jsonable_python(value)
    if _is_value_wrapper(tool.input_schema):
        return {"value": payload}
    if not isinstance(payload, dict):
        raise ScriptFailure(
            f"ScriptedModel script[{index}] output= must be an object (a model instance or "
            f"a dict) for this output type, not {type(value).__name__}."
        )
    return payload


def _is_value_wrapper(schema: dict[str, Any]) -> bool:
    """Whether an output schema is the ``{"value": ...}`` wrapper the agent
    builds for non-object output types (see ``agent._OutputSpec``). Pydantic
    titles every model, dataclass and TypedDict schema; the wrapper alone is
    an untitled object whose only property is ``value``."""
    return "title" not in schema and list(schema.get("properties", {})) == ["value"]


def _deltas(text: str) -> list[str]:
    """``text`` in word-sized chunks (each word with its trailing space)."""
    return re.findall(r"\S+\s*|\s+", text)


def _describe_request(request: ModelRequest) -> str:
    """A compact, readable summary of a request, for failure messages."""
    choice = request.tool_choice
    choice_text = choice if isinstance(choice, str) else f"force {choice.name}"
    tools = ", ".join(t.name for t in request.tools) or "none"
    lines = [f"The request: model={request.model!r}, tool_choice={choice_text}, tools=[{tools}]"]
    if request.system:
        lines.append(f"  system: {_clip(system_text(request.system))}")
    for i, message in enumerate(request.messages):
        blocks = "; ".join(_describe_block(b) for b in message.content) or "(empty)"
        lines.append(f"  messages[{i}] {message.role}: {blocks}")
    return "\n".join(lines)


def _describe_block(block: Any) -> str:
    if isinstance(block, TextBlock):
        return f"text {_clip(block.text)}"
    if isinstance(block, ToolUseBlock):
        return f"tool_use {block.id} {block.name} {_clip(json.dumps(block.input, default=str))}"
    if isinstance(block, ToolResultBlock):
        error = " (error)" if block.is_error else ""
        return f"tool_result {block.tool_use_id}{error} {_clip(content_text(block.content))}"
    if isinstance(block, ReasoningBlock):
        return f"reasoning {_clip(block.text)}"
    return str(getattr(block, "type", type(block).__name__))


def _clip(text: str, limit: int = 200) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")
