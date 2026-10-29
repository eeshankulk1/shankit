"""Internal: what the agent loop does to keep a run inside the context window.

Plain functions over the transcript and tool results, so the loop
(``Agent._loop``) reads as control flow: spill an oversized result to the
workspace, cap one, clear older results and images, drop reasoning an edit
invalidated, and answer tool calls that never ran so the transcript stays
valid to continue.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any, Optional

from .messages import (
    ImageBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    content_text,
    user_message,
)
from .tools.base import ToolResult
from .workspace.base import Workspace

__all__ = [
    "append_user_text",
    "cap_result",
    "clear_images",
    "clear_results",
    "close_interrupted",
    "spill",
    "unrun_results",
]

logger = logging.getLogger("shankit")

# Spilled results keep a head and a tail of the output inline.
_SPILL_HEAD_CHARS = 1500
_SPILL_TAIL_CHARS = 500
# Context clearing leaves this much of each cleared result in place, and
# doesn't bother with results already smaller than this.
_CLEAR_KEEP_CHARS = 200
_CLEAR_MIN_CHARS = 1000
# What an image dropped by keep_recent_images leaves behind.
_IMAGE_CLEARED = "[older image removed to save context]"


async def spill(
    tool_use: ToolUseBlock,
    result: ToolResult,
    *,
    workspace: Optional[Workspace],
    threshold: Optional[int],
    spilled: Optional[dict[str, str]],
    agent_name: str,
) -> ToolResult:
    """Write an oversized result to a workspace file and hand the model
    its head, tail, and path instead (``spill_threshold_chars``). Falls
    through unchanged — to the lossy ``max_tool_result_chars`` cap —
    when there's no workspace or the write fails."""
    if (
        workspace is None
        or threshold is None
        or not isinstance(result.content, str)
        or len(result.content) <= threshold
    ):
        return result
    path = await _save_output(workspace, tool_use.name, result.content)
    if path is None:
        return result
    if spilled is not None:
        spilled[tool_use.id] = path
    content = result.content
    preview = (
        content[:_SPILL_HEAD_CHARS]
        + f"\n\n… [{len(content) - _SPILL_HEAD_CHARS - _SPILL_TAIL_CHARS} characters omitted] …\n\n"
        + content[-_SPILL_TAIL_CHARS:]
        + f"\n\n[This output was {len(content)} characters, so it was saved to {path}. "
        "Search or slice that file (rg, jq, head, python3, or Read with offset/limit) "
        "instead of reading it all.]"
    )
    logger.info(
        "Agent %r: tool %r returned %d chars; spilled to %s.",
        agent_name,
        tool_use.name,
        len(content),
        path,
    )
    return result.model_copy(update={"content": preview})


def cap_result(
    result: ToolResult, *, cap: Optional[int], tool_name: str, agent_name: str
) -> ToolResult:
    """Apply ``max_tool_result_chars`` (loop safety bound, like
    ``max_iterations``): one oversized result must not be able to exceed
    the model's context window and kill the run. Applied at the loop's one
    choke point so every source — local functions, MCP, connectors,
    sub-agents, and ``ToolError`` text — is covered uniformly. The capped
    content is what the model, the trajectory, and evals all see."""
    if cap is None or len(content_text(result.content)) <= cap:
        return result
    if not isinstance(result.content, str):
        # Blocks: spend the cap across the text blocks in emission order
        # (images and provider blocks are bounded by their producers, not
        # touched here) — capping each block independently would let a
        # result with several under-cap text blocks sail past the cap in
        # aggregate, which is exactly the failure this guards against.
        remaining = cap
        capped: list[Any] = []
        for b in result.content:
            if not isinstance(b, TextBlock):
                capped.append(b)
            elif remaining <= 0:
                continue
            elif len(b.text) > remaining:
                capped.append(
                    b.model_copy(
                        update={"text": b.text[:remaining] + f"… [truncated at {cap} characters]"}
                    )
                )
                remaining = 0
            else:
                capped.append(b)
                remaining -= len(b.text)
        return result.model_copy(update={"content": capped, "truncated": True})
    logger.warning(
        "Agent %r: tool %r returned %d chars; capped at max_tool_result_chars=%d.",
        agent_name,
        tool_name,
        len(result.content),
        cap,
    )
    # Copy, don't mutate: the source may retain its ToolResult object.
    return result.model_copy(
        update={
            "content": result.content[:cap]
            + f"… [truncated: tool result exceeded {cap} characters]",
            "truncated": True,
        }
    )


async def clear_results(
    messages: list[Message],
    *,
    keep: int,
    cleared: set[str],
    spilled: dict[str, str],
    workspace: Optional[Workspace],
    agent_name: str,
) -> bool:
    """Stub out all but the newest ``keep`` tool results (``context_clear_*``).

    Rewrites ``messages`` in place with new Message objects (never
    mutates the originals, which may be the caller's history). Returns
    whether anything was cleared."""
    positions = [
        (mi, bi)
        for mi, message in enumerate(messages)
        for bi, block in enumerate(message.content)
        if isinstance(block, ToolResultBlock)
    ]
    candidates = positions[:-keep] if keep else positions
    changed: dict[int, list[Any]] = {}
    count = 0
    for mi, bi in candidates:
        block = messages[mi].content[bi]
        assert isinstance(block, ToolResultBlock)
        text = content_text(block.content)
        if block.tool_use_id in cleared or len(text) < _CLEAR_MIN_CHARS:
            continue
        path = spilled.get(block.tool_use_id)
        if path is None and workspace is not None:
            path = await _save_output(workspace, "cleared", text)
        where = f" Full output: {path}." if path else ""
        stub = text[:_CLEAR_KEEP_CHARS] + f"… [older tool output cleared to save context.{where}]"
        blocks = changed.setdefault(mi, list(messages[mi].content))
        blocks[bi] = block.model_copy(update={"content": stub})
        cleared.add(block.tool_use_id)
        count += 1
    if not count:
        return False
    for mi, blocks in changed.items():
        messages[mi] = messages[mi].model_copy(update={"content": blocks})
    _drop_reasoning(messages)
    logger.info("Agent %r: cleared %d older tool results from context.", agent_name, count)
    return True


def clear_images(messages: list[Message], keep: int) -> bool:
    """``keep_recent_images``: once tool results hold ``2 * keep`` images,
    replace all but the newest ``keep`` with a stub. Rewrites ``messages``
    in place with new Message objects (never mutating the originals);
    returns whether anything changed."""
    spots = [
        (mi, bi, ci)
        for mi, message in enumerate(messages)
        for bi, block in enumerate(message.content)
        if isinstance(block, ToolResultBlock) and not isinstance(block.content, str)
        for ci, item in enumerate(block.content)
        if isinstance(item, ImageBlock)
    ]
    if len(spots) < 2 * keep:
        return False
    changed: dict[int, list[Any]] = {}
    for mi, bi, ci in spots[:-keep]:
        blocks = changed.setdefault(mi, list(messages[mi].content))
        block = blocks[bi]
        items = list(block.content)
        items[ci] = TextBlock(text=_IMAGE_CLEARED)
        blocks[bi] = block.model_copy(update={"content": items})
    for mi, blocks in changed.items():
        messages[mi] = messages[mi].model_copy(update={"content": blocks})
    _drop_reasoning(messages)
    logger.info("Cleared %d older images from context.", len(spots) - keep)
    return True


def append_user_text(messages: list[Message], text: str) -> None:
    """Add ``text`` for the model after the last message: into it when it
    is already the user's (a pass's tool results, or the prompt), else as a
    new user turn. Never mutates the message objects themselves."""
    if messages and messages[-1].role == "user":
        last = messages[-1]
        messages[-1] = last.model_copy(update={"content": [*last.content, TextBlock(text=text)]})
    else:
        messages.append(user_message(text))


def close_interrupted(messages: list[Message], text: str) -> list[Message]:
    """A transcript whose last pass called tools that never returned gets
    an error result for each, so the conversation is valid to continue."""
    if not messages or messages[-1].role != "assistant":
        return messages
    in_flight = [b for b in messages[-1].content if isinstance(b, ToolUseBlock)]
    return [*messages, unrun_results(in_flight, text)] if in_flight else messages


def unrun_results(tool_uses: Sequence[ToolUseBlock], text: str) -> Message:
    """An error result saying ``text`` for each of ``tool_uses`` that never
    ran: every tool call must be answered for the transcript to be valid."""
    return Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id=tu.id, content=text, is_error=True) for tu in tool_uses
        ],
    )


def _drop_reasoning(messages: list[Message]) -> None:
    """Edited history invalidates the reasoning after the edit, and a
    reasoning block is only valid in the exact conversation that produced
    it - drop them all rather than send stale ones."""
    for mi, message in enumerate(messages):
        if any(isinstance(b, ReasoningBlock) for b in message.content):
            blocks = [b for b in message.content if not isinstance(b, ReasoningBlock)]
            messages[mi] = message.model_copy(update={"content": blocks})


async def _save_output(workspace: Workspace, tool_name: str, content: str) -> Optional[str]:
    """Save a tool output to ``<tmp>/tool-output/``; the path, or ``None``
    if the workspace refused the write."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:40] or "tool"
    path = f"{workspace.tmp.rstrip('/')}/tool-output/{safe}-{uuid.uuid4().hex[:8]}.txt"
    try:
        await workspace.write_file(path, content.encode("utf-8"))
    except Exception:
        logger.exception("Could not save tool output to the workspace; falling back to the cap.")
        return None
    return path
