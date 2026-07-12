"""``max_tool_result_chars`` — the tool-result size safety bound.

Spec under test (motivated by a throu production incident: a sub-agent
deep-fetched nine un-slimmed HTML emails and the run died at 244,716
tokens > the 200K context limit, surfacing to the end user as "Gmail is
unavailable"):

- With the cap set, any single tool result's content is cut at the cap,
  a truncation marker is appended, and the result is marked truncated —
  which feeds the run's sticky ``truncated`` flag.
- The capped content is what the model sees (conversation) AND what the
  trajectory records — evals must score against what the model saw.
- Applied uniformly at the loop's choke point: plain-``str`` returns and
  ``ToolError`` text are covered, not just ``ToolResult`` returns.
- Default (``None``) changes nothing; under-cap content passes through
  byte-identical.
"""

from __future__ import annotations

import pytest
from conftest import text_response, tool_call_response
from shankit import ToolError, tool
from shankit.messages import ToolResultBlock

CAP = 200
MARKER = f"… [truncated: tool result exceeded {CAP} characters]"


@tool
def fetch_blob(size: int) -> str:
    """Return a payload of `size` x's."""
    return "x" * size


@tool
def failing(size: int) -> str:
    """Raise a controlled error with a `size`-char message."""
    raise ToolError("e" * size)


def _tool_result_content(fake) -> str:
    """The tool-result content the model actually saw on the follow-up turn."""
    followup = fake.requests[1].messages[-1]
    blocks = [b for b in followup.content if isinstance(b, ToolResultBlock)]
    assert len(blocks) == 1
    return blocks[0].content


async def test_oversized_result_is_capped_marked_and_sticky(make_agent):
    agent, fake = make_agent(
        [tool_call_response("fetch_blob", {"size": 10_000}), text_response("done")],
        tools=[fetch_blob],
        max_tool_result_chars=CAP,
    )
    result = await agent.run("fetch it", output_type=str)

    seen = _tool_result_content(fake)
    assert seen == "x" * CAP + MARKER
    # the trajectory records what the model saw, not the raw payload
    assert result.trajectory[0].content == seen
    # a capped result is a truncated result: the sticky run flag fires
    assert result.truncated is True


async def test_under_cap_result_untouched_and_not_truncated(make_agent):
    agent, fake = make_agent(
        [tool_call_response("fetch_blob", {"size": CAP}), text_response("done")],
        tools=[fetch_blob],
        max_tool_result_chars=CAP,
    )
    result = await agent.run("fetch it", output_type=str)

    assert _tool_result_content(fake) == "x" * CAP
    assert result.truncated is False


async def test_no_cap_by_default(make_agent):
    agent, fake = make_agent(
        [tool_call_response("fetch_blob", {"size": 10_000}), text_response("done")],
        tools=[fetch_blob],
    )
    result = await agent.run("fetch it", output_type=str)

    assert _tool_result_content(fake) == "x" * 10_000
    assert result.truncated is False


async def test_tool_error_text_is_capped_too(make_agent):
    agent, fake = make_agent(
        [tool_call_response("failing", {"size": 10_000}), text_response("done")],
        tools=[failing],
        max_tool_result_chars=CAP,
    )
    result = await agent.run("try it", output_type=str)

    followup = fake.requests[1].messages[-1]
    block = next(b for b in followup.content if isinstance(b, ToolResultBlock))
    assert block.is_error is True
    assert block.content == "e" * CAP + MARKER
    assert result.truncated is True


def test_non_positive_cap_rejected(make_agent):
    with pytest.raises(ValueError, match="max_tool_result_chars"):
        make_agent([], max_tool_result_chars=0)
    with pytest.raises(ValueError, match="max_tool_result_chars"):
        make_agent([], max_tool_result_chars=-5)
