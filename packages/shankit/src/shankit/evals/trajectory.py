"""Trajectory assertions (design §7.2).

Scorers over *how* the agent got to its answer — the recorded tool calls in
``RunResult.trajectory`` — rather than the answer itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..agent import RunResult
from .scorers import Score, Scorer

__all__ = ["used_tool", "did_not_use_tool", "max_tool_calls", "tool_order", "no_tool_errors"]


def _tools(result: RunResult) -> list[str]:
    return [record.tool for record in result.trajectory]


def used_tool(name: str) -> Scorer:
    def scorer(case: Any, result: RunResult) -> Score:
        passed = name in _tools(result)
        return Score(
            name=f"used_tool:{name}",
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=None if passed else f"tools called: {_tools(result) or 'none'}",
        )

    scorer.__name__ = f"used_tool:{name}"
    return scorer


def did_not_use_tool(name: str) -> Scorer:
    def scorer(case: Any, result: RunResult) -> Score:
        passed = name not in _tools(result)
        return Score(name=f"did_not_use_tool:{name}", value=1.0 if passed else 0.0, passed=passed)

    scorer.__name__ = f"did_not_use_tool:{name}"
    return scorer


def max_tool_calls(limit: int) -> Scorer:
    def scorer(case: Any, result: RunResult) -> Score:
        count = len(result.trajectory)
        passed = count <= limit
        return Score(
            name=f"max_tool_calls:{limit}",
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"{count} tool call(s)",
        )

    scorer.__name__ = f"max_tool_calls:{limit}"
    return scorer


def tool_order(names: Sequence[str]) -> Scorer:
    """Pass iff the given tool names appear in the trajectory in order
    (as a subsequence — other calls may be interleaved)."""

    def scorer(case: Any, result: RunResult) -> Score:
        remaining = list(names)
        for called in _tools(result):
            if remaining and called == remaining[0]:
                remaining.pop(0)
        passed = not remaining
        return Score(
            name=f"tool_order:{'>'.join(names)}",
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=None if passed else f"missing (in order): {remaining}",
        )

    scorer.__name__ = "tool_order"
    return scorer


def no_tool_errors(case: Any, result: RunResult) -> Score:
    errors = [r.tool for r in result.trajectory if r.is_error]
    passed = not errors
    return Score(
        name="no_tool_errors",
        value=1.0 if passed else 0.0,
        passed=passed,
        reason=None if passed else f"errored tools: {errors}",
    )
