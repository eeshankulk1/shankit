"""The tool-call context: what a running tool can see of the run it is in.

The tool seam (``list_tools``/``execute``) stays two methods over the opaque
per-run context. This is the one extra channel, and it is optional: while a
tool executes, :func:`current_tool_call` returns the call it is serving, so
a long-running tool can push events into the parent run's stream *while it
works* (a delegated sub-agent's steps, a script's progress) instead of only
when it returns, and can keep per-run state (a budget, a dedup set) without
the source being rebuilt per run.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["ToolCallContext", "current_tool_call"]


@dataclass
class ToolCallContext:
    """One executing tool call.

    Attributes:
        tool_use_id: The model's id for this call.
        tool_name: The tool being executed.
        step_id: The loop's step id for this call (``s1``, ``s2``, ...) —
            prefix forwarded sub-step ids with it so they stay unique.
        agent_name: The agent whose loop is executing the call.
        emit: Push an event into the run's stream now. Events are yielded
            in the order emitted, before the call's own completion events.
        run_state: A dict shared by every tool call of this run and
            discarded with it. Key your entries by something unique to
            your source (e.g. ``id(self)``).
    """

    tool_use_id: str
    tool_name: str
    step_id: str
    agent_name: str
    emit: Callable[[Any], None]
    run_state: dict[Any, Any] = field(default_factory=dict)


_CURRENT: contextvars.ContextVar[Optional[ToolCallContext]] = contextvars.ContextVar(
    "shankit_tool_call", default=None
)


def current_tool_call() -> Optional[ToolCallContext]:
    """The tool call being executed on this task, or ``None`` outside one."""
    return _CURRENT.get()


def _set_current(call: Optional[ToolCallContext]) -> contextvars.Token[Optional[ToolCallContext]]:
    return _CURRENT.set(call)
