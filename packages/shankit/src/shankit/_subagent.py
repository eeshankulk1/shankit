"""Internal: running a child agent under a parent's tool call.

What :meth:`Agent.as_tool <shankit.Agent.as_tool>`, the ``delegate`` tool
(:mod:`shankit.tools.delegate`) and clones (:mod:`shankit.tools.clones`)
share: the child's steps relayed live into the parent run's stream, and
its usage, sources, artifacts and ``truncated`` flag rolled up into the
tool result. Each keeps its own failure text and policy (the delegate
roster's budget, clone batches and depth).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional

from ._calls import ToolCallContext, current_tool_call
from ._serialize import dump_str
from .events import AgentEvent, StepEvent
from .exceptions import ToolError, ToolNotFoundError
from .tools.base import ToolDef, ToolResult, ToolSource, single_task_tool_def

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["AgentToolSource", "relay_steps", "run_sub_agent"]

logger = logging.getLogger("shankit")


def relay_steps(
    call: Optional[ToolCallContext],
    *,
    via: Optional[str] = None,
    agent: Optional[str] = None,
    worker: Optional[str] = None,
) -> Callable[[AgentEvent], None]:
    """An ``on_event`` for a child run that relays its steps into the
    parent call's stream as they happen: ids prefixed by the call's step id
    (then ``via``, when given), attributed ``agent=`` / ``worker=`` unless
    the child already set them. Outside a tool call (a direct call) there is
    no stream to relay into."""

    def on_event(event: AgentEvent) -> None:
        if call is None or not isinstance(event, StepEvent):
            return
        prefix = f"{call.step_id}.{via}" if via is not None else call.step_id
        update: dict[str, Any] = {"id": f"{prefix}.{event.id}"}
        if agent is not None:
            update["agent"] = event.agent or agent
        if worker is not None:
            update["worker"] = event.worker or worker
        call.emit(event.model_copy(update=update))

    return on_event


async def run_sub_agent(
    agent: Agent,
    task: str,
    *,
    context: Any,
    on_event: Callable[[AgentEvent], None],
    structured: bool = False,
) -> ToolResult:
    """Run ``agent`` on ``task`` and roll its run up into a tool result:
    its final answer text (or, ``structured``, its default schema's result
    as JSON) plus its usage, sources, artifacts and ``truncated`` flag."""
    if structured:
        result = await agent.run(task, context=context, on_event=on_event)
        content = dump_str(result.output)
    else:
        result = await agent.run(task, context=context, output_type=str, on_event=on_event)
        content = result.output or ""
    return ToolResult(
        content=content,
        sources=result.sources,
        artifacts=result.artifacts,
        usage=result.usage,
        truncated=result.truncated,
    )


class AgentToolSource(ToolSource):
    """What :meth:`Agent.as_tool <shankit.Agent.as_tool>` returns."""

    def __init__(
        self, agent: Agent, *, name: Optional[str], description: Optional[str], output: str
    ) -> None:
        self.agent = agent
        self.tool_name = re.sub(r"[^a-zA-Z0-9_-]", "_", (name or agent.name).strip()) or "agent"
        self.description = (
            description
            or agent.description
            or f"Delegate a task to the {agent.name} agent. Describe the task in plain language."
        )
        self.output = output

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        return [single_task_tool_def(self.tool_name, self.description)]

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        if name != self.tool_name:
            raise ToolNotFoundError(name)
        task = str(arguments.get("task", "")).strip()
        if not task:
            raise ToolError("Provide a 'task' describing what this agent should do.")
        try:
            return await run_sub_agent(
                self.agent,
                task,
                context=context,
                on_event=relay_steps(current_tool_call(), agent=self.agent.name),
                structured=self.output == "structured",
            )
        except Exception:
            # Sanitized sub-agent failure: the parent model gets a calm,
            # uniform message; the real traceback goes to the logs.
            logger.exception("Sub-agent %r failed", self.agent.name)
            return ToolResult(
                content=f"The {self.agent.name} agent could not complete the task.",
                is_error=True,
            )
