"""Delegation: one ``delegate`` tool over a roster of sub-agents.

The Claude Code sub-agent model on shankit's primitives: the parent agent
sees ONE tool whose description lists the available sub-agents (name + what
each is for), picks one, and hands it a self-contained task. Adding a
sub-agent grows the description, not the tool list. Sub-agents are usually
file-defined (``agents/*.md``, see :meth:`DelegateToolSource.from_directory`)
and run with the parent's per-run context, so they share whatever it
carries — the acting user, and the same workspace when their
``Agent(workspace=)`` resolves from it — and can hand back a file path
instead of a wall of text.

What the source enforces (policy that proved necessary in production, so the
prompt doesn't have to):

- **Budget**: each sub-agent runs at most ``max_calls_per_delegate`` times
  per run; an identical task to the same sub-agent never runs twice. Over
  budget, the model gets a calm notice (not an error) so it adapts instead
  of retrying.
- **Sanitized failures**: a sub-agent that crashes returns a uniform
  "couldn't complete" result; the traceback goes to the log, never to the
  model (which might echo internals into user-facing text).
- **Live steps**: the sub-agent's steps are forwarded into the parent run's
  stream as they happen, attributed with ``agent=<sub-agent name>`` and ids
  prefixed by the parent step's id.
- **Roll-up**: the sub-agent's usage, sources, artifacts, and ``truncated``
  flag ride the tool result into the parent run's accounting.

A roster entry is a :class:`Delegate`. :class:`AgentDelegate` wraps a
shankit :class:`~shankit.Agent`; implement :class:`Delegate` directly to put
anything else behind the same tool (an agent with its own post-processing,
a remote agent).
"""

from __future__ import annotations

import abc
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from .._calls import current_tool_call
from ..events import AgentEvent, StepEvent
from ..exceptions import ToolError, ToolNotFoundError
from .base import ToolDef, ToolResult, ToolSource

if TYPE_CHECKING:
    from ..agent import Agent

__all__ = ["AgentDelegate", "Delegate", "DelegateToolSource"]

logger = logging.getLogger("shankit")

EventSink = Callable[[AgentEvent], None]


class Delegate(abc.ABC):
    """One entry in a delegation roster."""

    #: Identifier the parent model uses to pick this delegate.
    name: str
    #: What it's for — shown to the parent model in the tool description.
    description: str

    @abc.abstractmethod
    async def run(
        self,
        task: str,
        *,
        context: Any,
        arguments: Mapping[str, Any],
        on_event: EventSink,
    ) -> ToolResult:
        """Do ``task`` and return what the parent model should see.

        ``arguments`` is the full tool input (for sources configured with
        ``extra_properties``). ``on_event`` forwards this delegate's own
        events (step events are relayed into the parent stream)."""


class AgentDelegate(Delegate):
    """A shankit :class:`~shankit.Agent` as a delegate: runs it with
    ``output_type=str`` and returns its final answer text."""

    def __init__(
        self, agent: Agent, *, name: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        self.agent = agent
        self.name = name or agent.name
        self.description = description or agent.description or f"The {agent.name} agent."

    async def run(
        self,
        task: str,
        *,
        context: Any,
        arguments: Mapping[str, Any],
        on_event: EventSink,
    ) -> ToolResult:
        result = await self.agent.run(task, context=context, output_type=str, on_event=on_event)
        return ToolResult(
            content=result.output or "",
            sources=result.sources,
            artifacts=result.artifacts,
            usage=result.usage,
            truncated=result.truncated,
        )


class DelegateToolSource(ToolSource):
    """Exposes a roster of delegates through one tool.

    Args:
        delegates: :class:`Delegate`s and/or :class:`~shankit.Agent`s
            (wrapped in :class:`AgentDelegate`).
        tool_name: The tool's name (default ``delegate``).
        max_calls_per_delegate: Per-run budget per delegate.
        extra_properties: Additional JSON-schema properties on the tool's
            input (e.g. a ``mode`` your delegates branch on). They reach
            :meth:`Delegate.run` via ``arguments``.
        intro: The first paragraph of the tool description (the roster
            is appended to it).
    """

    def __init__(
        self,
        delegates: Sequence[Union[Delegate, Agent]],
        *,
        tool_name: str = "delegate",
        max_calls_per_delegate: int = 2,
        extra_properties: Optional[Mapping[str, Any]] = None,
        intro: Optional[str] = None,
    ) -> None:
        from ..agent import Agent

        roster: dict[str, Delegate] = {}
        for item in delegates:
            delegate = AgentDelegate(item) if isinstance(item, Agent) else item
            if delegate.name in roster:
                raise ValueError(f"Duplicate delegate name {delegate.name!r}")
            roster[delegate.name] = delegate
        self.delegates = roster
        self.tool_name = tool_name
        self.max_calls_per_delegate = max(1, max_calls_per_delegate)
        self.extra_properties = dict(extra_properties or {})
        self.intro = intro or (
            "Hand a self-contained task to a specialist agent and get its answer "
            "back. Give it everything it needs in `task` — it can't see this "
            "conversation. Independent tasks can go to several agents in parallel."
        )

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        *,
        agent_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> DelegateToolSource:
        """A roster of every ``*.md`` agent file in ``directory`` (see
        :func:`~shankit.load_agents`; ``agent_kwargs`` are passed through,
        e.g. ``{"workspace": ...}`` so sub-agents share the parent's
        workspace)."""
        from ..files import load_agents

        agents = load_agents(directory, agent_kwargs=agent_kwargs)
        return cls(list(agents.values()), **kwargs)

    # ----------------------------------------------------------- the seam

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        if not self.delegates:
            return []
        roster = "\n".join(f"- {d.name}: {d.description}" for d in self.delegates.values())
        return [
            ToolDef(
                name=self.tool_name,
                description=f"{self.intro}\n\nAvailable agents:\n{roster}",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": list(self.delegates),
                            "description": "Which agent to hand the task to.",
                        },
                        "task": {
                            "type": "string",
                            "description": "The complete, self-contained task.",
                        },
                        **self.extra_properties,
                    },
                    "required": ["agent", "task"],
                },
            )
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        if name != self.tool_name:
            raise ToolNotFoundError(name)
        agent_name = str(arguments.get("agent") or "").strip()
        delegate = self.delegates.get(agent_name)
        if delegate is None:
            raise ToolError(
                f"No agent named {agent_name!r}. Choose one of: {', '.join(self.delegates)}."
            )
        task = str(arguments.get("task") or "").strip()
        if not task:
            raise ToolError("Provide the `task` for the agent.")

        call = current_tool_call()
        # Budget state lives with the run (not on this source), so one
        # source can serve many runs. Outside a loop (a direct call) there
        # is no run to budget.
        state = (
            call.run_state.setdefault(id(self), {"counts": {}, "tasks": set()})
            if call is not None
            else {"counts": {}, "tasks": set()}
        )
        key = (delegate.name, " ".join(task.lower().split()))
        if key in state["tasks"]:
            return _notice(
                f"(The {delegate.name} agent was already given exactly this task — use "
                "its earlier answer. Ask something different or move on.)"
            )
        if state["counts"].get(delegate.name, 0) >= self.max_calls_per_delegate:
            return _notice(
                f"(The {delegate.name} agent has already run "
                f"{self.max_calls_per_delegate} times this turn — work with what you "
                "have. Don't mention internal limits to the user; offer to dig "
                "further in a follow-up if something is missing.)"
            )
        state["tasks"].add(key)
        state["counts"][delegate.name] = state["counts"].get(delegate.name, 0) + 1

        def on_event(event: AgentEvent) -> None:
            if call is None or not isinstance(event, StepEvent):
                return
            call.emit(
                event.model_copy(
                    update={
                        "id": f"{call.step_id}.{event.id}",
                        "agent": event.agent or delegate.name,
                    }
                )
            )

        try:
            return await delegate.run(task, context=context, arguments=arguments, on_event=on_event)
        except Exception:
            logger.exception("Delegate %r failed", delegate.name)
            return ToolResult(
                content=(
                    f"The {delegate.name} agent couldn't complete the task because of a "
                    "temporary problem. Say that part is unavailable right now and "
                    "help with the rest; don't invent an answer or mention internal "
                    "details."
                ),
                is_error=True,
            )


def _notice(text: str) -> ToolResult:
    return ToolResult(content=json.dumps({"notice": text}))
