"""EXPERIMENTAL — the graph/network escape hatch (design §9).

For the case agents-as-tools does *not* cover: when **code, not the model,
controls flow** — deterministic routing, cycles, state accumulated across
steps, and pauses for human approval. Reaching for this merely to chain two
agents is a smell; hand the sub-agents to a parent agent instead.

Shape: a **router-driven loop over shared state**, not a declared node/edge
graph. The loop repeatedly calls a router that reads the state and returns
the next step name, an :class:`Interrupt` to pause for a human, or ``None``
to stop. A step is anything callable that touches state — an adapted agent
(:func:`agent_step`), a plain function, or a nested :class:`Network`.

The state the router reads and writes **is** what the checkpointer persists
(one state concept, design §7.1); every router decision is a checkpoint
boundary, which is why human-in-the-loop falls out of the design::

    net = Network(
        name="act-with-approval",
        steps={
            "gather": agent_step(researcher, prompt="Gather context for: {task}", output_key="notes"),
            "draft": agent_step(writer, prompt="Draft the email.\\n{notes}", output_key="draft"),
            "act": send_email_step,
        },
        router=route,
        checkpointer=SqliteCheckpointer("threads.db"),
    )
    result = await net.run({"task": "..."}, thread_id="t1")
    if result.status == "interrupted":
        ...  # show result.interrupt to a human
        result = await net.resume("t1", value=True)
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from pydantic import BaseModel

from ..agent import Agent
from ..durability import Checkpoint, Checkpointer, InterruptInfo
from ..exceptions import ShankitError, ToolNotFoundError
from ..tools.base import ToolDef, ToolResult, ToolSource

__all__ = ["Interrupt", "Network", "NetworkResult", "agent_step"]

logger = logging.getLogger("shankit")

# A step: (state, context) -> optional dict merged into state.
StepFn = Callable[[dict, Any], Union[Optional[dict], Awaitable[Optional[dict]]]]
RouterDecision = Union[str, "Interrupt", None]
Router = Callable[[dict], Union[RouterDecision, Awaitable[RouterDecision]]]


@dataclass
class Interrupt:
    """Returned by a router to pause the run for a human.

    On :meth:`Network.resume`, the supplied value is written into the state
    under ``key`` and routing continues.
    """

    reason: str
    key: str = "approval"
    payload: Any = None


@dataclass
class NetworkResult:
    status: str  # "done" | "interrupted"
    state: dict[str, Any]
    interrupt: Optional[Interrupt] = None
    thread_id: Optional[str] = None
    steps_run: list[str] = field(default_factory=list)


class Network:
    """A router-driven loop over shared state. EXPERIMENTAL.

    A network is itself runnable like an agent, usable as a step in another
    network (it is a valid ``StepFn``), and exposable as a tool via
    :meth:`as_tool` — deterministic sub-flows and normal agent conversations
    compose in both directions.
    """

    def __init__(
        self,
        *,
        name: str = "network",
        steps: Mapping[str, Union[StepFn, Network]],
        router: Router,
        checkpointer: Optional[Checkpointer] = None,
        max_steps: int = 50,
        description: Optional[str] = None,
    ) -> None:
        for step_name, step in steps.items():
            if isinstance(step, Agent):
                raise TypeError(
                    f"Step {step_name!r} is an Agent; adapt it with "
                    "agent_step(agent, prompt=..., output_key=...) so the network "
                    "knows how it reads and writes state."
                )
            if not callable(step):
                raise TypeError(f"Step {step_name!r} is not callable.")
        self.name = name
        self.steps = dict(steps)
        self.router = router
        self.checkpointer = checkpointer
        self.max_steps = max_steps
        self.description = description

    async def run(
        self,
        state: Optional[dict[str, Any]] = None,
        *,
        context: Any = None,
        thread_id: Optional[str] = None,
    ) -> NetworkResult:
        """Run the router loop until it stops or interrupts."""
        if self.checkpointer is not None and thread_id is None:
            raise ShankitError(
                f"Network {self.name!r} has a checkpointer; pass thread_id= so the "
                "run has durable per-thread identity."
            )
        working = dict(state or {})
        return await self._drive(working, context=context, thread_id=thread_id, steps_run=[])

    async def resume(
        self,
        thread_id: str,
        value: Any = None,
        *,
        context: Any = None,
    ) -> NetworkResult:
        """Resume an interrupted thread, feeding ``value`` into the state
        under the interrupt's key."""
        if self.checkpointer is None:
            raise ShankitError(f"Network {self.name!r} has no checkpointer to resume from.")
        checkpoint = await self.checkpointer.load(thread_id)
        if checkpoint is None:
            raise ShankitError(f"No checkpoint found for thread {thread_id!r}.")
        if checkpoint.status != "interrupted" or checkpoint.interrupt is None:
            raise ShankitError(
                f"Thread {thread_id!r} is not awaiting input (status={checkpoint.status!r})."
            )
        working = dict(checkpoint.state)
        working[checkpoint.interrupt.key] = value
        return await self._drive(
            working, context=context, thread_id=thread_id, steps_run=list(checkpoint.steps_run)
        )

    async def recover(
        self,
        thread_id: str,
        *,
        context: Any = None,
        retry_in_flight: bool = False,
    ) -> NetworkResult:
        """Re-enter a ``running`` thread after a crash or a step failure.

        The state is re-loaded from the last checkpoint and the router loop
        continues; because routing is a function of state, the router will
        re-derive whatever step never completed.

        If the checkpoint has an ``in_flight`` step, the process died (or the
        step raised) *mid-step* — its side effects may have partially
        happened, so re-running it is only safe if the step is idempotent.
        This method refuses that case unless ``retry_in_flight=True``, by
        which the caller asserts exactly that.

        Threads that are ``interrupted`` resume via :meth:`resume` (they are
        waiting on input, not crashed); ``done`` threads have nothing to
        recover.
        """
        if self.checkpointer is None:
            raise ShankitError(f"Network {self.name!r} has no checkpointer to recover from.")
        checkpoint = await self.checkpointer.load(thread_id)
        if checkpoint is None:
            raise ShankitError(f"No checkpoint found for thread {thread_id!r}.")
        if checkpoint.status == "interrupted":
            raise ShankitError(
                f"Thread {thread_id!r} is awaiting input, not crashed; use resume()."
            )
        if checkpoint.status == "done":
            raise ShankitError(f"Thread {thread_id!r} already finished; nothing to recover.")
        if checkpoint.in_flight is not None and not retry_in_flight:
            raise ShankitError(
                f"Thread {thread_id!r} died while step {checkpoint.in_flight!r} was in "
                "flight; its side effects may have partially happened. Pass "
                "retry_in_flight=True to re-run it if (and only if) the step is idempotent."
            )
        return await self._drive(
            dict(checkpoint.state),
            context=context,
            thread_id=thread_id,
            steps_run=list(checkpoint.steps_run),
        )

    async def _drive(
        self,
        state: dict[str, Any],
        *,
        context: Any,
        thread_id: Optional[str],
        steps_run: list[str],
    ) -> NetworkResult:
        for _ in range(self.max_steps):
            decision = await _maybe_await(self.router(state))

            if decision is None:
                await self._save(thread_id, state, "done", None, steps_run)
                return NetworkResult(
                    status="done", state=state, thread_id=thread_id, steps_run=steps_run
                )

            if isinstance(decision, Interrupt):
                info = InterruptInfo(
                    reason=decision.reason, key=decision.key, payload=decision.payload
                )
                await self._save(thread_id, state, "interrupted", info, steps_run)
                return NetworkResult(
                    status="interrupted",
                    state=state,
                    interrupt=decision,
                    thread_id=thread_id,
                    steps_run=steps_run,
                )

            step = self.steps.get(decision)
            if step is None:
                known = ", ".join(sorted(self.steps))
                raise ShankitError(
                    f"Router of network {self.name!r} returned unknown step "
                    f"{decision!r}. Steps: {known}."
                )
            # Write-ahead marker: if the process dies (or the step raises)
            # past this point, the checkpoint records which step was in
            # flight so recover() can tell "between steps" from "mid-step".
            await self._save(thread_id, state, "running", None, steps_run, in_flight=decision)
            update = await _maybe_await(step(state, context))
            if update is not None:
                if not isinstance(update, dict):
                    raise ShankitError(
                        f"Step {decision!r} returned {type(update).__name__}; steps "
                        "return a dict of state updates or None."
                    )
                state.update(update)
            steps_run.append(decision)
            # A completed step = a router boundary = a checkpoint (design §7.1).
            await self._save(thread_id, state, "running", None, steps_run)

        raise ShankitError(
            f"Network {self.name!r} hit max_steps={self.max_steps} without stopping."
        )

    async def _save(
        self,
        thread_id: Optional[str],
        state: dict[str, Any],
        status: str,
        interrupt: Optional[InterruptInfo],
        steps_run: list[str],
        in_flight: Optional[str] = None,
    ) -> None:
        if self.checkpointer is None or thread_id is None:
            return
        await self.checkpointer.save(
            thread_id,
            Checkpoint(
                state=state,
                status=status,  # type: ignore[arg-type]
                interrupt=interrupt,
                steps_run=list(steps_run),
                in_flight=in_flight,
            ),
        )

    # A Network is a valid StepFn: nested networks share the parent's state.
    async def __call__(self, state: dict[str, Any], context: Any = None) -> Optional[dict]:
        result = await self._drive(state, context=context, thread_id=None, steps_run=[])
        if result.status == "interrupted":
            raise ShankitError(
                f"Nested network {self.name!r} interrupted; interrupts are only "
                "supported on the outermost network (give it the checkpointer)."
            )
        return result.state

    def as_tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_key: str = "task",
        output_key: Optional[str] = None,
    ) -> ToolSource:
        """Expose this network as a single tool.

        The model's ``task`` string seeds the state under ``input_key``; the
        result is ``state[output_key]`` (or the full final state as JSON).
        """
        return _NetworkToolSource(
            self, name=name, description=description, input_key=input_key, output_key=output_key
        )


class _NetworkToolSource(ToolSource):
    def __init__(
        self,
        network: Network,
        *,
        name: Optional[str],
        description: Optional[str],
        input_key: str,
        output_key: Optional[str],
    ) -> None:
        self.network = network
        self.tool_name = name or network.name.replace(" ", "_")
        self.description = (
            description or network.description or f"Run the {network.name} workflow on a task."
        )
        self.input_key = input_key
        self.output_key = output_key

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        return [
            ToolDef(
                name=self.tool_name,
                description=self.description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The task, in plain language."}
                    },
                    "required": ["task"],
                },
            )
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        if name != self.tool_name:
            raise ToolNotFoundError(name)
        try:
            result = await self.network.run(
                {self.input_key: arguments.get("task", "")}, context=context
            )
        except Exception:
            logger.exception("Network %r failed as a tool", self.network.name)
            return ToolResult(
                content=f"The {self.network.name} workflow could not complete the task.",
                is_error=True,
            )
        if result.status == "interrupted":
            return ToolResult(
                content=(
                    f"The {self.network.name} workflow paused for human input: "
                    f"{result.interrupt.reason if result.interrupt else ''}"
                ),
                is_error=True,
            )
        if self.output_key is not None:
            value = result.state.get(self.output_key)
            return ToolResult(
                content=value if isinstance(value, str) else json.dumps(value, default=str)
            )
        return ToolResult(content=json.dumps(result.state, default=str))


def agent_step(
    agent: Agent,
    *,
    prompt: Union[str, Callable[[dict], str]],
    output_key: str,
    output_type: Optional[type] = None,
) -> StepFn:
    """Adapt an agent into a network step.

    ``prompt`` is a fill-in template over the state (``"Draft: {notes}"``)
    or a function of the state. The agent's output lands in
    ``state[output_key]`` (pydantic outputs are stored as plain dicts so the
    state stays checkpointable).
    """

    async def step(state: dict[str, Any], context: Any = None) -> dict[str, Any]:
        rendered = prompt(state) if callable(prompt) else prompt.format_map(_Missing(state))
        result = await agent.run(rendered, context=context, output_type=output_type or str)
        output = result.output
        if isinstance(output, BaseModel):
            output = output.model_dump(mode="json")
        return {output_key: output}

    step.__name__ = f"{agent.name}_step"
    return step


class _Missing(dict):
    def __missing__(self, key: str) -> str:
        raise ShankitError(
            f"agent_step prompt placeholder {{{key}}} is not present in the network state."
        )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
