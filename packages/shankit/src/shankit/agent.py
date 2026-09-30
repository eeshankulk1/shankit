"""The agent primitive (design §3.2).

One agent, runnable two ways over the same underlying tool-use loop:

- :meth:`Agent.run` — **structured**: returns a typed, validated deliverable.
- :meth:`Agent.stream` — **streaming**: yields the typed event stream
  (text deltas, steps, sources, usage, done).

The two modes split on output shape, not sync-vs-async, and whether
structured output is requested is the caller's choice at call time: an agent
with a declared default schema *can* be run structured; one without can only
stream (or be run with an explicit ``output_type=``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel, TypeAdapter, ValidationError

from . import _tracing
from ._calls import ToolCallContext, _set_current, current_tool_call
from ._serialize import dump_str
from .control import RunControl
from .events import (
    AgentEvent,
    Artifact,
    ArtifactEvent,
    DoneEvent,
    Source,
    SourceEvent,
    StepEvent,
    TextDeltaEvent,
    UsageEvent,
    error_event_for,
)
from .exceptions import (
    MaxIterationsError,
    ModelError,
    OutputValidationError,
    RunTimeoutError,
    ShankitError,
    ToolError,
    ToolNotFoundError,
)
from .messages import (
    ImageBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    assistant_text,
    coerce_message,
    content_text,
    user_message,
)
from .models.base import (
    ForcedTool,
    ModelClient,
    ModelRequest,
    ModelResponseComplete,
    ModelTextDelta,
    Reasoning,
    SystemPrompt,
)
from .models.registry import resolve_model
from .observe import StepDescriber, StepInfo, default_step_describer
from .tools.aggregate import CompositeToolSource
from .tools.base import (
    ToolDef,
    ToolResult,
    Toolset,
    ToolSource,
    is_tool_source,
    single_task_tool_def,
)
from .tools.local import FunctionTool, FunctionToolSource
from .usage import Usage
from .workspace.base import Workspace, WorkspaceSpec, resolve_workspace

__all__ = ["OUTPUT_TOOL_NAME", "Agent", "RunResult", "ToolCallRecord"]

logger = logging.getLogger("shankit")

OutputT = TypeVar("OutputT")

OUTPUT_TOOL_NAME = "final_result"
_OUTPUT_NUDGE = f"Provide the final result now by calling the `{OUTPUT_TOOL_NAME}` tool."

Instructions = Union[SystemPrompt, Callable[[Any], Union[SystemPrompt, Awaitable[SystemPrompt]]]]

ReasoningSpec = Union[bool, str, Reasoning, None]

#: Why a :class:`~shankit.RunControl` ended a run early.
StopReason = Literal["budget", "cancelled"]

# Spilled results keep a head and a tail of the output inline.
_SPILL_HEAD_CHARS = 1500
_SPILL_TAIL_CHARS = 500
# Context clearing leaves this much of each cleared result in place, and
# doesn't bother with results already smaller than this.
_CLEAR_KEEP_CHARS = 200
_CLEAR_MIN_CHARS = 1000
# What an image dropped by keep_recent_images leaves behind.
_IMAGE_CLEARED = "[older image removed to save context]"


def _normalize_reasoning(value: ReasoningSpec) -> Optional[Reasoning]:
    if value is None or isinstance(value, Reasoning):
        return value
    if isinstance(value, bool):
        return Reasoning(enabled=value)
    if isinstance(value, str):
        return Reasoning(effort=value)  # type: ignore[arg-type]
    raise TypeError(f"reasoning must be a bool, an effort string, or Reasoning; got {value!r}")


class ToolCallRecord(BaseModel):
    """One tool call in a run's trajectory (used by evals, design §7.2)."""

    tool: str
    arguments: dict[str, Any]
    content: str
    is_error: bool = False


@dataclass
class RunResult(Generic[OutputT]):
    """The deliverable of a structured run.

    ``output`` is the answer: the validated structured result, or — for
    ``output_type=str`` — the *final* pass's text. ``text`` is the
    transcript: every assistant text pass of the run joined with blank
    lines, including interim acknowledgments before tool calls. Score and
    act on ``output``; persist and display ``text``. ``artifacts`` are the
    typed structured payloads tools surfaced during the run, in emission
    order (see ``ToolResult.artifacts``). ``truncated`` is sticky: true if
    *any* model pass stopped at the token limit (including a sub-agent's —
    see ``ToolResult.truncated``), since a truncated intermediate pass can
    corrupt a run as much as a truncated answer.
    """

    output: OutputT
    text: str
    usage: Usage
    trajectory: list[ToolCallRecord] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    truncated: bool = False
    #: The run's own transcript (see ``DoneEvent.messages``).
    messages: list[Message] = field(default_factory=list)
    #: Why a :class:`~shankit.RunControl` ended the run early:
    #: ``"budget"`` or ``"cancelled"`` (``None``: it finished).
    stopped: Optional[StopReason] = None


class _OutputSpec:
    """The synthetic output tool built from an output type (non-``str``)."""

    def __init__(self, output_type: type) -> None:
        self.adapter: TypeAdapter[Any] = TypeAdapter(output_type)
        schema = self.adapter.json_schema()
        self.wrapped = schema.get("type") != "object"
        if self.wrapped:
            # Hoist $defs to the wrapper root: pydantic emits "#/$defs/..."
            # refs relative to the schema root, so leaving them nested under
            # properties.value would dangle.
            defs = schema.pop("$defs", None)
            schema = {
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"],
            }
            if defs:
                schema["$defs"] = defs
        self.tool_def = ToolDef(
            name=OUTPUT_TOOL_NAME,
            description=(
                "Deliver the final result of this task. "
                "Call this exactly once, when the task is complete."
            ),
            input_schema=schema,
        )

    def validate(self, data: dict[str, Any]) -> Any:
        payload: Any = data.get("value") if self.wrapped else data
        return self.adapter.validate_python(payload)


class Agent:
    """A model + instructions + tool sources + an optional step-describer.

    Args:
        name: Identity, also the default tool name via :meth:`as_tool`.
        model: ``"provider:model_id"`` spec, or a bare model id when
            ``model_client`` is given.
        instructions: Static prose, or a (sync/async) function of the opaque
            per-run context for dynamic instructions. Either may be a list
            of parts, ordered stable to volatile: a provider that caches
            prompt prefixes caches each part (see ``SystemPrompt``).
        tools: Any mix of :class:`ToolSource` instances, ``@tool`` functions,
            and plain callables. Sub-agents join via ``sub.as_tool()``.
        output_type: Optional *default* output schema. Its presence lets the
            agent be run structured; it is not a mode flag.
        describe_step: Step-describer for user-facing narration; return
            ``None`` from it to hide a call. Defaults to a generic describer.
        model_client: Escape hatch — bring your own :class:`ModelClient`.
        max_tool_result_chars: Loop safety bound, like ``max_iterations``:
            cap any single tool result's ``content`` at this many characters
            before it enters the conversation. Over-cap content is cut, a
            truncation marker is appended (so the returned content slightly
            exceeds the cap), and the result is marked ``truncated`` —
            feeding the run's sticky ``truncated`` flag. ``None`` (default)
            disables the cap. Without one, a single oversized result — a
            big file over MCP, a bulky vendor payload, a verbose sub-agent —
            can exceed the model's context window and kill the run with a
            non-retryable provider error.
        workspace: A :class:`~shankit.Workspace`, or a (sync/async) function
            of the per-run context returning one (``None`` for none). The
            agent's own tools reach it through a
            :class:`~shankit.WorkspaceTools` given the same spec; the loop
            uses it to spill oversized tool results to files and to keep
            cleared context recoverable. Resolved once per run.
        spill_threshold_chars: With a workspace, a tool result longer than
            this is written in full to ``<workspace.tmp>/tool-output/`` and
            the model gets its head and tail plus the file path instead —
            lossless where ``max_tool_result_chars`` is lossy. ``None``
            disables spilling. (Coding agents use 30-50K; the default is
            25K characters.)
        reasoning: Provider-neutral reasoning ("thinking"): ``True`` on at
            the provider's default effort, an effort string (``"low"`` …
            ``"max"``), ``False`` explicitly off, a :class:`Reasoning` for
            full control, or ``None`` (default) to send nothing. Reasoning
            blocks round-trip within the run automatically.
        context_clear_threshold_tokens: Loop-level context clearing for
            long runs. When a model pass's prompt exceeds this many tokens,
            older tool results are replaced by a short stub (the full text
            goes to a workspace file first, when there is one) and reasoning
            blocks are dropped, keeping the newest
            ``context_keep_recent_results`` results intact. The pass after a
            clear runs with reasoning off (a tool continuation can't carry
            reasoning whose history was edited). ``None`` (default)
            disables it; a provider's own context editing, when available,
            is the better tool.
        timeout_s: Wall-clock bound on the run, checked before each model
            pass; exceeding it raises :class:`RunTimeoutError` (``timeout``
            on the event stream). Complements ``max_iterations``: long runs
            need a time bound, not just a step bound.
        keep_recent_images: Image-aware context clearing for tools that
            return images (screenshots). Once tool results hold twice this
            many images, all but the newest ``keep_recent_images`` are
            replaced by a short text stub - in batches, so the prompt cache
            survives the passes in between. Like ``context_clear_*``, a
            clear drops reasoning blocks and runs the next pass with
            reasoning off. ``None`` (default) keeps every image.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str,
        instructions: Instructions = "",
        description: Optional[str] = None,
        tools: Sequence[Any] = (),
        output_type: Optional[type] = None,
        describe_step: Optional[StepDescriber] = default_step_describer,
        model_client: Optional[ModelClient] = None,
        max_iterations: int = 20,
        output_retries: int = 2,
        max_tokens: int = 4096,
        max_tool_result_chars: Optional[int] = None,
        temperature: Optional[float] = None,
        workspace: Optional[WorkspaceSpec] = None,
        spill_threshold_chars: Optional[int] = 25_000,
        reasoning: ReasoningSpec = None,
        context_clear_threshold_tokens: Optional[int] = None,
        context_keep_recent_results: int = 3,
        timeout_s: Optional[float] = None,
        keep_recent_images: Optional[int] = None,
    ) -> None:
        # What copy() rebuilds from (so a copy is validated and normalized
        # exactly as the original was).
        self._init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        if keep_recent_images is not None and keep_recent_images < 1:
            raise ValueError("keep_recent_images must be at least 1 (or None to disable)")
        if max_tool_result_chars is not None and max_tool_result_chars <= 0:
            raise ValueError("max_tool_result_chars must be positive (or None to disable)")
        if spill_threshold_chars is not None and spill_threshold_chars <= 0:
            raise ValueError("spill_threshold_chars must be positive (or None to disable)")
        self.name = name
        self.model = model
        self.instructions = instructions
        self.description = description
        self.output_type = output_type
        self.describe_step = describe_step
        self.model_client = model_client
        self.max_iterations = max_iterations
        self.output_retries = output_retries
        self.max_tokens = max_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.temperature = temperature
        self.workspace = workspace
        self.spill_threshold_chars = spill_threshold_chars
        self.reasoning = _normalize_reasoning(reasoning)
        self.context_clear_threshold_tokens = context_clear_threshold_tokens
        self.context_keep_recent_results = max(0, context_keep_recent_results)
        self.timeout_s = timeout_s
        self.keep_recent_images = keep_recent_images
        self.tool_source: Optional[ToolSource] = _normalize_tools(tools)

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.model!r})"

    # ------------------------------------------------------------------ run

    @overload
    async def run(
        self,
        prompt: str,
        *,
        context: Any = None,
        output_type: type[OutputT],
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> RunResult[OutputT]: ...
    @overload
    async def run(
        self,
        prompt: str,
        *,
        context: Any = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> RunResult[Any]: ...

    async def run(
        self,
        prompt: str,
        *,
        context: Any = None,
        output_type: Optional[type] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> RunResult[Any]:
        """Run to a typed, validated deliverable.

        ``output_type`` overrides the agent's default schema for this call;
        ``output_type=str`` makes the final assistant text the deliverable
        (the full transcript stays on ``RunResult.text``).
        ``on_event`` optionally observes the event stream (steps, sources,
        usage) while the structured run progresses; it must be a plain
        sync callable, and an exception it raises aborts the run. ``history``
        is prior conversation turns (``Message`` objects or
        ``{"role", "content"}`` dicts) prepended before this call's
        ``prompt``. ``control`` steers the run from outside it (cancel,
        messages, a budget, call-time refusals, checkpoints; see
        :class:`~shankit.RunControl`).

        A failed model provider call raises :class:`shankit.ModelError`
        (check ``retryable`` for backoff) — never a provider SDK exception.
        """
        return await self._run_structured(
            prompt,
            context=context,
            output_type=output_type,
            on_event=on_event,
            history=history,
            control=control,
        )

    async def resume(
        self,
        transcript: Sequence[Any],
        *,
        context: Any = None,
        output_type: Optional[type] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        control: Optional[RunControl] = None,
        interrupted: str = (
            "Interrupted by a restart before this finished. Check whether it "
            "happened before doing it again."
        ),
    ) -> RunResult[Any]:
        """Continue a structured run from its saved transcript (what
        ``RunControl.checkpoint`` was given, or ``RunResult.messages``).

        A tool call the transcript shows in flight (an assistant pass whose
        results never landed) gets an error result saying ``interrupted``,
        so the model checks before redoing a side effect. A transcript that
        already ends in the model's final answer returns it without another
        model call. Step ids continue the original run's numbering; the
        returned ``usage`` covers only what the resume itself spent.
        """
        messages = _close_interrupted([coerce_message(m) for m in transcript], interrupted)
        if not messages:
            raise ShankitError("Agent.resume() needs a non-empty transcript.")
        effective = output_type or self.output_type
        last = messages[-1]
        ended = last.role == "assistant" and not any(
            isinstance(b, ToolUseBlock) for b in last.content
        )
        if effective is str and ended:
            # The final answer already landed.
            final = assistant_text(last)
            return RunResult(output=final, text=final, usage=Usage(), messages=messages)
        if effective not in (None, str):
            recorded = _recorded_output(messages, _OutputSpec(effective))
            if recorded is not None:
                return RunResult(output=recorded[0], text="", usage=Usage(), messages=messages)
            if ended:
                # Prose where the result was due: ask for it, as the loop would.
                messages.append(user_message(_OUTPUT_NUDGE))
        return await self._run_structured(
            None,
            context=context,
            output_type=output_type,
            on_event=on_event,
            history=messages,
            control=control,
        )

    def copy(self, **changes: Any) -> Agent:
        """A copy of this agent with some constructor arguments changed
        (``agent.copy(model="...", timeout_s=None)``). A copy that keeps the
        tools shares the original's tool source, and one that keeps the
        model, instructions, tools and reasoning sends the same prompt
        prefix."""
        unknown = set(changes) - set(self._init_kwargs)
        if unknown:
            raise TypeError(f"Agent.copy() got unknown arguments: {sorted(unknown)}")
        clone = Agent(**{**self._init_kwargs, **changes})
        if "tools" not in changes:
            clone.tool_source = self.tool_source
        return clone

    async def _run_structured(
        self,
        prompt: Optional[str],
        *,
        context: Any,
        output_type: Optional[type],
        on_event: Optional[Callable[[AgentEvent], None]],
        history: Optional[Sequence[Any]],
        control: Optional[RunControl],
    ) -> RunResult[Any]:
        effective = output_type or self.output_type
        if effective is None:
            raise ShankitError(
                f"Agent {self.name!r} has no output schema. Declare a default "
                "output_type on the agent, pass output_type= to run(), or use "
                "stream() for a free-form conversation."
            )
        trajectory: list[ToolCallRecord] = []
        sources: list[Source] = []
        artifacts: list[Artifact] = []
        with _tracing.span("shankit.agent.run", agent=self.name, model=self.model):
            async for event in self._loop(
                prompt,
                context=context,
                output_type=effective,
                streaming=False,
                trajectory=trajectory,
                sources=sources,
                artifacts=artifacts,
                history=history,
                control=control,
            ):
                if on_event is not None:
                    on_event(event)
                if isinstance(event, DoneEvent):
                    return RunResult(
                        output=event.output,
                        text=event.text,
                        usage=event.usage,
                        trajectory=trajectory,
                        sources=sources,
                        artifacts=artifacts,
                        truncated=event.truncated,
                        messages=event.messages,
                        stopped=event.stopped,
                    )
        raise ShankitError("Agent loop ended without a result.")  # pragma: no cover

    # --------------------------------------------------------------- stream

    async def stream(
        self,
        prompt: str,
        *,
        context: Any = None,
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run as a live conversation, yielding the typed event stream.

        The stream ends with a ``done`` event, or an ``error`` event if the
        run fails in an expected way (framework errors). Unexpected
        exceptions propagate after an ``error`` event is emitted.
        ``history`` is prior conversation turns (``Message`` objects or
        ``{"role", "content"}`` dicts) prepended before ``prompt``.
        ``control`` steers the run from outside it (see
        :class:`~shankit.RunControl`).
        """
        trajectory: list[ToolCallRecord] = []
        sources: list[Source] = []
        artifacts: list[Artifact] = []
        with _tracing.span("shankit.agent.stream", agent=self.name, model=self.model):
            try:
                async for event in self._loop(
                    prompt,
                    context=context,
                    output_type=None,
                    streaming=True,
                    trajectory=trajectory,
                    sources=sources,
                    artifacts=artifacts,
                    history=history,
                    control=control,
                ):
                    yield event
            except ShankitError as exc:
                yield error_event_for(exc)
            except Exception as exc:
                yield error_event_for(exc)
                raise

    # ----------------------------------------------------------- the loop

    async def _loop(
        self,
        prompt: Optional[str],
        *,
        context: Any,
        output_type: Optional[type],
        streaming: bool,
        trajectory: list[ToolCallRecord],
        sources: list[Source],
        artifacts: list[Artifact],
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> AsyncIterator[AgentEvent]:
        """The loop. ``prompt`` ``None`` continues ``history`` as the run's
        own transcript (:meth:`resume`) instead of starting a new turn."""
        client, model_id = resolve_model(self.model, self.model_client)
        started = time.monotonic()
        deadline = started + self.timeout_s if self.timeout_s is not None else None

        # Instructions, the tool catalog, and the workspace are independent;
        # resolving them concurrently matters when they hit the network
        # (dynamic instructions + a connector-backed tool source) on a cold
        # cache.
        async def list_tools() -> list[ToolDef]:
            if self.tool_source is None:
                return []
            return list(await self.tool_source.list_tools(context))

        system, tool_defs, workspace = await asyncio.gather(
            self._resolve_instructions(context),
            list_tools(),
            resolve_workspace(self.workspace, context),
        )
        known_tools = {t.name for t in tool_defs}
        toolsets = {t.name: t.toolset for t in tool_defs if t.toolset is not None}

        output_spec: Optional[_OutputSpec] = None
        if output_type is not None and output_type is not str:
            if OUTPUT_TOOL_NAME in known_tools:
                raise ShankitError(
                    f"Agent {self.name!r} has a tool named {OUTPUT_TOOL_NAME!r}, which "
                    "collides with the synthetic structured-output tool. Rename the "
                    "tool, or run this agent unstructured."
                )
            output_spec = _OutputSpec(output_type)
            tool_defs = [*tool_defs, output_spec.tool_def]

        messages: list[Message] = [coerce_message(m) for m in history or ()]
        if prompt is None:
            run_start = 0
        else:
            run_start = len(messages)
            messages.append(user_message(prompt))
        total_usage = Usage()
        texts: list[str] = []
        truncated = False
        output_attempts = 0
        force_output = False
        # A structured agent with NO real tools has nothing to explore — its
        # only job is the final_result call, so force it on every pass. Left
        # on "auto", small models often burn pass 1 answering in plain text;
        # the forced retry then re-derives the result with the answer already
        # spent as prose, which intermittently yields empty/degraded fields
        # (observed: an entity-linking extractor deterministically returning
        # [] for certain inputs). Forcing also saves that wasted first call.
        # Agents WITH tools keep "auto" — they must search before answering.
        always_force_output = output_spec is not None and not known_tools
        # A resumed run keeps numbering its passes and steps where the
        # original left off, so step ids stay unique across the two.
        passes = step_counter = 0
        for m in messages[run_start:]:
            if m.role == "assistant":
                passes += 1
                step_counter += sum(
                    1
                    for b in m.content
                    if isinstance(b, ToolUseBlock) and b.name != OUTPUT_TOOL_NAME
                )
        stopped: Optional[StopReason] = None
        winding_down = False
        # Shared by every tool call of this run (see ToolCallContext).
        run_state: dict[Any, Any] = {}
        # Context clearing bookkeeping: results already stubbed, where
        # spilled results live, and whether the next pass must run without
        # reasoning (its tool continuation's history was just edited).
        cleared: set[str] = set()
        spilled: dict[str, str] = {}
        reasoning_off_once = False
        last_prompt_tokens = 0

        def done_event(output: Any) -> DoneEvent:
            # text is the transcript (every pass); output is the answer.
            return DoneEvent(
                text="\n\n".join(texts),
                output=output,
                usage=total_usage,
                truncated=truncated,
                messages=list(messages[run_start:]),
                stopped=stopped,
            )

        save_checkpoint = control.checkpoint if control is not None else None

        async def checkpoint() -> None:
            if save_checkpoint is not None:
                saved = save_checkpoint(messages[run_start:])
                if inspect.isawaitable(saved):
                    await saved

        for _ in range(self.max_iterations):
            if control is not None:
                if control.cancelled:
                    stopped = "cancelled"
                    yield done_event(None)
                    return
                notes = control.drain()
                if notes:
                    _append_user_text(messages, "\n\n".join(notes))
                budget = control.budget
                limit = (
                    budget.reached(
                        passes=passes, elapsed_s=time.monotonic() - started, usage=total_usage
                    )
                    if budget is not None
                    else None
                )
                if budget is not None and limit:
                    # One last pass, tools off, to report what it has.
                    logger.info("Agent %r: %s budget reached; winding down.", self.name, limit)
                    winding_down = True
                    _append_user_text(messages, budget.notice)
            if deadline is not None and time.monotonic() > deadline:
                raise RunTimeoutError(
                    f"Agent {self.name!r} ran past its {self.timeout_s:g}s time limit."
                )
            if (
                self.context_clear_threshold_tokens is not None
                and last_prompt_tokens > self.context_clear_threshold_tokens
            ):
                if await self._clear_context(messages, cleared, spilled, workspace):
                    reasoning_off_once = True
                last_prompt_tokens = 0
            if self.keep_recent_images is not None and _clear_images(
                messages, self.keep_recent_images
            ):
                reasoning_off_once = True

            reasoning = self.reasoning
            if reasoning_off_once and reasoning is not None:
                reasoning = Reasoning(enabled=False)
            reasoning_off_once = False
            if winding_down:
                tool_choice: Any = (
                    ForcedTool(name=OUTPUT_TOOL_NAME) if output_spec is not None else "none"
                )
            elif force_output or always_force_output:
                tool_choice = ForcedTool(name=OUTPUT_TOOL_NAME)
            else:
                tool_choice = "auto"
            request = ModelRequest(
                model=model_id,
                system=system or None,
                messages=messages,
                tools=tool_defs,
                tool_choice=tool_choice,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning=reasoning,
            )
            force_output = False

            # Fallback half of the model error contract: the shipped clients
            # map their SDK's exceptions to ModelError themselves (they know
            # which failures are transient); anything a custom client leaks
            # is wrapped here so callers never see provider exception types.
            try:
                if streaming:
                    response = None
                    async for model_event in client.stream(request):
                        if isinstance(model_event, ModelTextDelta):
                            yield TextDeltaEvent(text=model_event.text)
                        elif isinstance(model_event, ModelResponseComplete):
                            response = model_event.response
                    if response is None:
                        raise ShankitError("Model stream ended without a complete response.")
                else:
                    response = await client.complete(request)
            except ShankitError:
                raise
            except Exception as exc:
                # Log before wrapping: stream() turns ModelError into a calm
                # terminal event, so without this the real traceback (which
                # may be a client bug, not a provider failure) is never seen.
                logger.exception(
                    "Model client %r raised a non-ModelError exception", type(client).__name__
                )
                raise ModelError("The model call failed.") from exc

            passes += 1
            total_usage.add(response.usage)
            yield UsageEvent(usage=response.usage)
            last_prompt_tokens = (
                response.usage.input_tokens
                + response.usage.cache_read_tokens
                + response.usage.cache_write_tokens
            )

            if response.stop_reason == "max_tokens":
                truncated = True
                logger.warning(
                    "Agent %r: model response truncated at max_tokens=%d; "
                    "the answer (or a tool call's input) may be incomplete.",
                    self.name,
                    self.max_tokens,
                )

            assistant = Message(role="assistant", content=response.content)
            messages.append(assistant)
            turn_text = assistant_text(assistant)
            if turn_text:
                texts.append(turn_text)

            tool_uses = [b for b in assistant.content if isinstance(b, ToolUseBlock)]

            if winding_down:
                # The budget's last pass is the report, whatever it holds.
                stopped = "budget"
                output = None
                for tu in tool_uses:
                    if output_spec is not None and tu.name == OUTPUT_TOOL_NAME:
                        with contextlib.suppress(ValidationError):
                            output = output_spec.validate(tu.input)
                if output_type is str:
                    output = turn_text
                if tool_uses:
                    # Keep the transcript valid for a later continuation.
                    messages.append(_unrun_results(tool_uses, "Not run: out of budget."))
                yield done_event(output)
                return

            if not tool_uses and control is not None and control.has_messages:
                # A message arrived while the model wrote what would have
                # been its answer: it gets one more pass to act on it.
                continue

            if not tool_uses:
                if output_spec is not None:
                    if output_attempts >= self.output_retries:
                        raise OutputValidationError(
                            f"Agent {self.name!r} did not produce a structured result "
                            f"after {output_attempts} retries."
                        )
                    output_attempts += 1
                    messages.append(user_message(_OUTPUT_NUDGE))
                    force_output = True
                    continue
                # For output_type=str the deliverable is the final pass —
                # the answer — not the transcript with its interim passes.
                # This pass's text specifically: an empty final pass must not
                # promote an earlier pass's narration to "the answer".
                output = turn_text if output_type is str else None
                yield done_event(output)
                return

            if control is not None and control.cancelled:
                # Stopped between the model's answer and its tools: none of
                # them run, and the transcript stays valid to continue.
                messages.append(_unrun_results(tool_uses, "Not run: stopped."))
                stopped = "cancelled"
                yield done_event(None)
                return

            await checkpoint()

            blocks_by_id: dict[str, ToolResultBlock] = {}
            finished: Optional[tuple[Any]] = None  # 1-tuple so None output is representable
            pending: list[tuple[ToolUseBlock, str, Optional[StepInfo]]] = []

            for tool_use in tool_uses:
                if output_spec is not None and tool_use.name == OUTPUT_TOOL_NAME:
                    try:
                        finished = (output_spec.validate(tool_use.input),)
                        blocks_by_id[tool_use.id] = ToolResultBlock(
                            tool_use_id=tool_use.id, content="Final result recorded."
                        )
                    except ValidationError as exc:
                        if output_attempts >= self.output_retries:
                            raise OutputValidationError(
                                f"Agent {self.name!r} produced output that failed "
                                f"validation after {output_attempts} retries: {exc}"
                            ) from exc
                        output_attempts += 1
                        blocks_by_id[tool_use.id] = ToolResultBlock(
                            tool_use_id=tool_use.id,
                            content=(
                                f"The result did not match the schema:\n{exc}\n"
                                f"Fix the issues and call `{OUTPUT_TOOL_NAME}` again."
                            ),
                            is_error=True,
                        )
                    continue

                step_counter += 1
                step_id = f"s{step_counter}"
                info = self._describe(tool_use, context)
                if info is not None:
                    yield _step_event(step_id, info, "running")
                pending.append((tool_use, step_id, info))

            # The model emits multiple tool calls in one turn knowing they are
            # independent, so execute them concurrently. Running steps were
            # already emitted in emission order above; results are processed
            # in that same order so the event stream stays deterministic.
            # While they run, events the tools push through their
            # ToolCallContext (a sub-agent's steps, a script's progress) are
            # yielded as they arrive.
            # If the consumer closes the stream (client disconnect) while we
            # are suspended here, cancel the in-flight tool tasks instead of
            # orphaning them to run (and side-effect) in the background.
            live: asyncio.Queue[AgentEvent] = asyncio.Queue()
            slots, tasks = self._start_batch(
                pending,
                toolsets,
                context,
                known_tools,
                live,
                run_state,
                workspace,
                spilled,
                pass_index=passes,
                control=control,
            )
            gathered = asyncio.gather(*tasks)
            try:
                while not gathered.done():
                    getter = asyncio.ensure_future(live.get())
                    try:
                        await asyncio.wait({gathered, getter}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        if not getter.done():
                            getter.cancel()
                            # Await the cancellation rather than dropping
                            # the task reference: otherwise it may still be
                            # pending when garbage collected, which logs a
                            # "Task was destroyed but it is pending!" warning.
                            with contextlib.suppress(asyncio.CancelledError):
                                await getter
                    if getter.done() and not getter.cancelled():
                        yield self._record_live(getter.result(), sources, artifacts)
                while not live.empty():
                    yield self._record_live(live.get_nowait(), sources, artifacts)
                gathered.result()
                results = [r for r in slots if r is not None]
            except BaseException:
                for task in tasks:
                    task.cancel()
                with contextlib.suppress(BaseException):
                    await gathered
                raise

            for (tool_use, step_id, info), result in zip(pending, results, strict=True):
                trajectory.append(
                    ToolCallRecord(
                        tool=tool_use.name,
                        arguments=tool_use.input,
                        content=content_text(result.content),
                        is_error=result.is_error,
                    )
                )
                for source in result.sources:
                    sources.append(source)
                    yield SourceEvent(source=source)
                for artifact in result.artifacts:
                    artifacts.append(artifact)
                    yield ArtifactEvent(artifact=artifact)
                if result.usage is not None:
                    # Usage a tool reports (e.g. a sub-agent's spend) is part
                    # of the run total, so it must also be part of the stream:
                    # UsageEvents always sum to DoneEvent.usage.
                    total_usage.add(result.usage)
                    yield UsageEvent(usage=result.usage)
                if result.truncated:
                    # A truncated sub-agent answer corrupts this run's
                    # deliverable just like a truncated own pass would.
                    truncated = True
                if info is not None:
                    yield _step_event(step_id, info, "error" if result.is_error else "done")
                blocks_by_id[tool_use.id] = ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                    toolset=tool_use.toolset,
                )

            messages.append(Message(role="user", content=[blocks_by_id[tu.id] for tu in tool_uses]))
            await checkpoint()

            if finished is not None:
                yield done_event(finished[0])
                return
            if any(result.end_run for result in results):
                # A tool ended the run (ToolResult.end_run): every result in
                # the batch has landed, and the model gets no further pass.
                # A structured run ends without a validated output.
                yield done_event(turn_text if output_type is str else None)
                return

        raise MaxIterationsError(
            f"Agent {self.name!r} hit max_iterations={self.max_iterations} without finishing."
        )

    # ------------------------------------------------------------- helpers

    def _start_batch(
        self,
        pending: list[tuple[ToolUseBlock, str, Optional[StepInfo]]],
        toolsets: dict[str, Toolset],
        context: Any,
        known_tools: set[str],
        live: asyncio.Queue[AgentEvent],
        run_state: dict[Any, Any],
        workspace: Optional[Workspace],
        spilled: dict[str, str],
        *,
        pass_index: int = 0,
        control: Optional[RunControl] = None,
    ) -> tuple[list[Optional[ToolResult]], list[asyncio.Future[None]]]:
        """Start one model turn's tool calls. Calls run concurrently, except
        an ordered toolset's (see :class:`Toolset`), which run one at a time
        in emission order and stop at the first failure. Results land in the
        returned slots, in ``pending`` order, as the tasks finish."""
        slots: list[Optional[ToolResult]] = [None] * len(pending)

        async def run_call(index: int) -> None:
            tu, sid, _ = pending[index]
            call = ToolCallContext(
                tool_use_id=tu.id,
                tool_name=tu.name,
                step_id=sid,
                agent_name=self.name,
                emit=live.put_nowait,
                run_state=run_state,
                pass_index=pass_index,
                agent=self,
                control=control,
            )
            slots[index] = await self._execute_tool(
                tu, context, known_tools, call, workspace, spilled
            )

        async def run_ordered(indices: list[int], toolset: Toolset) -> None:
            for position, index in enumerate(indices):
                await run_call(index)
                result = slots[index]
                if result is not None and result.is_error:
                    for rest in indices[position + 1 :]:
                        slots[rest] = ToolResult(content=toolset.not_executed, is_error=True)
                    return

        groups: dict[str, tuple[Toolset, list[int]]] = {}
        tasks: list[asyncio.Future[None]] = []
        for index, (tu, _, _) in enumerate(pending):
            toolset = toolsets.get(tu.name)
            if toolset is not None and toolset.ordered:
                groups.setdefault(toolset.name, (toolset, []))[1].append(index)
            else:
                tasks.append(asyncio.ensure_future(run_call(index)))
        for toolset, indices in groups.values():
            tasks.append(asyncio.ensure_future(run_ordered(indices, toolset)))
        return slots, tasks

    @staticmethod
    def _record_live(
        event: AgentEvent, sources: list[Source], artifacts: list[Artifact]
    ) -> AgentEvent:
        """A source or artifact a running tool emitted live (a card that must
        show before a long tool returns) belongs to the run's results too,
        like one returned on its ``ToolResult``."""
        if isinstance(event, ArtifactEvent):
            artifacts.append(event.artifact)
        elif isinstance(event, SourceEvent):
            sources.append(event.source)
        return event

    async def _resolve_instructions(self, context: Any) -> SystemPrompt:
        instructions = self.instructions
        if callable(instructions):
            resolved = instructions(context)
            if hasattr(resolved, "__await__"):
                resolved = await resolved  # type: ignore[misc]
            return resolved if isinstance(resolved, list) else str(resolved)
        return instructions

    def _describe(self, tool_use: ToolUseBlock, context: Any) -> Optional[StepInfo]:
        if self.describe_step is None:
            return None
        try:
            return self.describe_step(tool_use.name, tool_use.input, context)
        except Exception:
            logger.exception("Step describer failed for tool %r", tool_use.name)
            return default_step_describer(tool_use.name, tool_use.input, context)

    async def _execute_tool(
        self,
        tool_use: ToolUseBlock,
        context: Any,
        known_tools: set[str],
        call: Optional[ToolCallContext] = None,
        workspace: Optional[Workspace] = None,
        spilled: Optional[dict[str, str]] = None,
    ) -> ToolResult:
        """The uniform error contract (design §4): every failure comes back as
        a consistent ``is_error`` result the loop and the model can handle."""
        if self.tool_source is None or tool_use.name not in known_tools:
            return ToolResult(
                content=f"No tool named {tool_use.name!r} is available.", is_error=True
            )
        control = call.control if call is not None else None
        if control is not None and control.refuse is not None:
            try:
                refusal = control.refuse(tool_use.name, tool_use.input)
            except Exception:
                # A veto that fails closed: the call doesn't run.
                logger.exception("RunControl.refuse failed for tool %r", tool_use.name)
                refusal = f"The tool {tool_use.name!r} isn't available right now."
            if refusal:
                return ToolResult(content=refusal, is_error=True)
        # Runs inside this call's own task, so the context var is scoped to
        # it — concurrent calls each see their own ToolCallContext.
        _set_current(call)
        try:
            with _tracing.span("shankit.tool.execute", tool=tool_use.name, agent=self.name):
                raw = await self.tool_source.execute(tool_use.name, tool_use.input, context)
        except ToolError as exc:
            return self._cap_result(tool_use.name, ToolResult(content=str(exc), is_error=True))
        except Exception:
            logger.exception("Tool %r failed", tool_use.name)
            return ToolResult(
                content=f"The tool {tool_use.name!r} failed unexpectedly.", is_error=True
            )
        if isinstance(raw, str):  # leniency for duck-typed sources
            raw = ToolResult(content=raw)
        raw = await self._spill(tool_use, raw, workspace, spilled)
        return self._cap_result(tool_use.name, raw)

    async def _spill(
        self,
        tool_use: ToolUseBlock,
        result: ToolResult,
        workspace: Optional[Workspace],
        spilled: Optional[dict[str, str]],
    ) -> ToolResult:
        """Write an oversized result to a workspace file and hand the model
        its head, tail, and path instead (``spill_threshold_chars``). Falls
        through unchanged — to the lossy ``max_tool_result_chars`` cap —
        when there's no workspace or the write fails."""
        threshold = self.spill_threshold_chars
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
            self.name,
            tool_use.name,
            len(content),
            path,
        )
        return result.model_copy(update={"content": preview})

    async def _clear_context(
        self,
        messages: list[Message],
        cleared: set[str],
        spilled: dict[str, str],
        workspace: Optional[Workspace],
    ) -> bool:
        """Stub out all but the newest tool results (``context_clear_*``).

        Rewrites ``messages`` in place with new Message objects (never
        mutates the originals, which may be the caller's history). Returns
        whether anything was cleared."""
        positions = [
            (mi, bi)
            for mi, message in enumerate(messages)
            for bi, block in enumerate(message.content)
            if isinstance(block, ToolResultBlock)
        ]
        keep = self.context_keep_recent_results
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
            stub = (
                text[:_CLEAR_KEEP_CHARS] + f"… [older tool output cleared to save context.{where}]"
            )
            blocks = changed.setdefault(mi, list(messages[mi].content))
            blocks[bi] = block.model_copy(update={"content": stub})
            cleared.add(block.tool_use_id)
            count += 1
        if not count:
            return False
        for mi, blocks in changed.items():
            messages[mi] = messages[mi].model_copy(update={"content": blocks})
        _drop_reasoning(messages)
        logger.info("Agent %r: cleared %d older tool results from context.", self.name, count)
        return True

    def _cap_result(self, tool_name: str, result: ToolResult) -> ToolResult:
        """Apply ``max_tool_result_chars`` (loop safety bound, like
        ``max_iterations``): one oversized result must not be able to exceed
        the model's context window and kill the run. Applied at this choke
        point so every source — local functions, MCP, connectors, sub-agents,
        and ``ToolError`` text — is covered uniformly. The capped content is
        what the model, the trajectory, and evals all see."""
        cap = self.max_tool_result_chars
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
                            update={
                                "text": b.text[:remaining] + f"… [truncated at {cap} characters]"
                            }
                        )
                    )
                    remaining = 0
                else:
                    capped.append(b)
                    remaining -= len(b.text)
            return result.model_copy(update={"content": capped, "truncated": True})
        logger.warning(
            "Agent %r: tool %r returned %d chars; capped at max_tool_result_chars=%d.",
            self.name,
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

    # ------------------------------------------------------------- as_tool

    def as_tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output: str = "text",
    ) -> ToolSource:
        """Adapt this agent into a tool source, so handing sub-agents to a
        parent agent *is* orchestration (design §3.4).

        The sub-agent's failures are sanitized, its sources and artifacts
        propagate, and its token usage is folded into the parent run's
        accounting.

        Args:
            output: ``"text"`` (default) returns the sub-agent's final
                answer text (interim passes stay on its transcript);
                ``"structured"`` runs against the agent's default output
                schema and returns it as JSON.
        """
        if output not in ("text", "structured"):
            raise ValueError("output must be 'text' or 'structured'")
        if output == "structured" and self.output_type in (None, str):
            raise ShankitError(
                f"Agent {self.name!r} has no structured output schema; "
                "declare output_type on the agent to expose it structured."
            )
        return _AgentToolSource(self, name=name, description=description, output=output)


class _AgentToolSource(ToolSource):
    def __init__(
        self, agent: Agent, *, name: Optional[str], description: Optional[str], output: str
    ) -> None:
        self.agent = agent
        self.tool_name = _sanitize_tool_name(name or agent.name)
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
        call = current_tool_call()

        def forward(event: AgentEvent) -> None:
            # The sub-agent's steps surface live in the parent's stream.
            if call is not None and isinstance(event, StepEvent):
                call.emit(
                    event.model_copy(
                        update={
                            "id": f"{call.step_id}.{event.id}",
                            "agent": event.agent or self.agent.name,
                        }
                    )
                )

        try:
            if self.output == "structured":
                result = await self.agent.run(task, context=context, on_event=forward)
                content = dump_str(result.output)
            else:
                result = await self.agent.run(
                    task, context=context, output_type=str, on_event=forward
                )
                content = result.output
        except Exception:
            # Sanitized sub-agent failure: the parent model gets a calm,
            # uniform message; the real traceback goes to the logs.
            logger.exception("Sub-agent %r failed", self.agent.name)
            return ToolResult(
                content=f"The {self.agent.name} agent could not complete the task.",
                is_error=True,
            )
        return ToolResult(
            content=content,
            sources=result.sources,
            artifacts=result.artifacts,
            usage=result.usage,
            truncated=result.truncated,
        )


def _drop_reasoning(messages: list[Message]) -> None:
    """Edited history invalidates the reasoning after the edit, and a
    reasoning block is only valid in the exact conversation that produced
    it - drop them all rather than send stale ones."""
    for mi, message in enumerate(messages):
        if any(isinstance(b, ReasoningBlock) for b in message.content):
            blocks = [b for b in message.content if not isinstance(b, ReasoningBlock)]
            messages[mi] = message.model_copy(update={"content": blocks})


def _clear_images(messages: list[Message], keep: int) -> bool:
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


def _step_event(step_id: str, info: StepInfo, status: str) -> StepEvent:
    return StepEvent(
        id=step_id,
        title=info.title,
        detail=info.detail,
        phase=info.phase,
        status=status,  # type: ignore[arg-type]
        agent=info.agent,
    )


def _append_user_text(messages: list[Message], text: str) -> None:
    """Add ``text`` for the model after the last message: into it when it
    is already the user's (a pass's tool results, or the prompt), else as a
    new user turn. Never mutates the message objects themselves."""
    if messages and messages[-1].role == "user":
        last = messages[-1]
        messages[-1] = last.model_copy(update={"content": [*last.content, TextBlock(text=text)]})
    else:
        messages.append(user_message(text))


def _recorded_output(messages: list[Message], spec: _OutputSpec) -> Optional[tuple[Any]]:
    """The validated result of a transcript that ends with the structured
    result recorded (its call answered "Final result recorded."), as a
    1-tuple, else ``None``."""
    if len(messages) < 2 or messages[-1].role != "user" or messages[-2].role != "assistant":
        return None
    for use in messages[-2].content:
        if not isinstance(use, ToolUseBlock) or use.name != OUTPUT_TOOL_NAME:
            continue
        answered = any(
            isinstance(b, ToolResultBlock) and b.tool_use_id == use.id and not b.is_error
            for b in messages[-1].content
        )
        if answered:
            with contextlib.suppress(ValidationError):
                return (spec.validate(use.input),)
    return None


def _close_interrupted(messages: list[Message], text: str) -> list[Message]:
    """A transcript whose last pass called tools that never returned gets
    an error result for each, so the conversation is valid to continue."""
    if not messages or messages[-1].role != "assistant":
        return messages
    in_flight = [b for b in messages[-1].content if isinstance(b, ToolUseBlock)]
    return [*messages, _unrun_results(in_flight, text)] if in_flight else messages


def _unrun_results(tool_uses: Sequence[ToolUseBlock], text: str) -> Message:
    """An error result saying ``text`` for each of ``tool_uses`` that never
    ran: every tool call must be answered for the transcript to be valid."""
    return Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id=tu.id, content=text, is_error=True) for tu in tool_uses
        ],
    )


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


def _sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip()) or "agent"


def _normalize_tools(items: Sequence[Any]) -> Optional[ToolSource]:
    """Fold the ``tools=`` argument into a single seam implementation."""
    sources: list[ToolSource] = []
    functions: list[FunctionTool] = []
    for item in items:
        if isinstance(item, Agent):
            raise TypeError(
                f"Pass sub-agents as tools explicitly: {item.name}.as_tool() — "
                "not the Agent object itself."
            )
        if is_tool_source(item):
            sources.append(item)
        elif isinstance(item, FunctionTool):
            functions.append(item)
        elif callable(item):
            functions.append(FunctionTool(item))
        else:
            raise TypeError(
                f"Unsupported tool entry {item!r}: expected a ToolSource, a @tool "
                "function, a plain callable, or agent.as_tool()."
            )
    if functions:
        sources.insert(0, FunctionToolSource(functions))
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0]
    return CompositeToolSource(sources)
