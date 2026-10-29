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
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel, TypeAdapter, ValidationError

from . import _tracing
from ._calls import ToolCallContext, _set_current
from ._context import (
    append_user_text,
    cap_result,
    clear_images,
    clear_results,
    close_interrupted,
    spill,
    unrun_results,
)
from ._run_state import RunState, StopReason
from ._subagent import AgentToolSource
from .control import RunControl
from .events import (
    AgentEvent,
    Artifact,
    DoneEvent,
    Source,
    StepEvent,
    TextDeltaEvent,
    error_event_for,
)
from .exceptions import (
    MaxIterationsError,
    ModelError,
    OutputValidationError,
    RunTimeoutError,
    ShankitError,
    ToolError,
)
from .messages import (
    Message,
    PromptBlock,
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

#: What a run starts from: text, or text and image blocks in the order the
#: model should read them (a photo the user attached, then their question).
Prompt = Union[str, Sequence[PromptBlock]]


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
        prompt: Prompt,
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
        prompt: Prompt,
        *,
        context: Any = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> RunResult[Any]: ...

    async def run(
        self,
        prompt: Prompt,
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
        ``prompt``, which is text or a list of text and image blocks
        (:data:`Prompt`). ``control`` steers the run from outside it (cancel,
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
        messages = close_interrupted([coerce_message(m) for m in transcript], interrupted)
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
        prompt: Optional[Prompt],
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

    def stream(
        self,
        prompt: Prompt,
        *,
        context: Any = None,
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run as a live conversation, yielding the typed event stream.

        The stream ends with a ``done`` event, or an ``error`` event if the
        run fails in an expected way (framework errors). Unexpected
        exceptions propagate after an ``error`` event is emitted.
        ``history`` is prior conversation turns (``Message`` objects or
        ``{"role", "content"}`` dicts) prepended before ``prompt`` (text, or
        text and image blocks: :data:`Prompt`).
        ``control`` steers the run from outside it (see
        :class:`~shankit.RunControl`).

        Close a stream you stop reading early (``aclose()``, or
        ``contextlib.aclosing``): closing cancels the run's in-flight tools.
        """
        return _tracing.traced(
            "shankit.agent.stream",
            self._stream(prompt, context=context, history=history, control=control),
            agent=self.name,
            model=self.model,
        )

    async def _stream(
        self,
        prompt: Prompt,
        *,
        context: Any,
        history: Optional[Sequence[Any]],
        control: Optional[RunControl],
    ) -> AsyncGenerator[AgentEvent, None]:
        loop = self._loop(
            prompt,
            context=context,
            output_type=None,
            streaming=True,
            trajectory=[],
            sources=[],
            artifacts=[],
            history=history,
            control=control,
        )
        try:
            async for event in loop:
                yield event
        except ShankitError as exc:
            yield error_event_for(exc)
        except Exception as exc:
            yield error_event_for(exc)
            raise
        finally:
            await loop.aclose()

    # ----------------------------------------------------------- the loop

    async def _loop(
        self,
        prompt: Optional[Prompt],
        *,
        context: Any,
        output_type: Optional[type],
        streaming: bool,
        trajectory: list[ToolCallRecord],
        sources: list[Source],
        artifacts: list[Artifact],
        history: Optional[Sequence[Any]] = None,
        control: Optional[RunControl] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
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

        # A structured agent with NO real tools has nothing to explore — its
        # only job is the final_result call, so force it on every pass. Left
        # on "auto", small models often burn pass 1 answering in plain text;
        # the forced retry then re-derives the result with the answer already
        # spent as prose, which intermittently yields empty/degraded fields
        # (observed: an entity-linking extractor deterministically returning
        # [] for certain inputs). Forcing also saves that wasted first call.
        # Agents WITH tools keep "auto" — they must search before answering.
        always_force_output = output_spec is not None and not known_tools
        run = RunState.start(
            prompt, history, sources=sources, artifacts=artifacts, output_tool=OUTPUT_TOOL_NAME
        )

        for _ in range(self.max_iterations):
            if control is not None:
                if control.cancelled:
                    run.stopped = "cancelled"
                    yield run.done(None)
                    return
                self._steer(run, control, started)
            if deadline is not None and time.monotonic() > deadline:
                raise RunTimeoutError(
                    f"Agent {self.name!r} ran past its {self.timeout_s:g}s time limit."
                )
            await self._trim_context(run, workspace)

            reasoning = self.reasoning
            if run.reasoning_off_once and reasoning is not None:
                reasoning = Reasoning(enabled=False)
            run.reasoning_off_once = False
            if run.winding_down:
                tool_choice: Any = (
                    ForcedTool(name=OUTPUT_TOOL_NAME) if output_spec is not None else "none"
                )
            elif run.force_output or always_force_output:
                tool_choice = ForcedTool(name=OUTPUT_TOOL_NAME)
            else:
                tool_choice = "auto"
            request = ModelRequest(
                model=model_id,
                system=system or None,
                messages=run.messages,
                tools=tool_defs,
                tool_choice=tool_choice,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning=reasoning,
                agent=self.name,
            )
            run.force_output = False

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

            # Tagged with its model, so a run that mixes models prices right.
            yield run.record_pass(response.usage.for_model(model_id))
            if response.stop_reason == "max_tokens":
                run.truncated = True
                logger.warning(
                    "Agent %r: model response truncated at max_tokens=%d; "
                    "the answer (or a tool call's input) may be incomplete.",
                    self.name,
                    self.max_tokens,
                )

            assistant = Message(role="assistant", content=response.content)
            run.messages.append(assistant)
            turn_text = assistant_text(assistant)
            if turn_text:
                run.texts.append(turn_text)

            tool_uses = [b for b in assistant.content if isinstance(b, ToolUseBlock)]

            if run.winding_down:
                # The budget's last pass is the report, whatever it holds.
                run.stopped = "budget"
                output = None
                for tu in tool_uses:
                    if output_spec is not None and tu.name == OUTPUT_TOOL_NAME:
                        with contextlib.suppress(ValidationError):
                            output = output_spec.validate(tu.input)
                if output_type is str:
                    output = turn_text
                if tool_uses:
                    # Keep the transcript valid for a later continuation.
                    run.messages.append(unrun_results(tool_uses, "Not run: out of budget."))
                yield run.done(output)
                return

            if not tool_uses and control is not None and control.has_messages:
                # A message arrived while the model wrote what would have
                # been its answer: it gets one more pass to act on it.
                continue

            if not tool_uses:
                if output_spec is not None:
                    if run.output_attempts >= self.output_retries:
                        raise OutputValidationError(
                            f"Agent {self.name!r} did not produce a structured result "
                            f"after {run.output_attempts} retries."
                        )
                    run.output_attempts += 1
                    run.messages.append(user_message(_OUTPUT_NUDGE))
                    run.force_output = True
                    continue
                # For output_type=str the deliverable is the final pass —
                # the answer — not the transcript with its interim passes.
                # This pass's text specifically: an empty final pass must not
                # promote an earlier pass's narration to "the answer".
                yield run.done(turn_text if output_type is str else None)
                return

            if control is not None and control.cancelled:
                # Stopped between the model's answer and its tools: none of
                # them run, and the transcript stays valid to continue.
                run.messages.append(unrun_results(tool_uses, "Not run: stopped."))
                run.stopped = "cancelled"
                yield run.done(None)
                return

            await run.checkpoint(control)

            blocks_by_id: dict[str, ToolResultBlock] = {}
            finished: Optional[tuple[Any]] = None  # 1-tuple so None output is representable
            pending: list[tuple[ToolUseBlock, str, Optional[StepInfo]]] = []
            for tool_use in tool_uses:
                if output_spec is not None and tool_use.name == OUTPUT_TOOL_NAME:
                    blocks_by_id[tool_use.id], recorded = self._take_output(
                        tool_use, output_spec, run
                    )
                    if recorded is not None:
                        finished = recorded
                    continue
                step_id = run.next_step_id()
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
            live: asyncio.Queue[AgentEvent] = asyncio.Queue()
            slots, tasks = self._start_batch(
                pending, toolsets, context, known_tools, live, run, workspace, control=control
            )
            # aclosing: a consumer that closes this stream mid-batch closes
            # the drain too, which cancels the in-flight tools.
            async with contextlib.aclosing(_drain_while_running(tasks, live)) as drained:
                async for event in drained:
                    yield run.record_live(event)
            results = [r for r in slots if r is not None]

            for (tool_use, step_id, info), result in zip(pending, results, strict=True):
                trajectory.append(
                    ToolCallRecord(
                        tool=tool_use.name,
                        arguments=tool_use.input,
                        content=content_text(result.content),
                        is_error=result.is_error,
                    )
                )
                for event in run.record_result(result):
                    yield event
                if info is not None:
                    yield _step_event(step_id, info, "error" if result.is_error else "done")
                blocks_by_id[tool_use.id] = ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                    toolset=tool_use.toolset,
                )

            run.messages.append(
                Message(role="user", content=[blocks_by_id[tu.id] for tu in tool_uses])
            )
            await run.checkpoint(control)

            if finished is not None:
                yield run.done(finished[0])
                return
            if any(result.end_run for result in results):
                # A tool ended the run (ToolResult.end_run): every result in
                # the batch has landed, and the model gets no further pass.
                # A structured run ends without a validated output.
                yield run.done(turn_text if output_type is str else None)
                return

        raise MaxIterationsError(
            f"Agent {self.name!r} hit max_iterations={self.max_iterations} without finishing."
        )

    # ------------------------------------------------------------- helpers

    def _steer(self, run: RunState, control: RunControl, started: float) -> None:
        """Before a pass: hand the model the messages queued on ``control``,
        and start winding down once its budget is reached."""
        notes = control.drain()
        if notes:
            append_user_text(run.messages, "\n\n".join(notes))
        budget = control.budget
        limit = (
            budget.reached(passes=run.passes, elapsed_s=time.monotonic() - started, usage=run.usage)
            if budget is not None
            else None
        )
        if budget is not None and limit:
            # One last pass, tools off, to report what it has.
            logger.info("Agent %r: %s budget reached; winding down.", self.name, limit)
            run.winding_down = True
            append_user_text(run.messages, budget.notice)

    async def _trim_context(self, run: RunState, workspace: Optional[Workspace]) -> None:
        """Before a pass: context clearing (``context_clear_*``,
        ``keep_recent_images``). A clear runs the next pass without
        reasoning."""
        if (
            self.context_clear_threshold_tokens is not None
            and run.last_prompt_tokens > self.context_clear_threshold_tokens
        ):
            if await clear_results(
                run.messages,
                keep=self.context_keep_recent_results,
                cleared=run.cleared,
                spilled=run.spilled,
                workspace=workspace,
                agent_name=self.name,
            ):
                run.reasoning_off_once = True
            run.last_prompt_tokens = 0
        if self.keep_recent_images is not None and clear_images(
            run.messages, self.keep_recent_images
        ):
            run.reasoning_off_once = True

    def _take_output(
        self, tool_use: ToolUseBlock, spec: _OutputSpec, run: RunState
    ) -> tuple[ToolResultBlock, Optional[tuple[Any]]]:
        """Answer a ``final_result`` call: its validated result (a 1-tuple)
        when it matches the schema, else a retry request until
        ``output_retries`` runs out."""
        try:
            recorded = (spec.validate(tool_use.input),)
        except ValidationError as exc:
            if run.output_attempts >= self.output_retries:
                raise OutputValidationError(
                    f"Agent {self.name!r} produced output that failed "
                    f"validation after {run.output_attempts} retries: {exc}"
                ) from exc
            run.output_attempts += 1
            retry = ToolResultBlock(
                tool_use_id=tool_use.id,
                content=(
                    f"The result did not match the schema:\n{exc}\n"
                    f"Fix the issues and call `{OUTPUT_TOOL_NAME}` again."
                ),
                is_error=True,
            )
            return retry, None
        return ToolResultBlock(tool_use_id=tool_use.id, content="Final result recorded."), recorded

    def _start_batch(
        self,
        pending: list[tuple[ToolUseBlock, str, Optional[StepInfo]]],
        toolsets: dict[str, Toolset],
        context: Any,
        known_tools: set[str],
        live: asyncio.Queue[AgentEvent],
        run: RunState,
        workspace: Optional[Workspace],
        *,
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
                run_state=run.tool_state,
                pass_index=run.passes,
                agent=self,
                control=control,
            )
            slots[index] = await self._execute_tool(
                tu, context, known_tools, call, workspace, run.spilled
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
            raw = ToolResult(content=str(exc), is_error=True)
        except Exception:
            logger.exception("Tool %r failed", tool_use.name)
            return ToolResult(
                content=f"The tool {tool_use.name!r} failed unexpectedly.", is_error=True
            )
        else:
            if isinstance(raw, str):  # leniency for duck-typed sources
                raw = ToolResult(content=raw)
            raw = await spill(
                tool_use,
                raw,
                workspace=workspace,
                threshold=self.spill_threshold_chars,
                spilled=spilled,
                agent_name=self.name,
            )
        return cap_result(
            raw, cap=self.max_tool_result_chars, tool_name=tool_use.name, agent_name=self.name
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
        return AgentToolSource(self, name=name, description=description, output=output)


def _step_event(step_id: str, info: StepInfo, status: str) -> StepEvent:
    return StepEvent(
        id=step_id,
        title=info.title,
        detail=info.detail,
        phase=info.phase,
        status=status,  # type: ignore[arg-type]
        agent=info.agent,
    )


async def _drain_while_running(
    tasks: list[asyncio.Future[None]], live: asyncio.Queue[AgentEvent]
) -> AsyncGenerator[AgentEvent, None]:
    """Yield the events tools push onto ``live`` as they arrive, until every
    task is done, then raise the first task failure if there was one. If the
    consumer stops early (it closed the stream, or the run was cancelled),
    the tasks are cancelled instead of orphaned to run (and side-effect) in
    the background."""
    gathered = asyncio.gather(*tasks)
    try:
        while not gathered.done():
            getter = asyncio.ensure_future(live.get())
            try:
                await asyncio.wait({gathered, getter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not getter.done():
                    getter.cancel()
                    # Await the cancellation rather than dropping the task
                    # reference: otherwise it may still be pending when
                    # garbage collected, which logs a "Task was destroyed but
                    # it is pending!" warning.
                    with contextlib.suppress(asyncio.CancelledError):
                        await getter
            if getter.done() and not getter.cancelled():
                yield getter.result()
        while not live.empty():
            yield live.get_nowait()
        gathered.result()
    except BaseException:
        for task in tasks:
            task.cancel()
        with contextlib.suppress(BaseException):
            await gathered
        raise


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
