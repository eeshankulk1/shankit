"""Internal: the state one run of the agent loop carries across its passes.

``Agent._loop`` owns the control flow; :class:`RunState` owns what it
counts and collects (the transcript, usage, texts, step numbering, the
context-clearing bookkeeping) and the events that fall out of recording a
pass or a tool result, so the totals and the stream can't drift apart.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from .events import AgentEvent, Artifact, ArtifactEvent, DoneEvent, Source, SourceEvent, UsageEvent
from .messages import Message, ToolUseBlock, coerce_message, user_message
from .tools.base import ToolResult
from .usage import Usage

if TYPE_CHECKING:
    from .agent import Prompt
    from .control import RunControl

__all__ = ["RunState", "StopReason"]

#: Why a :class:`~shankit.RunControl` ended a run early.
StopReason = Literal["budget", "cancelled"]


@dataclass
class RunState:
    """One run: its messages (history first, from ``run_start`` its own
    transcript) and what the loop tracks across passes."""

    messages: list[Message]
    run_start: int
    sources: list[Source]
    artifacts: list[Artifact]
    usage: Usage = field(default_factory=Usage)
    texts: list[str] = field(default_factory=list)
    truncated: bool = False
    stopped: Optional[StopReason] = None
    passes: int = 0
    step_counter: int = 0
    output_attempts: int = 0
    force_output: bool = False
    winding_down: bool = False
    # Shared by every tool call of this run (see ToolCallContext).
    tool_state: dict[Any, Any] = field(default_factory=dict)
    # Context clearing bookkeeping: results already stubbed, where
    # spilled results live, and whether the next pass must run without
    # reasoning (its tool continuation's history was just edited).
    cleared: set[str] = field(default_factory=set)
    spilled: dict[str, str] = field(default_factory=dict)
    reasoning_off_once: bool = False
    last_prompt_tokens: int = 0

    @classmethod
    def start(
        cls,
        prompt: Optional[Prompt],
        history: Optional[Sequence[Any]],
        *,
        sources: list[Source],
        artifacts: list[Artifact],
        output_tool: str,
    ) -> RunState:
        """``prompt`` ``None`` continues ``history`` as the run's own
        transcript (a resume) instead of starting a new turn."""
        messages = [coerce_message(m) for m in history or ()]
        if prompt is None:
            run_start = 0
        else:
            run_start = len(messages)
            messages.append(user_message(prompt))
        run = cls(messages=messages, run_start=run_start, sources=sources, artifacts=artifacts)
        # A resumed run keeps numbering its passes and steps where the
        # original left off, so step ids stay unique across the two.
        for m in run.transcript:
            if m.role == "assistant":
                run.passes += 1
                run.step_counter += sum(
                    1 for b in m.content if isinstance(b, ToolUseBlock) and b.name != output_tool
                )
        return run

    @property
    def transcript(self) -> list[Message]:
        """The run's own messages (after the history)."""
        return self.messages[self.run_start :]

    def next_step_id(self) -> str:
        self.step_counter += 1
        return f"s{self.step_counter}"

    def done(self, output: Any) -> DoneEvent:
        # text is the transcript (every pass); output is the answer.
        return DoneEvent(
            text="\n\n".join(self.texts),
            output=output,
            usage=self.usage,
            truncated=self.truncated,
            messages=self.transcript,
            stopped=self.stopped,
        )

    async def checkpoint(self, control: Optional[RunControl]) -> None:
        if control is not None and control.checkpoint is not None:
            saved = control.checkpoint(self.transcript)
            if inspect.isawaitable(saved):
                await saved

    def record_pass(self, usage: Usage) -> UsageEvent:
        """Count a model pass and add its usage to the run total."""
        self.passes += 1
        self.usage.add(usage)
        self.last_prompt_tokens = (
            usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        )
        return UsageEvent(usage=usage)

    def record_live(self, event: AgentEvent) -> AgentEvent:
        """A source or artifact a running tool emitted live (a card that must
        show before a long tool returns) belongs to the run's results too,
        like one returned on its ``ToolResult``."""
        if isinstance(event, ArtifactEvent):
            self.artifacts.append(event.artifact)
        elif isinstance(event, SourceEvent):
            self.sources.append(event.source)
        return event

    def record_result(self, result: ToolResult) -> list[AgentEvent]:
        """Fold a tool result's sources, artifacts, usage and truncation into
        the run; the events that report them."""
        events: list[AgentEvent] = []
        for source in result.sources:
            self.sources.append(source)
            events.append(SourceEvent(source=source))
        for artifact in result.artifacts:
            self.artifacts.append(artifact)
            events.append(ArtifactEvent(artifact=artifact))
        if result.usage is not None:
            # Usage a tool reports (e.g. a sub-agent's spend) is part of the
            # run total, so it must also be part of the stream: UsageEvents
            # always sum to DoneEvent.usage.
            self.usage.add(result.usage)
            events.append(UsageEvent(usage=result.usage))
        if result.truncated:
            # A truncated sub-agent answer corrupts this run's deliverable
            # just like a truncated own pass would.
            self.truncated = True
        return events
