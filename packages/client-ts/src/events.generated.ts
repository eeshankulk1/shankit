// GENERATED FILE — DO NOT EDIT.
// Source of truth: packages/shankit/src/shankit/events.py
// Regenerate with: python scripts/generate_ts_events.py

/**
 * A citation/source surfaced by a tool result.
 *
 * Extra fields are allowed so tools can attach vendor-specific metadata
 * without the framework needing to know about it.
 */
export interface Source {
  title: string;
  url: string | null;
  snippet: string | null;
  [key: string]: unknown;
}

/**
 * Token/request accounting for a model call or a whole run.
 *
 * Usage is additive: sub-agent runs surface their usage through the tool
 * seam (``ToolResult.usage``) and the parent loop folds it into the run
 * total, so multi-agent token accounting works without any special casing.
 */
export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  requests: number;
}

/**
 * A chunk of assistant text as it is generated.
 */
export interface TextDeltaEvent {
  type: "text_delta";
  text: string;
}

/**
 * A user-facing narration step, produced by the step-describer.
 *
 * Each tool call yields a ``running`` event followed by a ``done`` or
 * ``error`` event with the same ``id``.
 */
export interface StepEvent {
  type: "step";
  id: string;
  title: string;
  detail: string | null;
  phase: string | null;
  status: "running" | "done" | "error";
}

/**
 * A source/citation surfaced by a tool during the run.
 */
export interface SourceEvent {
  type: "source";
  source: Source;
}

/**
 * One usage increment within the run: a model call, or usage a tool
 * reported (e.g. a sub-agent's spend surfacing through the tool seam).
 *
 * Invariant: the UsageEvents of a stream sum to ``DoneEvent.usage``, so a
 * consumer can meter cost live without waiting for the terminal event.
 */
export interface UsageEvent {
  type: "usage";
  usage: Usage;
}

/**
 * Terminal event: the run failed.
 */
export interface ErrorEvent {
  type: "error";
  message: string;
}

/**
 * Terminal event: the run finished.
 *
 * ``text`` is every assistant text pass of the run joined with blank
 * lines — the same transcript the ``text_delta`` events streamed — not
 * just the final pass. ``output`` is set only when the run produced a
 * structured deliverable; for a plain streamed conversation it is
 * ``None``. ``truncated`` is true if any model pass of the run stopped at
 * the token limit, meaning the answer (or a tool call's input) may be
 * incomplete.
 */
export interface DoneEvent {
  type: "done";
  text: string;
  output: unknown;
  usage: Usage;
  truncated: boolean;
}

/** Every event a shankit agent stream can emit. A stream ends with
 * exactly one terminal event: `done` or `error`. */
export type AgentEvent = TextDeltaEvent | StepEvent | SourceEvent | UsageEvent | ErrorEvent | DoneEvent;

export type AgentEventType = AgentEvent["type"];
