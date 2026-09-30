// GENERATED FILE — DO NOT EDIT.
// Source of truth: packages/shankit/src/shankit/events.py
// Regenerate with: python scripts/generate_ts_events.py

/**
 * A typed structured payload surfaced by a tool result.
 *
 * Where a :class:`Source` is semantically a citation, an artifact is
 * content: a piece of typed data the application wants to carry from a
 * tool execution out to the caller intact (e.g. an email card, a
 * calendar event). ``type`` names the application-defined kind;
 * ``data`` is opaque to the framework — it is never inspected, only
 * carried (same stance as the per-run context, design §3.3).
 */
export interface Artifact {
  type: string;
  data: Record<string, unknown>;
}

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
 * ``error`` event with the same ``id``. ``agent`` attributes the step to
 * a delegated sub-agent (its steps are forwarded into the parent run
 * with ids prefixed by the parent step's id) or whatever the
 * step-describer names; ``None`` means the running agent itself.
 * ``worker`` is separate from ``agent``: the id of the run that took the
 * step when several copies of one agent work at once (a clone, see
 * ``WorkerEvent``); ``None`` for the top-level run.
 */
export interface StepEvent {
  type: "step";
  id: string;
  title: string;
  detail: string | null;
  phase: string | null;
  status: "running" | "done" | "error";
  agent: string | null;
  worker: string | null;
}

/**
 * A source/citation surfaced by a tool during the run.
 */
export interface SourceEvent {
  type: "source";
  source: Source;
}

/**
 * A structured payload surfaced by a tool during the run.
 */
export interface ArtifactEvent {
  type: "artifact";
  artifact: Artifact;
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
 * A clone's lifecycle in the stream of the run that spawned it.
 *
 * ``id`` is the clone's worker id (its steps carry it as
 * ``StepEvent.worker``); ``batch`` groups the clones one model response
 * spawned. ``status`` is ``running`` when it starts, then one of
 * ``done``, ``partial`` (it reached its budget and reported what it had),
 * ``failed``, ``stopped`` (cancelled), or ``background`` (still running
 * when the spawning call returned; its host delivers the report later).
 */
export interface WorkerEvent {
  type: "worker";
  id: string;
  title: string;
  status: "running" | "done" | "partial" | "failed" | "stopped" | "background";
  batch: string | null;
}

/**
 * Terminal event: the run failed.
 *
 * ``message`` is human-safe and may be shown to end users. ``code`` says
 * *what kind* of failure without parsing the message; the codes emitted
 * today are ``model_error`` (the model provider call failed),
 * ``max_iterations``, ``output_validation``, ``timeout`` (the run passed
 * its ``timeout_s``), ``error`` (other framework
 * errors), and ``unexpected`` — the field stays an open string so new
 * codes are not a breaking change. ``retryable`` is true when retrying
 * the run shortly is reasonable (rate limits, provider overloads).
 */
export interface ErrorEvent {
  type: "error";
  message: string;
  code: string;
  retryable: boolean;
}

/**
 * Terminal event: the run finished.
 *
 * ``text`` is the transcript: every assistant text pass of the run — the
 * same content the ``text_delta`` events streamed — joined with blank
 * lines (the deltas themselves carry no separator between passes).
 * ``output`` is set only when the run produced a deliverable (for
 * ``output_type=str`` runs, the final pass's text); for a plain streamed
 * conversation it is ``None``. ``truncated`` is true if any model pass of
 * the run (or of a sub-agent run reporting through the tool seam) stopped
 * at the token limit, meaning the result may be incomplete. ``stopped``
 * says the run ended early by :class:`~shankit.RunControl`: ``"budget"``
 * (it reached its budget and wrote a last report) or ``"cancelled"``.
 */
export interface DoneEvent {
  type: "done";
  text: string;
  output: unknown;
  usage: Usage;
  truncated: boolean;
  stopped: "budget" | "cancelled" | null;
}

/** Every event a shankit agent stream can emit. A stream ends with
 * exactly one terminal event: `done` or `error`. */
export type AgentEvent = TextDeltaEvent | StepEvent | SourceEvent | ArtifactEvent | UsageEvent | WorkerEvent | ErrorEvent | DoneEvent;

export type AgentEventType = AgentEvent["type"];
