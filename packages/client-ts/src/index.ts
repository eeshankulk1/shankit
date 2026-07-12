/**
 * @shankit/client — typed consumption of shankit agent event streams.
 *
 * This SDK consumes agent runs over SSE/HTTP; it does not run agents
 * (design §8). Event types are generated from the Python source of truth.
 */

export * from "./events.generated.js";
export { parseSSEStream } from "./sse.js";

import type { AgentEvent, DoneEvent, ErrorEvent } from "./events.generated.js";
import { parseSSEStream } from "./sse.js";

export interface StreamOptions extends RequestInit {
  /** Bring your own fetch (for polyfills, auth wrappers, or tests). */
  fetch?: typeof fetch;
}

/**
 * Open an agent stream and yield typed events.
 *
 * ```ts
 * for await (const event of streamEvents("/agents/support/stream", {
 *   method: "POST",
 *   body: JSON.stringify({ prompt }),
 * })) {
 *   switch (event.type) {
 *     case "text_delta": render(event.text); break;
 *     case "step": showStep(event); break;
 *     case "done": finish(event); break;
 *     case "error": fail(event.message); break;
 *   }
 * }
 * ```
 */
export async function* streamEvents(
  input: string | URL | Request,
  options: StreamOptions = {},
): AsyncGenerator<AgentEvent, void, void> {
  const { fetch: fetchImpl, ...init } = options;
  const doFetch = fetchImpl ?? fetch;
  // Normalize through Headers: spreading a Headers instance (or the
  // [["k","v"]] array form) as an object silently drops every entry.
  const headers = new Headers(init.headers);
  if (!headers.has("accept")) headers.set("accept", "text/event-stream");
  const response = await doFetch(input, { ...init, headers });
  if (!response.ok) {
    throw new Error(`shankit: stream request failed with HTTP ${response.status}`);
  }
  if (response.body === null) {
    throw new Error("shankit: stream response has no body");
  }
  yield* parseSSEStream(response.body);
}

/** Narrowing helper: is this the successful terminal event? */
export function isDone(event: AgentEvent): event is DoneEvent {
  return event.type === "done";
}

/** Narrowing helper: is this the failure terminal event? */
export function isError(event: AgentEvent): event is ErrorEvent {
  return event.type === "error";
}

/**
 * Convenience for non-streaming consumers: drain the stream and return the
 * terminal event, throwing on an `error` event.
 */
export async function collectRun(
  events: AsyncIterable<AgentEvent>,
): Promise<DoneEvent> {
  for await (const event of events) {
    if (isDone(event)) return event;
    if (isError(event)) throw new Error(`shankit: run failed: ${event.message}`);
  }
  throw new Error("shankit: stream ended without a terminal event");
}
