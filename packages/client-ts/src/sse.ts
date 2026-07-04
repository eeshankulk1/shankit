/**
 * A minimal, dependency-free SSE parser for shankit event streams.
 *
 * The wire format is produced by `shankit.sse.format_sse` on the Python
 * side: one SSE message per event, JSON in the `data` field (which carries
 * the discriminating `type`, so the SSE `event` field is informational).
 */

import type { AgentEvent } from "./events.generated.js";

/** Parse a byte stream of SSE messages into typed agent events. */
export async function* parseSSEStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<AgentEvent, void, void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary: number;
      // SSE messages are separated by a blank line.
      while ((boundary = findMessageBoundary(buffer)) !== -1) {
        const rawMessage = buffer.slice(0, boundary);
        buffer = buffer.slice(skipBoundary(buffer, boundary));
        const event = parseMessage(rawMessage);
        if (event !== null) yield event;
      }
    }
    // Flush a final message without a trailing blank line, if any.
    const tail = parseMessage(buffer);
    if (tail !== null) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function findMessageBoundary(buffer: string): number {
  const lf = buffer.indexOf("\n\n");
  const crlf = buffer.indexOf("\r\n\r\n");
  if (lf === -1) return crlf;
  if (crlf === -1) return lf;
  return Math.min(lf, crlf);
}

function skipBoundary(buffer: string, boundary: number): number {
  return buffer.startsWith("\r\n\r\n", boundary) ? boundary + 4 : boundary + 2;
}

function parseMessage(raw: string): AgentEvent | null {
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    // `event:` lines are redundant (type travels in the JSON); comments
    // (lines starting with ":") and unknown fields are ignored per SSE.
  }
  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  try {
    return JSON.parse(payload) as AgentEvent;
  } catch {
    throw new Error(`shankit: received malformed event data: ${payload.slice(0, 200)}`);
  }
}
