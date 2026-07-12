/**
 * A minimal, dependency-free SSE parser for shankit event streams.
 *
 * The wire format is produced by `shankit.sse.format_sse` on the Python
 * side: one SSE message per event, JSON in the `data` field (which carries
 * the discriminating `type`, so the SSE `event` field is informational).
 * Line endings may be LF, CRLF, or lone CR (all spec-legal), in any mix.
 *
 * One deliberate leniency: a final message not terminated by a blank line
 * is still dispatched at end-of-stream (the spec says to discard it). The
 * Python server always terminates messages, so this only matters for
 * truncated streams — and a truncated JSON payload fails parsing anyway.
 */

import type { AgentEvent } from "./events.generated.js";

/** Parse a byte stream of SSE messages into typed agent events. */
export async function* parseSSEStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<AgentEvent, void, void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];
  // True when a chunk ended exactly at a CR: if the next chunk starts with
  // the matching LF of a split CRLF, that LF is not a line of its own.
  let pendingLF = false;
  let finished = false;

  const nextLine = (): string | null => {
    if (pendingLF) {
      if (buffer === "") return null;
      if (buffer.startsWith("\n")) buffer = buffer.slice(1);
      pendingLF = false;
    }
    const lf = buffer.indexOf("\n");
    const cr = buffer.indexOf("\r");
    if (lf === -1 && cr === -1) return null;
    let end: number;
    let skip: number;
    if (cr !== -1 && (lf === -1 || cr < lf)) {
      end = cr;
      if (cr === buffer.length - 1) {
        // The CR is the last byte we have; a matching LF may follow in the
        // next chunk.
        skip = 1;
        pendingLF = true;
      } else {
        skip = buffer[cr + 1] === "\n" ? 2 : 1;
      }
    } else {
      end = lf;
      skip = 1;
    }
    const line = buffer.slice(0, end);
    buffer = buffer.slice(end + skip);
    return line;
  };

  const takeEvent = (): AgentEvent | null => {
    if (dataLines.length === 0) return null;
    const payload = dataLines.join("\n");
    dataLines = [];
    try {
      return JSON.parse(payload) as AgentEvent;
    } catch {
      throw new Error(`shankit: received malformed event data: ${payload.slice(0, 200)}`);
    }
  };

  const handleLine = (line: string): AgentEvent | null => {
    if (line === "") return takeEvent(); // blank line dispatches the message
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    } else if (line === "data") {
      dataLines.push(""); // bare field name: empty value, per spec
    }
    // `event:` lines are redundant (type travels in the JSON); comments
    // (lines starting with ":") and unknown fields are ignored per SSE.
    return null;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let line: string | null;
      while ((line = nextLine()) !== null) {
        const event = handleLine(line);
        if (event !== null) yield event;
      }
    }
    // End of stream: the remaining buffer is one final (unterminated) line.
    buffer += decoder.decode();
    if (pendingLF && buffer.startsWith("\n")) buffer = buffer.slice(1);
    if (buffer !== "") handleLine(buffer); // non-blank, so it never dispatches
    const tail = takeEvent();
    if (tail !== null) yield tail;
    finished = true;
  } finally {
    if (!finished) {
      // The consumer stopped early (break, return, throw): cancel the
      // source so the HTTP connection closes and the server can stop the
      // run, instead of leaving it generating for nobody.
      try {
        await reader.cancel();
      } catch {
        // Cancellation is best-effort; the lock is released below.
      }
    }
    reader.releaseLock();
  }
}
