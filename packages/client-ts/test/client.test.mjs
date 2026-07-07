import assert from "node:assert/strict";
import { test } from "node:test";

import { collectRun, isDone, isError, parseSSEStream, streamEvents } from "../dist/index.js";

function byteStream(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

const SSE_BODY =
  'event: text_delta\ndata: {"type": "text_delta", "text": "hel"}\n\n' +
  'event: text_delta\ndata: {"type": "text_delta", "text": "lo"}\n\n' +
  'event: step\ndata: {"type": "step", "id": "s1", "title": "Searched", "detail": null, "phase": null, "status": "done"}\n\n' +
  'event: usage\ndata: {"type": "usage", "usage": {"input_tokens": 3, "output_tokens": 2, "requests": 1}}\n\n' +
  'event: done\ndata: {"type": "done", "text": "hello", "output": null, "usage": {"input_tokens": 3, "output_tokens": 2, "requests": 1}, "truncated": false}\n\n';

test("parses a well-formed stream into typed events", async () => {
  const events = [];
  for await (const event of parseSSEStream(byteStream([SSE_BODY]))) {
    events.push(event);
  }
  assert.deepEqual(
    events.map((e) => e.type),
    ["text_delta", "text_delta", "step", "usage", "done"],
  );
  assert.equal(events[0].text, "hel");
  assert.equal(events[2].title, "Searched");
  assert.equal(events[4].usage.input_tokens, 3);
});

test("handles chunk boundaries mid-line and CRLF", async () => {
  const crlfBody = SSE_BODY.replaceAll("\n", "\r\n");
  // split into awkward 7-byte chunks
  const chunks = [];
  for (let i = 0; i < crlfBody.length; i += 7) chunks.push(crlfBody.slice(i, i + 7));
  const events = [];
  for await (const event of parseSSEStream(byteStream(chunks))) {
    events.push(event);
  }
  assert.equal(events.length, 5);
  assert.equal(events.at(-1).type, "done");
});

test("ignores comments and flushes trailing message without blank line", async () => {
  const body = ':keepalive\n\ndata: {"type": "text_delta", "text": "x"}';
  const events = [];
  for await (const event of parseSSEStream(byteStream([body]))) {
    events.push(event);
  }
  assert.deepEqual(events, [{ type: "text_delta", text: "x" }]);
});

test("throws on malformed JSON data", async () => {
  const body = "data: {not json}\n\n";
  await assert.rejects(async () => {
    for await (const _ of parseSSEStream(byteStream([body]))) {
      void _;
    }
  }, /malformed event data/);
});

test("streamEvents fetches with SSE accept header and yields events", async () => {
  let seenRequest;
  const fakeFetch = async (input, init) => {
    seenRequest = { input, init };
    return new Response(byteStream([SSE_BODY]), { status: 200 });
  };
  const events = [];
  for await (const event of streamEvents("https://api.example/stream", {
    method: "POST",
    body: '{"prompt":"hi"}',
    fetch: fakeFetch,
  })) {
    events.push(event);
  }
  assert.equal(events.length, 5);
  assert.equal(seenRequest.init.headers.accept, "text/event-stream");
  assert.equal(seenRequest.init.method, "POST");
});

test("streamEvents throws on non-2xx", async () => {
  const fakeFetch = async () => new Response("nope", { status: 503 });
  await assert.rejects(async () => {
    for await (const _ of streamEvents("https://x", { fetch: fakeFetch })) {
      void _;
    }
  }, /HTTP 503/);
});

test("collectRun returns the done event and guards narrow types", async () => {
  const done = await collectRun(parseSSEStream(byteStream([SSE_BODY])));
  assert.equal(done.text, "hello");
  assert.ok(isDone(done));
  assert.ok(!isError(done));
});

test("collectRun throws on error terminal event", async () => {
  const body = 'data: {"type": "error", "message": "boom", "code": "model_error", "retryable": true}\n\n';
  await assert.rejects(collectRun(parseSSEStream(byteStream([body]))), /run failed: boom/);
});
