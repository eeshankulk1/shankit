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

// Golden fixture: the exact bytes shankit.sse.format_sse emits (compact
// JSON, all Usage fields present, LF endings). Regenerate with:
//   python -c "from shankit import format_sse; ..."
const SSE_BODY =
  'event: text_delta\ndata: {"type":"text_delta","text":"hel"}\n\n' +
  'event: text_delta\ndata: {"type":"text_delta","text":"lo"}\n\n' +
  'event: step\ndata: {"type":"step","id":"s1","title":"Searched","detail":null,"phase":null,"status":"done"}\n\n' +
  'event: usage\ndata: {"type":"usage","usage":{"input_tokens":3,"output_tokens":2,"cache_read_tokens":0,"cache_write_tokens":0,"requests":1}}\n\n' +
  'event: done\ndata: {"type":"done","text":"hello","output":null,"usage":{"input_tokens":3,"output_tokens":2,"cache_read_tokens":0,"cache_write_tokens":0,"requests":1},"truncated":false}\n\n';

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
  assert.equal(events[4].usage.cache_read_tokens, 0);
});

test("handles chunk boundaries mid-line and CRLF", async () => {
  const crlfBody = SSE_BODY.replaceAll("\n", "\r\n");
  // split into awkward 7-byte chunks (this also splits CRLF pairs)
  const chunks = [];
  for (let i = 0; i < crlfBody.length; i += 7) chunks.push(crlfBody.slice(i, i + 7));
  const events = [];
  for await (const event of parseSSEStream(byteStream(chunks))) {
    events.push(event);
  }
  assert.equal(events.length, 5);
  assert.equal(events.at(-1).type, "done");
});

test("handles lone-CR line endings and mixed endings", async () => {
  const body =
    'data: {"type":"text_delta","text":"a"}\r\r' +
    'data: {"type":"text_delta","text":"b"}\r\n\n';
  const events = [];
  for await (const event of parseSSEStream(byteStream([body]))) {
    events.push(event);
  }
  assert.deepEqual(
    events.map((e) => e.text),
    ["a", "b"],
  );
});

test("LF data lines with a CRLF blank line do not merge adjacent messages", async () => {
  const body =
    'data: {"type":"text_delta","text":"a"}\n\r\n' +
    'data: {"type":"text_delta","text":"b"}\n\r\n';
  const events = [];
  for await (const event of parseSSEStream(byteStream([body]))) {
    events.push(event);
  }
  assert.deepEqual(
    events.map((e) => e.text),
    ["a", "b"],
  );
});

test("joins multi-line data fields per spec", async () => {
  const body = "data: {\ndata: \"type\": \"text_delta\", \"text\": \"x\"\ndata: }\n\n";
  const events = [];
  for await (const event of parseSSEStream(byteStream([body]))) {
    events.push(event);
  }
  assert.deepEqual(events, [{ type: "text_delta", text: "x" }]);
});

test("ignores comments and flushes trailing message without blank line", async () => {
  const body = ':keepalive\n\ndata: {"type":"text_delta","text":"x"}';
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

test("breaking out of the stream cancels the source", async () => {
  let cancelled = false;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(SSE_BODY));
      // Stream stays open: a live run would keep producing.
    },
    cancel() {
      cancelled = true;
    },
  });
  for await (const event of parseSSEStream(stream)) {
    if (event.type === "step") break; // consumer stops early
  }
  assert.ok(cancelled, "early exit must cancel the underlying stream");
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
  assert.equal(seenRequest.init.headers.get("accept"), "text/event-stream");
  assert.equal(seenRequest.init.method, "POST");
});

test("streamEvents preserves caller headers passed as a Headers instance", async () => {
  let seenRequest;
  const fakeFetch = async (input, init) => {
    seenRequest = { input, init };
    return new Response(byteStream([SSE_BODY]), { status: 200 });
  };
  const events = [];
  for await (const event of streamEvents("https://api.example/stream", {
    method: "POST",
    headers: new Headers({ "content-type": "application/json", authorization: "Bearer t" }),
    body: '{"prompt":"hi"}',
    fetch: fakeFetch,
  })) {
    events.push(event);
  }
  assert.equal(events.length, 5);
  assert.equal(seenRequest.init.headers.get("content-type"), "application/json");
  assert.equal(seenRequest.init.headers.get("authorization"), "Bearer t");
  assert.equal(seenRequest.init.headers.get("accept"), "text/event-stream");
});

test("streamEvents lets an explicit accept header win", async () => {
  let seenRequest;
  const fakeFetch = async (input, init) => {
    seenRequest = { input, init };
    return new Response(byteStream([SSE_BODY]), { status: 200 });
  };
  for await (const _ of streamEvents("https://api.example/stream", {
    headers: [["accept", "text/event-stream; charset=utf-8"]],
    fetch: fakeFetch,
  })) {
    void _;
  }
  assert.equal(seenRequest.init.headers.get("accept"), "text/event-stream; charset=utf-8");
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

test("collectRun cancels the source once done arrives", async () => {
  let cancelled = false;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(SSE_BODY));
      // Stream stays open after `done` (e.g. keep-alive comments would follow).
    },
    cancel() {
      cancelled = true;
    },
  });
  const done = await collectRun(parseSSEStream(stream));
  assert.equal(done.type, "done");
  assert.ok(cancelled, "collectRun must release the connection after the terminal event");
});

test("collectRun throws on error terminal event", async () => {
  const body =
    'data: {"type":"error","message":"boom","code":"model_error","retryable":true}\n\n';
  await assert.rejects(collectRun(parseSSEStream(byteStream([body]))), /run failed: boom/);
});
