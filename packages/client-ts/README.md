# @shankit/client

Typed TypeScript consumer for [shankit](https://github.com/eeshankulk1/shankit) agent event streams over SSE/HTTP. It consumes agent runs; it does not run agents.

Event types are **generated** from the Python pydantic models (`shankit.events`), so the two languages cannot drift — CI enforces it.

## Install

```bash
npm install @shankit/client
```

Zero runtime dependencies. Node >= 18 (or any runtime with `fetch` and Web Streams).

## Usage

```ts
import { streamEvents } from "@shankit/client";

for await (const event of streamEvents("/agents/support/stream", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ prompt }),
})) {
  switch (event.type) {
    case "text_delta": render(event.text); break;
    case "step":       showStep(event.title, event.status); break;
    case "error":      showError(event.message); break;
    case "done":       finish(event); break;
  }
}
```

Breaking out of the loop cancels the underlying HTTP stream, so the server can stop the run.

For non-streaming consumers:

```ts
import { collectRun, streamEvents } from "@shankit/client";

const done = await collectRun(streamEvents(url, options));
console.log(done.text, done.usage);
```

- `parseSSEStream(body)` — parse any `ReadableStream<Uint8Array>` of shankit SSE messages (bring your own transport).
- `isDone(event)` / `isError(event)` — type-narrowing helpers.

The parser targets shankit's wire format (JSON in `data:`, type discriminant in the payload). It handles LF/CRLF/CR line endings and multi-line `data` fields; it does not implement `id:`/`retry:` reconnection semantics — runs are one-shot POST streams.

## Server side

```python
# FastAPI (any ASGI framework works)
from shankit import sse_stream

return StreamingResponse(
    sse_stream(agent.stream(prompt, context=ctx)),
    media_type="text/event-stream",
)
```

See the [shankit docs](https://github.com/eeshankulk1/shankit/blob/main/docs/README.md) for the full framework.
