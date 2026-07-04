/**
 * Consuming a shankit agent stream from TypeScript.
 *
 * npm install @shankit/client
 * (pair with examples/06_sse_server.py)
 */

import { streamEvents } from "@shankit/client";

for await (const event of streamEvents("http://localhost:8000/agents/support/stream", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ prompt: "Where is order A-100?", user_id: "u1" }),
})) {
  switch (event.type) {
    case "text_delta":
      process.stdout.write(event.text);
      break;
    case "step":
      console.log(`\n[${event.status}] ${event.title}`);
      break;
    case "source":
      console.log(`\nsource: ${event.source.title}`);
      break;
    case "done":
      console.log(`\n(tokens: ${event.usage.input_tokens} in / ${event.usage.output_tokens} out)`);
      break;
    case "error":
      console.error(`run failed: ${event.message}`);
      break;
  }
}
