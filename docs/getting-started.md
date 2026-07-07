# Getting started

This guide takes you from an empty environment to a working agent you can run
two ways. It assumes you're comfortable with Python and `async`/`await`.

## Install

shankit is a small core with opt-in extras — you only pull in the provider SDKs
you actually use.

```bash
pip install shankit[anthropic]   # Anthropic models
pip install shankit[openai]      # OpenAI models
pip install shankit[mcp]         # MCP tool servers
pip install shankit[otel]        # OpenTelemetry tracing
pip install shankit[all]         # everything above
```

> **Not yet on PyPI.** Until the first release, install from source:
> ```bash
> pip install -e packages/shankit
> ```

Set the API key for whichever provider you're using:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Your first agent

Three things make an agent: a **model**, **instructions**, and **tools**. A tool
is just a function decorated with `@tool`; its docstring is what the model sees.

```python
import asyncio
from pydantic import BaseModel, Field
from shankit import Agent, ToolError, tool

# --- Tools are code. ------------------------------------------------------
FAKE_ORDERS = {
    "u1": [
        {"id": "A-100", "item": "keyboard", "status": "shipped"},
        {"id": "A-101", "item": "usb hub", "status": "processing"},
    ]
}

@tool
def list_orders(ctx) -> list:
    """List the acting user's orders."""
    # `ctx` is the opaque per-run context. The framework never reads inside it —
    # whose orders these are is YOUR concern, resolved from your own data.
    return FAKE_ORDERS.get(ctx["user_id"], [])

@tool
def cancel_order(ctx, order_id: str) -> str:
    """Cancel an order that has not shipped yet."""
    for order in FAKE_ORDERS.get(ctx["user_id"], []):
        if order["id"] == order_id:
            if order["status"] == "shipped":
                raise ToolError(f"Order {order_id} already shipped; cannot cancel.")
            order["status"] = "cancelled"
            return f"Cancelled {order_id}."
    raise ToolError(f"No order {order_id} for this user.")

# --- The agent. -----------------------------------------------------------
agent = Agent(
    name="orders-assistant",
    model="anthropic:claude-sonnet-4-5",   # provider prefix is required
    instructions="You help users manage their orders. Be concise.",
    tools=[list_orders, cancel_order],
)
```

> The `model` string **always** carries a provider prefix (`anthropic:...`,
> `openai:...`). There is no silent default vendor. To use a provider shankit
> doesn't know, register it with `register_provider()` or pass your own
> `model_client=`.

## Run mode 1 — structured

Call `run()` with an `output_type` to get a typed, validated deliverable. This
is the path for skills, background jobs, and anything that produces a result
rather than a conversation.

```python
class OrdersSummary(BaseModel):
    open_orders: int = Field(description="Orders not yet delivered or cancelled")
    headline: str

async def main():
    result = await agent.run(
        "Summarize my orders.",
        context={"user_id": "u1"},
        output_type=OrdersSummary,
    )
    print(result.output)      # → OrdersSummary(open_orders=2, headline="...")
    print(result.trajectory)  # → every tool call the run made, for evals
    print(result.usage)       # → token usage, including any sub-agents

asyncio.run(main())
```

`result.output` is the **answer** (validated against your schema).
`result.text` is the **transcript** — every assistant text pass joined, including
any interim narration before tool calls. Score and act on `output`; display and
persist `text`.

## Run mode 2 — streaming

Call `stream()` for a live, typed event stream — text deltas, user-facing steps,
sources, usage, and a terminal `done` (or `error`). This is the path for chat
UIs and anything a human watches in real time.

```python
async for event in agent.stream("Cancel my usb hub order.", context={"user_id": "u1"}):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "step":
        print(f"\n[{event.status}] {event.title}")   # e.g. "[done] Cancelled an order"
    elif event.type == "done":
        print(f"\n({event.usage.input_tokens} in / {event.usage.output_tokens} out)")
```

Both modes run the **same agent over the same loop**. Whether to ask for
structured output is the caller's choice at call time — there is no "chat agent"
vs. "skill agent" distinction baked into the object.

## Handling failure

Tool and model failures follow one uniform contract:

- **Controlled tool failure** — `raise ToolError("message the model sees")`. The
  model receives exactly that text and can recover.
- **Unexpected tool failure** — any other exception is caught, logged in full,
  and sanitized to a generic message before the model sees it. Secrets and
  tracebacks never leak into the transcript.
- **Provider failure** — a failed model call surfaces as `shankit.ModelError`
  (with `provider`, `status`, and a `retryable` flag), never a raw provider SDK
  exception. In `stream()`, framework errors arrive as a terminal `error` event.

## Next steps

- **[Core concepts](concepts.md)** — the tool seam, the per-run context, and how
  the two run modes share one loop.
- **[File-first definitions](file-first.md)** — author this same agent as a
  reviewable Markdown file.
- **[Examples](../examples)** — runnable scripts for multi-agent, evals, SSE, and
  the experimental network.
</content>
