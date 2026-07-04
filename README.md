# shankit

> Working name. The real name is unresolved — see [`docs/design.md` §13](docs/design.md).

An open-source AI agent framework that takes the **integration seam** — the boundary
between an agent and the real, authenticated external tools it acts on — as seriously
as the agent loop itself, **without imposing how you authenticate.**

Most frameworks make the agent loop the star and treat "where do the tools come from"
as the developer's problem. shankit inverts that: the framework owns the *shape* of the
seam (tool format, error contract, observability, per-run context) and never the
*mechanism* (auth, credential storage, vendor choice).

## Guiding principles

1. **Minimal core, escape hatches over abstraction.** A small set of primitives with
   concrete defaults. When power is needed, add an escape hatch — never a heavier
   abstraction imposed on everyone.
2. **Definitions are prose-first.** Behavior is authored as prose plus a few
   declarative knobs (the Claude Code ergonomic), not builder-pattern code. Code
   appears only where it's genuinely required: tool and connector implementations.
3. **The framework defines contracts, not systems.**

## Install

```bash
pip install shankit[anthropic]          # or [openai], [mcp], [otel], [all]
pip install shankit-connectors[composio]  # opt-in connector layer
npm install @shankit/client               # TypeScript stream consumer
```

> Not yet published to PyPI/npm — install from this repo for now:
> `pip install -e packages/shankit`.

## The core in one file

```python
from pydantic import BaseModel
from shankit import Agent, ToolError, tool

@tool
def search_orders(ctx, query: str) -> str:
    """Search the order database."""
    # ctx is the opaque per-run context: your plain data, threaded to tools.
    # The framework never reads inside it — this is how a tool resolves
    # *whose* credentials to use without the loop knowing what a "user" is.
    return db.search(query, user_id=ctx["user_id"])

class Summary(BaseModel):
    headline: str
    open_orders: int

agent = Agent(
    name="orders",
    model="anthropic:claude-sonnet-4-5",   # or "openai:gpt-4.1", or bring your own ModelClient
    instructions="You help users with their orders.",
    tools=[search_orders],
)

# One agent, two run modes over the same loop — the split is output shape,
# and it's the CALLER's choice:

# 1. Structured: a typed, validated deliverable.
result = await agent.run("Summarize my orders", context={"user_id": "u1"}, output_type=Summary)
result.output      # Summary(...)
result.trajectory  # every tool call, for evals
result.usage       # tokens, including any sub-agents

# 2. Streaming: a typed event stream (text deltas, steps, sources, usage, done).
async for event in agent.stream("Where is order 42?", context={"user_id": "u1"}):
    ...
```

Tool failures follow one uniform contract: raise `ToolError("message the model sees")`
for a controlled failure; anything else is sanitized before the model sees it and
logged in full for you.

## File-first definitions

The headline authoring surface. Frontmatter mirrors Claude Code subagents;
the one addition is `output_schema`, whose presence lets the agent be run
structured. The body is a prompt with fill-in-the-blank substitution only —
the moment a prompt needs an `if`, define the agent in Python instead.

```markdown
---
name: email-triage
description: Triages the inbox.
model: anthropic:claude-sonnet-4-5
tools:
  - myapp.tools:search_email     # standard module:attribute import reference
  - gmail                        # a live object you explicitly registered
output_schema: myapp.schemas:TriagePlan
---
You are an email triage assistant for {user_name}.
```

```python
from shankit import load_agent, register

register("gmail", gmail_connector)          # only non-importables need this
agent = load_agent("agents/triage.md")      # → the same Agent the constructor makes
```

## Multi-agent: agents as tools

Handing sub-agents to a parent agent *is* orchestration — with routing,
sanitized sub-agent failures, source propagation, and cross-agent token
accounting built into the seam:

```python
oncall = Agent(
    name="oncall",
    model="anthropic:claude-sonnet-4-5",
    tools=[docs_agent.as_tool(), metrics_agent.as_tool()],
)
```

## When code must control flow (experimental)

For deterministic routing, cycles, shared state, and human-in-the-loop
pauses, there's a router-driven network — a loop over shared state, not a
node/edge DSL. The state the router reads **is** what the checkpointer
persists, so durable approval pauses fall out of the design:

```python
from shankit import SqliteCheckpointer
from shankit.experimental.graph import Interrupt, Network, agent_step

net = Network(
    name="act-with-approval",
    steps={
        "gather": agent_step(researcher, prompt="Context for: {task}", output_key="notes"),
        "draft":  agent_step(writer, prompt="Draft it.\n{notes}", output_key="draft"),
        "act":    send_email,                       # any (state, ctx) callable
    },
    router=route,                                   # code decides; returns a step name,
    checkpointer=SqliteCheckpointer("threads.db"),  # an Interrupt, or None to stop
)

result = await net.run({"task": "reply to Bob"}, thread_id="t1")
if result.status == "interrupted":
    ...                                             # show the draft to a human — hours later, another process:
    result = await net.resume("t1", value=True)
```

`shankit.experimental.*` ships in v1 with no stability guarantee; the
namespace is the flag.

## What else is in the box

| | |
|---|---|
| **Tool sources** | `@tool` local functions, any MCP server (`MCPToolSource`), aggregation, sub-agents, connectors — all behind one two-method seam (`list_tools` / `execute`) in Anthropic tool shape |
| **Observability** | pluggable step-describer → user-facing step narration on the event stream; OpenTelemetry spans (optional) are orthogonal dev tracing |
| **Durability** | `Checkpointer` contract backed by *your* storage; in-memory + SQLite ship in-box |
| **Evals** | datasets, scorers (incl. `llm_judge`), trajectory assertions over `result.trajectory` |
| **Models** | provider-neutral `ModelClient`; Anthropic + OpenAI at launch, `register_provider()` for more |
| **Connectors** | opt-in [`shankit-connectors`](packages/shankit-connectors) package: connection lifecycle (initiate / check-status / adopt) + official Composio implementation |
| **TypeScript** | [`@shankit/client`](packages/client-ts): typed SSE consumption, event types generated from the Python source of truth |

## Serving a stream to TypeScript

```python
# FastAPI (any ASGI framework works)
return StreamingResponse(sse_stream(agent.stream(prompt, context=ctx)),
                         media_type="text/event-stream")
```

```ts
import { streamEvents } from "@shankit/client";

for await (const event of streamEvents("/agents/support/stream", { method: "POST", body })) {
  if (event.type === "text_delta") render(event.text);
  if (event.type === "step") showStep(event.title, event.status);
}
```

## Repository layout

```
packages/
  shankit/              # Python core (the framework)
  shankit-connectors/   # opt-in connector layer + Composio
  client-ts/            # @shankit/client TypeScript consumer SDK
examples/               # runnable examples for every major surface
scripts/                # TS event-type generation from the Python models
docs/design.md          # the design decision record
```

## Development

```bash
uv venv && uv pip install -e packages/shankit -e packages/shankit-connectors \
    pytest pytest-asyncio anthropic openai ruff
uv run pytest packages/shankit/tests packages/shankit-connectors/tests
uv run ruff check packages scripts
python scripts/generate_ts_events.py --check   # TS event types in sync
cd packages/client-ts && npm install && npm test
```

## Status

v1 implementation of the design in [`docs/design.md`](docs/design.md) — the living
decision record. The framework is being harvested from an existing product
([throu](https://throu.ai)) rather than designed in a vacuum; public API freezes
only after throu fully runs on it (design §12).

## License

[MIT](LICENSE)
