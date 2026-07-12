<div align="center">

# shankit

**The AI agent framework that takes the _integration seam_ — the boundary between an agent and the real, authenticated tools it acts on — as seriously as the agent loop itself. Without imposing how you authenticate.**

[![CI](https://github.com/eeshankulk1/shankit/actions/workflows/ci.yml/badge.svg)](https://github.com/eeshankulk1/shankit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

[Quick start](#quick-start) · [Why shankit](#why-shankit) · [Concepts](docs/concepts.md) · [Docs](docs/README.md) · [Examples](examples) · [Design record](docs/design.md)

</div>

---

Most frameworks make the agent loop the star and treat _"where do the tools come from"_ as the developer's problem. shankit inverts that: the framework owns the **shape** of the seam — tool format, error contract, observability, per-run context — and never the **mechanism** — auth, credential storage, vendor choice. Bring your own auth; shankit won't have an opinion about it.

It ships one small, typed core (one agent, two run modes, one tool seam) plus the satellites a real product needs — file-first agent definitions, multi-agent orchestration, durability, evals, and a typed TypeScript stream consumer — and nothing it doesn't.

## Table of contents

- [Why shankit](#why-shankit)
- [Install](#install)
- [Quick start](#quick-start)
- [File-first definitions](#file-first-definitions)
- [Multi-agent: agents as tools](#multi-agent-agents-as-tools)
- [Connectors: the opt-in integration layer](#connectors-the-opt-in-integration-layer)
- [Streaming to TypeScript](#streaming-to-typescript)
- [What else is in the box](#what-else-is-in-the-box)
- [Examples](#examples)
- [Documentation](#documentation)
- [How shankit compares](#how-shankit-compares)
- [Development](#development)
- [Contributing](#contributing)
- [Project status](#project-status)
- [Telemetry](#telemetry)
- [License](#license)

## Why shankit

- **Integration seam as a first-class contract.** The entire tool boundary is _two capabilities_ — `list_tools` and `execute` — in Anthropic tool shape, receiving nothing but an opaque per-run context. Local functions, MCP servers, sub-agents, and OAuth connectors all sit behind the same seam.
- **Auth stays yours.** Nothing in the core mentions authentication. A tool resolves _whose_ credentials to use by reading your plain data off the per-run context; the loop never learns what a "user" is.
- **One loop, two run modes.** The _same_ agent runs `run()` for a typed, validated deliverable or `stream()` for a live event stream (text deltas, steps, sources, usage). The split is output shape, chosen by the caller — not two engines to keep in sync.
- **File-first definitions.** Author agents as prose-plus-frontmatter files that mirror Claude Code subagents — diffable, reviewable, no new concepts. Drop to Python the moment you need real types or logic.
- **Observability as a product surface.** A pluggable step-describer turns each tool call into user-facing narration on the stream — not just developer tracing (OpenTelemetry is available and orthogonal).
- **Durable by contract, not by platform.** A `Checkpointer` protocol backed by _your_ storage (in-memory and SQLite ship in-box) powers long-running agents and human-in-the-loop approval pauses. No hosted runtime to buy into.
- **Provider-neutral.** Anthropic and OpenAI at launch behind one `ModelClient` contract; `register_provider()` adds more, or pass your own client.
- **Python core, typed TypeScript consumer.** `@shankit/client` consumes agent streams over SSE with event types generated from the Python source of truth — CI fails on drift.

## Install

```bash
pip install shankit[anthropic]            # or [openai], [mcp], [otel], [all]
pip install shankit-connectors[composio]  # opt-in OAuth connector layer
npm install @shankit/client               # typed TypeScript stream consumer
```

> **Not yet published to PyPI / npm.** Install from source for now:
> ```bash
> pip install -e packages/shankit
> ```

## Quick start

### 1. Your first agent

A tool is a plain function. The agent is a model, instructions, and tools. Run it structured for a typed result, or stream it for a live conversation — over the same loop.

```python
import asyncio
from pydantic import BaseModel
from shankit import Agent, ToolError, tool

@tool
def list_orders(ctx) -> list:
    """List the acting user's orders."""
    # `ctx` is the opaque per-run context: YOUR plain data, threaded to tools.
    # The framework never reads inside it — this is how a tool knows *whose*
    # orders to fetch without the loop knowing what a "user" is.
    return db.orders_for(ctx["user_id"])

class OrdersSummary(BaseModel):
    open_orders: int
    headline: str

agent = Agent(
    name="orders",
    model="anthropic:claude-sonnet-4-5",   # or "openai:gpt-4.1", or bring your own ModelClient
    instructions="You help users manage their orders. Be concise.",
    tools=[list_orders],
)

async def main():
    # Structured: a typed, validated deliverable (the caller chooses the shape).
    result = await agent.run(
        "Summarize my orders.", context={"user_id": "u1"}, output_type=OrdersSummary
    )
    print(result.output)      # OrdersSummary(open_orders=2, headline="...")
    print(result.trajectory)  # every tool call, for evals
    print(result.usage)       # tokens, including any sub-agents

asyncio.run(main())
```

### 2. Stream the same agent

Streaming yields a typed event stream — the single source of truth the TypeScript client renders and that your SSE route maps onto directly.

```python
async for event in agent.stream("Where is order 42?", context={"user_id": "u1"}):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "step":
        print(f"\n[{event.status}] {event.title}")   # user-facing narration
    elif event.type == "error":
        print(f"\n[error] {event.message}")          # calm, human-safe terminal event
    elif event.type == "done":
        print(f"\n({event.usage.input_tokens} in / {event.usage.output_tokens} out)")
```

Tool failures follow one uniform contract: raise `ToolError("message the model sees")` for a controlled failure; any other exception is sanitized before the model sees it and logged in full for you. A failed provider call surfaces as `ModelError` (with `retryable`) — never a raw SDK exception across the neutral boundary.

## File-first definitions

The headline authoring surface. Frontmatter mirrors Claude Code subagents; the one addition is `output_schema`, whose presence lets the agent be run structured. The body is a prompt with fill-in-the-blank substitution only — the moment it needs an `if`, define the agent in Python instead.

```markdown
---
name: email-triage
description: Triages the inbox and proposes what to do next.
model: anthropic:claude-sonnet-4-5
tools:
  - myapp.tools:search_email     # standard module:attribute import reference
  - gmail                        # a live object you explicitly registered
output_schema: myapp.schemas:TriagePlan
---
You are an email triage assistant for {user_name}.
Work through the inbox with your tools, then deliver a triage plan.
```

```python
from shankit import load_agent, register

register("gmail", gmail_connector)      # only non-importable live objects need this
agent = load_agent("agents/triage.md")  # → the SAME Agent the constructor makes
```

A file produces the identical `Agent` object the Python constructor would — the file is sugar, never a second code path. See the [file-first guide](docs/file-first.md) for the full frontmatter reference.

## Multi-agent: agents as tools

Handing sub-agents to a parent agent _is_ orchestration — with routing, sanitized sub-agent failures, source propagation, and cross-agent token accounting built into the seam:

```python
oncall = Agent(
    name="oncall",
    model="anthropic:claude-sonnet-4-5",
    instructions="Coordinate specialists to answer on-call questions.",
    tools=[docs_agent.as_tool(), metrics_agent.as_tool()],
)
```

When **code**, not the model, must control flow — deterministic routing, cycles, shared state, and durable human-in-the-loop pauses — there's an experimental router-driven [network](docs/durability.md#networks-experimental). The state the router reads _is_ what the checkpointer persists, so approval pauses fall out of the design rather than being bolted on.

## Connectors: the opt-in integration layer

Local-function and MCP users never touch this. For tools that need OAuth, the separate [`shankit-connectors`](packages/shankit-connectors) package wraps a tool source and owns the connection lifecycle (initiate / check-status / adopt), shipping one official implementation — Composio.

```python
from shankit_connectors import ComposioConnector

gmail = ComposioConnector(
    toolkit="GMAIL",
    user_id=lambda ctx: ctx["user_id"],   # you decide what identity means
)
agent = Agent(name="mail", model="anthropic:claude-sonnet-4-5", tools=[gmail])
```

The litmus for this layer: **you could swap Composio for a completely different system without the framework noticing.**

## Streaming to TypeScript

```python
# FastAPI (any ASGI framework works)
return StreamingResponse(sse_stream(agent.stream(prompt, context=ctx)),
                         media_type="text/event-stream")
```

```ts
import { streamEvents } from "@shankit/client";

for await (const event of streamEvents("/agents/support/stream", { method: "POST", body })) {
  if (event.type === "text_delta") render(event.text);
  if (event.type === "step")       showStep(event.title, event.status);
  if (event.type === "error")      showError(event.message);
}
```

Event types are **generated** from the Python pydantic models, so the two languages cannot drift — CI enforces it.

## What else is in the box

| | |
|---|---|
| **Tool sources** | `@tool` local functions, any MCP server (`MCPToolSource`), aggregation (`CompositeToolSource`), sub-agents, and connectors — all behind one two-method seam in Anthropic tool shape |
| **Observability** | pluggable step-describer → user-facing step narration on the event stream; optional OpenTelemetry spans for dev tracing |
| **Durability** | `Checkpointer` contract backed by _your_ storage; in-memory + SQLite ship in-box; powers HITL approval pauses |
| **Evals** | datasets, scorers (`exact_match`, `output_contains`, `llm_judge`), and trajectory assertions over `result.trajectory` |
| **Models** | provider-neutral `ModelClient`; Anthropic + OpenAI at launch, `register_provider()` for more |
| **Connectors** | opt-in [`shankit-connectors`](packages/shankit-connectors): connection lifecycle + official Composio implementation |
| **TypeScript** | [`@shankit/client`](packages/client-ts): typed SSE consumption, event types generated from the Python source of truth |

## Examples

Every major surface has a runnable example in [`examples/`](examples):

| Example | Shows |
|---|---|
| [`01_quickstart.py`](examples/01_quickstart.py) | tools, per-run context, both run modes, uniform tool errors |
| [`02_file_first/`](examples/02_file_first) | file-first definitions, import refs, prompt substitution |
| [`03_multi_agent.py`](examples/03_multi_agent.py) | agents as tools: sanitized failures, sources, token accounting |
| [`04_act_with_approval.py`](examples/04_act_with_approval.py) | experimental network: deterministic routing, durable HITL approval |
| [`05_evals.py`](examples/05_evals.py) | datasets, scorers, trajectory assertions |
| [`06_sse_server.py`](examples/06_sse_server.py) | serving the event stream over SSE (FastAPI) |
| [`07_ts_consumer.ts`](examples/07_ts_consumer.ts) | consuming the stream with `@shankit/client` |

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- **[Getting started](docs/getting-started.md)** — install, your first agent, both run modes.
- **[Core concepts](docs/concepts.md)** — the tool seam, per-run context, run modes, multi-agent, models, errors.
- **[File-first definitions](docs/file-first.md)** — frontmatter reference, substitution, import vs. registry refs.
- **[Connectors](docs/connectors.md)** — the OAuth connector contract, the connection lifecycle, Composio.
- **[Durability & networks](docs/durability.md)** — the checkpointer contract and the experimental graph.
- **[Evals](docs/evals.md)** — datasets, scorers, model-graded judging, trajectory assertions.
- **[Design decision record](docs/design.md)** — why the framework is shaped this way (the living spec).

## How shankit compares

shankit's differentiation is **orientation**, not raw primitives — four things the leaders don't optimize for _together_: a vendor-neutral connector contract, observability as a user-facing surface, file-first definitions, and a paired Python-core + TypeScript-consumer SDK. On core agent features it aims for parity, not novelty.

| Framework | Identity | Where shankit differs |
|---|---|---|
| **LangGraph** | low-level graph runtime; state/checkpointing/HITL are the product | shankit keeps the graph an _experimental escape hatch_, not the headline; the default is agents-as-tools |
| **Agno** | batteries-included multi-agent runtime (memory/RAG/knowledge in-box) | shankit deliberately punts memory/RAG/scheduling to your app |
| **Pydantic AI** | type-safe, Python-first — the closest neighbour | shankit matches the core and adds the connector seam, file-first authoring, and the TS consumer |
| **Inngest AgentKit** | TS-first networks welded to a durable engine | shankit's network borrows the router model but stays storage-agnostic and un-platformed |

**shankit is _not_** a "better LangChain," a hosted platform, or opinionated about auth/memory/RAG/scheduling. Those are your app's concern; the framework leaves room without shipping an opinion. See [`docs/design.md`](docs/design.md) for the full landscape and rationale.

## Development

```bash
uv sync --all-packages          # both packages editable + all dev tools, pinned by uv.lock

uv run pytest packages          # core + connector tests, no network
uv run ruff check packages scripts examples && uv run ruff format --check packages scripts examples
uv run pyright                  # the types are part of the public contract
uv run python scripts/generate_ts_events.py --check   # TS event types in sync
cd packages/client-ts && npm ci && npm test
```

The test suite fakes the LLM, Composio, and network boundaries, so it runs anywhere without credentials.

## Contributing

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the dev setup, the test/lint gates every PR must pass, and the one rule that governs new API: _the framework is never more than one real usage ahead of itself_ — no abstraction ships without a consumer that needs it today.

## Project status

**Alpha (`0.1.0`), pre-1.0, API not yet frozen.** shankit is being _harvested_ from a production product ([throu](https://throu.ai)) rather than designed in a vacuum — the public API freezes only after throu fully runs on it (see [design §12](docs/design.md#12-rollout-plan)). `shankit.experimental.*` ships with no stability guarantee; the namespace _is_ the flag.

> **On the name:** `shankit` is a working title carried across the packages. A final name will be chosen before the first tagged release; renaming is a mechanical find-replace.

## Telemetry

**shankit collects no telemetry.** It makes no network calls of its own — only the LLM and tool calls you configure. Optional OpenTelemetry spans are emitted only to an exporter _you_ wire up.

## License

[MIT](LICENSE)
</content>
</invoke>
