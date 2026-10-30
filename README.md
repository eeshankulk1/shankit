<div align="center">

# Throukit

**A small, provider-neutral Python framework for AI agents that act on real, signed-in tools.**

[![CI](https://github.com/throuai/throukit/actions/workflows/ci.yml/badge.svg)](https://github.com/throuai/throukit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Typed](https://img.shields.io/badge/typed-pyright-blue.svg)](pyproject.toml)

[Docs](docs/README.md) · [Getting started](docs/getting-started.md) · [Concepts](docs/concepts.md) · [Examples](examples) · [Changelog](CHANGELOG.md)

</div>

---

Most agent frameworks put the loop at the center and leave "where do the tools come from, and whose account do they act on" to you. Throukit treats that boundary, the **integration seam**, as part of the framework: one tool format, one error contract, one per-run context, one event stream. It never decides how you authenticate.

The core is one agent with two run modes and a two-method tool interface. Around it sit the pieces a real product needs: agents defined in Markdown files, sub-agents, durable runs with approval pauses, evals, and a typed TypeScript client.

## Install

```bash
pip install "throukit[anthropic]"            # or [openai], [mcp], [otel], [all]
pip install "throukit-connectors[composio]"  # optional OAuth connector layer
npm install @throukit/client                 # typed TypeScript stream consumer
```

> **Not on PyPI or npm yet.** Until the first release, install from GitHub:
>
> ```bash
> pip install "throukit[anthropic] @ git+https://github.com/throuai/throukit#subdirectory=packages/throukit"
> ```

## Quick start

A tool is a plain function. An agent is a model, instructions and tools. The same agent runs two ways: `run()` returns a typed result, and `stream()` yields live events.

```python
import asyncio
from pydantic import BaseModel
from throukit import Agent, tool

@tool
def list_orders(ctx) -> list:
    """List the signed-in user's orders."""
    # ctx is your own per-run data. Throukit never reads it, so the tool
    # decides whose orders to fetch, not the framework.
    return db.orders_for(ctx["user_id"])

class Summary(BaseModel):
    open_orders: int
    headline: str

agent = Agent(
    name="orders",
    model="anthropic:claude-sonnet-5-5",   # or "openai:gpt-5.5", or your own ModelClient
    instructions="You help users manage their orders. Be concise.",
    tools=[list_orders],
)

async def main():
    # A typed, validated result.
    result = await agent.run("Summarize my orders.", context={"user_id": "u1"}, output_type=Summary)
    print(result.output, result.usage)

    # The same agent, streamed.
    async for event in agent.stream("Where is order 42?", context={"user_id": "u1"}):
        if event.type == "text_delta":
            print(event.text, end="")
        elif event.type == "step":
            print(f"\n[{event.status}] {event.title}")   # narration you can show users

asyncio.run(main())
```

A tool reports a failure the model should see with `ToolError("...")`. Any other exception is logged in full for you and reaches the model only as a short, safe message. A failed provider call raises `ModelError`, never a raw SDK exception.

[Getting started](docs/getting-started.md) walks through this example step by step.

## Why Throukit

- **The tool seam is the contract.** A tool source has two methods, `list_tools` and `execute`, and gets nothing but your per-run context. Local functions, MCP servers, sub-agents and OAuth connectors all sit behind it.
- **Auth stays yours.** Nothing in the core knows what a user or a credential is. Tools read whose account to act on from your own data.
- **One loop, two outputs.** `run()` and `stream()` share one loop, so typed results and live streams never drift apart.
- **Steps users can read.** Each tool call becomes a short, human title on the event stream, ready to show in your UI. OpenTelemetry tracing is there too, for you.
- **Provider-neutral, durable on your storage.** Anthropic and OpenAI ship in the box, `register_provider()` adds more, and checkpoints go to storage you own (in-memory and SQLite included). No hosted runtime.

## What's in the box

| | |
|---|---|
| **Tools** | `@tool` functions, MCP servers (`MCPToolSource`), sub-agents (`agent.as_tool()`, `DelegateToolSource`), agent copies that split work (`CloneToolSource`), and connectors, all behind one interface |
| **Agent files** | Define agents as Markdown with frontmatter, in the same shape as Claude Code subagents ([guide](docs/file-first.md)) |
| **Run control** | Cancel, message, budget and checkpoint a run from outside it (`RunControl`) |
| **Durability** | `Checkpointer` contract on your storage; powers human approval pauses ([guide](docs/durability.md)) |
| **Evals** | Suites with repeated runs, pass bars and a hard spend cap, plus scorers, model-graded judging and checks on the tools an agent called ([guide](docs/evals.md)) |
| **Connectors** | [`throukit-connectors`](packages/throukit-connectors): OAuth connection lifecycle with an official Composio implementation ([guide](docs/connectors.md)) |
| **TypeScript** | [`@throukit/client`](packages/client-ts): typed SSE consumer whose event types are generated from the Python models, so CI catches any drift |
| **Models** | Anthropic and OpenAI behind one `ModelClient`, `register_provider()` for more, and `ObservedModel` to meter, log or cap every call |
| **Testing** | `throukit.testing.ScriptedModel` scripts model turns, and record/replay `Cassette`s capture real ones, for fast tests with no network |

## Agent files

```markdown
---
name: email-triage
description: Triages the inbox and proposes what to do next.
model: anthropic:claude-sonnet-5-5
tools:
  - myapp.tools:search_email
output_schema: myapp.schemas:TriagePlan
---
You are an email triage assistant for {user_name}.
Work through the inbox with your tools, then deliver a triage plan.
```

```python
from throukit import load_agent

agent = load_agent("agents/triage.md")  # the same Agent the constructor builds
```

## Stream to a web app

```python
# FastAPI, or any ASGI framework
return StreamingResponse(sse_stream(agent.stream(prompt, context=ctx)), media_type="text/event-stream")
```

```ts
import { streamEvents } from "@throukit/client";

for await (const event of streamEvents("/agents/support/stream", { method: "POST", body })) {
  if (event.type === "text_delta") render(event.text);
  if (event.type === "step") showStep(event.title, event.status);
}
```

## Documentation

| Guide | Covers |
|---|---|
| [Getting started](docs/getting-started.md) | Install, a first agent, both run modes |
| [Core concepts](docs/concepts.md) | The tool seam, per-run context, multi-agent, workspaces, models, errors |
| [Agent files](docs/file-first.md) | Frontmatter reference and prompt substitution |
| [Connectors](docs/connectors.md) | The OAuth connector contract and Composio |
| [Durability & networks](docs/durability.md) | Checkpointers and the experimental graph |
| [Evals](docs/evals.md) | Suites, pass bars, spend caps, scorers, trajectory checks |
| [Design record](docs/design.md) | Why each piece is shaped the way it is |

Runnable scripts for each surface live in [`examples/`](examples).

## How it compares

| Framework | What it is | How Throukit differs |
|---|---|---|
| **LangGraph** | A graph runtime; state and checkpoints are the product | Agents-as-tools is the default; the graph is an experimental escape hatch |
| **Pydantic AI** | Type-safe, Python-first agents; the closest neighbour | Same core ideas, plus the connector seam, agent files and a TypeScript client |
| **Agno** | A batteries-included runtime with memory and RAG built in | Leaves memory, RAG and scheduling to your app |
| **Inngest AgentKit** | TypeScript-first agent networks tied to a durable engine | Borrows the router idea but stays storage-agnostic, with no platform to adopt |

Throukit is not a hosted platform, and it holds no opinion on auth, memory, RAG or scheduling.

## Project status

**Version `0.1.0`, before 1.0.** The API grows only when a real use needs it, and a minor release may still break it; [CHANGELOG.md](CHANGELOG.md) says when. Anything under `throukit.experimental` can change in any release.

Throukit makes no network calls of its own and collects no telemetry. OpenTelemetry spans go only to an exporter you set up.

## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the one-command dev setup and the checks every PR must pass. Report security issues privately, as [SECURITY.md](SECURITY.md) describes.

## License

[MIT](LICENSE). Copyright Throu, Inc.
