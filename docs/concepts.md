# Core concepts

Everything in shankit is a small set of primitives; the rest is satellites. This
page is the mental model. Read [Getting started](getting-started.md) first if you
haven't run an agent yet.

## The agent

One `Agent` primitive, runnable two ways over the **same** underlying tool-use
loop:

| Method | Returns | Use for |
|---|---|---|
| `await agent.run(prompt, output_type=T)` | a `RunResult[T]` — a typed, validated deliverable | skills, jobs, evals, sub-agents |
| `async for e in agent.stream(prompt)` | a typed event stream | chat UIs, anything watched live |

The two modes split on **output shape** (typed deliverable vs. streamed
conversation), not on sync-vs-async — both are async. Whether structured output
is requested is the **caller's** choice at call time. An agent given a default
`output_type` _can_ be run structured; one without can only stream (or be run
with an explicit `output_type=`). There is no mode baked into the agent.

An `Agent` is configured with:

```python
Agent(
    name="...",                 # identity; also the default tool name via as_tool()
    model="anthropic:...",      # "provider:model_id", or a bare id with model_client=
    instructions="...",         # static prose, or a function of the per-run context
    tools=[...],                # any mix of @tool fns, ToolSources, and sub.as_tool()
    output_type=MyModel,        # optional DEFAULT schema (not a mode flag)
    describe_step=...,          # optional observability hook
    max_iterations=20,          # loop safety bound
    max_tokens=4096,
    max_tool_result_chars=None, # loop safety bound: cap any single tool result's
                                # content (marker appended, result marked truncated).
                                # None disables. Set it when tools can return
                                # unbounded payloads (files over MCP, vendor APIs) —
                                # one oversized result can otherwise exceed the
                                # model's context window and kill the run.
)
```

### `RunResult`

```python
result.output      # the ANSWER: validated schema instance, or final text for output_type=str
result.text        # the TRANSCRIPT: every assistant text pass, joined with blank lines
result.usage       # Usage(input_tokens, output_tokens, ...), including sub-agents
result.trajectory  # list[ToolCallRecord] — every tool call, for evals
result.sources     # list[Source] surfaced during the run
result.truncated   # True if any pass (or sub-agent) stopped at the token limit,
                   # or any tool result was capped by max_tool_result_chars
```

`truncated` is deliberately sticky — a cut-off intermediate pass can corrupt a
run as much as a cut-off answer. Caveat for orchestrators: if you run
sub-agents with deliberately tight `max_tokens` budgets (cheap fact-finder
patterns), a sub-agent brushing its budget flips the *parent* run's flag too.
Treat the flag as "something, somewhere, was cut" — not as "the final answer
is broken" — and decide what to surface to users accordingly.

`output` is the deliverable to score and act on; `text` is what you display and
persist. They differ deliberately: interim narration belongs in the transcript,
never in the answer.

## The tool seam

The single abstraction the agent loop knows about for tools exposes exactly
**two capabilities**:

```python
class ToolSource:
    async def list_tools(self, context) -> Sequence[ToolDef]: ...
    async def execute(self, name, arguments, context) -> ToolResult: ...
```

Both receive the opaque per-run context and nothing more. Tool definitions are
in **Anthropic tool-use shape** at the boundary — one format everywhere; model
clients re-convert internally. That two-method contract is deliberately too
small to over-abstract.

Everything is a tool source behind this seam:

| Source | How you get it |
|---|---|
| **Local functions** | `@tool` on a function, or pass a plain callable |
| **MCP servers** | `MCPToolSource(...)` — any MCP server, first-class in core |
| **Aggregation** | `CompositeToolSource([...])` — combine several into one |
| **Sub-agents** | `sub_agent.as_tool()` — see [multi-agent](#multi-agent) |
| **Connectors** | the opt-in [`shankit-connectors`](../packages/shankit-connectors) layer for OAuth sources |

When you pass a mixed `tools=[...]` list to `Agent`, shankit folds it into a
single seam implementation for you.

### Writing a tool

```python
from shankit import tool, ToolError, ToolResult
from shankit.events import Source

@tool
def search_docs(ctx, query: str) -> str:      # `ctx` first param is optional
    """Search the internal docs."""           # docstring = what the model sees
    if not query:
        raise ToolError("Provide a search query.")   # controlled, model-visible failure
    return db.search(query)

@tool
def cited_search(ctx, query: str) -> ToolResult:      # return ToolResult for richness
    return ToolResult(
        content="Deploys run at 2pm UTC daily.",
        sources=[Source(title="Runbook", url="https://internal/runbook")],
    )
```

A tool may return a plain `str` (the common case) or a `ToolResult` carrying
`content`, `sources`, `usage`, and a `truncated` flag. The first parameter is the
per-run context by convention; tools that don't need it can omit it.

## The per-run context

A typed, **opaque** container threaded to tools and to dynamic instructions. It
carries whatever the caller needs at runtime — the acting user's id, a timezone,
a tenant. **The framework never reads inside it.** This is how a tool resolves
_whose_ credentials to use without the loop knowing what a "user" is.

```python
await agent.run("...", context={"user_id": "u1", "tz": "UTC"}, output_type=T)
```

The guardrail: it's your plain data (or nothing). No base class, no lifecycle, no
framework-owned context object with hooks. If you deleted it, the only thing that
should break is passing per-run data to tools — nothing about the agent's core
behavior.

## Multi-agent

The default multi-agent pattern is **agents as tools**: adapt an agent into a
tool source and hand it to a parent. The parent's model decides when to consult
each specialist.

```python
oncall = Agent(
    name="oncall",
    model="anthropic:claude-sonnet-4-5",
    instructions="Coordinate specialists to answer on-call questions.",
    tools=[docs_agent.as_tool(), metrics_agent.as_tool()],
)
```

`as_tool()` folds the whole orchestration concern into the ordinary seam: routing
is the parent model's job, sub-agent failures are sanitized, sources propagate to
the parent's stream, and sub-agent token usage folds into the parent run's
accounting. Pass `output="structured"` to expose a sub-agent's schema instead of
its text.

> Pass sub-agents **as tools explicitly** (`sub.as_tool()`), never the `Agent`
> object itself — shankit raises a `TypeError` to keep the boundary obvious.

For the case agents-as-tools does _not_ cover — when **code, not the model**,
controls flow — reach for the experimental [network](durability.md#networks-experimental).

## Models

A provider-neutral `ModelClient` contract sits under every agent. Anthropic and
OpenAI ship at launch.

```python
model="anthropic:claude-sonnet-4-5"   # prefix selects the provider
model="openai:gpt-4.1"
```

- The **provider prefix is required** — neutrality over convenience; there is no
  silent default vendor.
- `register_provider("myvendor", factory)` extends the prefix set.
- Passing `model_client=` to `Agent` bypasses the registry entirely — bring your
  own client, and `model` becomes a bare model id.

## The event stream

`stream()` (and `run(on_event=...)`) emit a closed set of typed events. Their
types are the source of truth the [TypeScript client](../packages/client-ts)
generates from.

| `event.type` | Payload |
|---|---|
| `text_delta` | `text` — an incremental chunk of the assistant's reply |
| `step` | `id`, `title`, `detail`, `phase`, `status` (`running`/`done`/`error`) — user-facing narration |
| `source` | `source` — a citation surfaced by a tool |
| `usage` | `usage` — token usage for one model pass or sub-agent (they sum to `done.usage`) |
| `done` | `text`, `output`, `usage`, `truncated` — terminal success |
| `error` | `message`, `code`, `retryable` — terminal failure (human-safe message) |

A stream **always** terminates with exactly one of `done` or `error`, so an SSE
consumer can tell "finished" from "the connection just dropped."

## Observability

Observability is first-class and **user-facing**, not just dev tracing:

- A pluggable **step-describer** — `(tool_name, arguments, context) -> StepInfo`
  — turns each tool call into a short past-tense title, optional detail, and a
  phase. Return `None` to hide a call. A generic describer ships by default; pass
  your own via `describe_step=`.
- The resulting **step events** ride the stream and are exactly what a chat UI
  renders as "what the agent is doing."
- **OpenTelemetry** spans (`pip install shankit[otel]`) are available and
  orthogonal — developer tracing that never mixes with the user-facing steps.

## The uniform error contract

One choke point applies the same rules to every tool source and model:

- `ToolError("...")` → the model sees that exact message and can recover.
- Any other tool exception → logged in full, sanitized to a generic message.
- Sub-agent failures → surface as the same sanitized shape.
- Model provider failures → `ModelError(provider, status, retryable)`, never a
  raw SDK exception across the neutral boundary; the shipped clients classify
  their own transient failures.

## Where to go next

- **[File-first definitions](file-first.md)** — author agents as Markdown.
- **[Durability & networks](durability.md)** — checkpointers and code-controlled
  flow.
- **[Design decision record](design.md)** — the reasoning behind every primitive
  above.
</content>
