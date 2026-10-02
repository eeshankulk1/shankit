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
    instructions="...",         # static prose, or a function of the per-run context;
                                # either may be a list of parts, stable to volatile:
                                # Anthropic caches each part as its own block
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
    workspace=None,              # a Workspace, or a function of the per-run context
                                # returning one — see Workspaces below.
    spill_threshold_chars=25_000, # with a workspace, spill an oversized tool result
                                # to a file instead of truncating it.
    reasoning=None,             # True | "low".."max" | False | Reasoning(); provider-
                                # neutral "thinking". None sends nothing (the model's own
                                # default); False sends each model's lowest setting
                                # (models that can't turn thinking off think briefly:
                                # `anthropic.model_rules`).
    context_clear_threshold_tokens=None, # stub old tool results once a pass's prompt
                                # crosses this many tokens (long-run context management).
    context_keep_recent_results=3, # results kept intact when clearing context.
    timeout_s=None,             # wall-clock bound on the whole run, beside max_iterations.
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
result.messages    # every message the run added, prompt first — in-process only,
                   # not on the stream or in RunResult's serialization; pass it
                   # back as `history` to continue with full tool context
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

`ToolResult(end_run=True)` ends the run once that tool batch is processed,
instead of going back to the model: for a tool the model has nothing to add
to, like a question put to a human whose answer arrives later as a new
message. The run finishes with `done` as usual (a structured run gets no
validated `output`).

### Images and provider blocks in results

`ToolResult.content` may also be a list of blocks: `TextBlock`, `ImageBlock`
(base64, e.g. a screenshot), and `ProviderBlock` - a provider-specific block
carried opaquely (Anthropic's `browser_state`). The client of that provider
sends the block's `data` verbatim; every other client sends its `text`
fallback, so one result reads well on any provider. `content_text()` renders
any content as text (it's what trajectories and evals record).

```python
ToolResult(content=[
    TextBlock(text="Screenshot of the checkout page."),
    ImageBlock(media_type="image/jpeg", data=b64),
    ProviderBlock(provider="anthropic", data=browser_state, text="1 tab: Checkout"),
])
```

Screenshots add up fast. `Agent(keep_recent_images=3)` replaces all but the
newest three images with a short stub once tool results hold twice that many
(batched, so the prompt cache survives in between); like context clearing, a
clear drops reasoning blocks and runs the next pass with reasoning off.

### Images in a prompt

A prompt is text, or a list of `TextBlock` and `ImageBlock` in the order the
model should read them - a photo the user attached, then their question:

```python
await agent.run([
    TextBlock(text="What's on this receipt?"),
    ImageBlock(media_type="image/jpeg", data=b64),
])
```

Anthropic gets an image block, OpenAI an `image_url` part. A `history` message
may carry image blocks the same way. `keep_recent_images` counts only images
in tool results; what a prompt carries is the caller's to size.

### Native toolsets

Some providers train their models on a toolset they define and your code
executes - Anthropic's browser toolset (`browser_toolset_20260801`) is one
entry that stands for ~30 member tools (`navigate`, `left_click`, ...).
Describe the members as ordinary `ToolDef`s sharing a `Toolset`:

```python
BROWSER = Toolset(
    name="browser",
    native={"anthropic": {"type": "browser_toolset_20260801"}},
    ordered=True,
)
ToolDef(name="navigate", description="Open a URL.", input_schema=..., toolset=BROWSER)
```

`AnthropicModel` sends the native entry once in place of the members and
round-trips `toolset_name` on each call (`ToolUseBlock.toolset`) and its
result - the API rejects a result that drops it. Every other client sends
the members as plain tools, so the same source works on any provider (the
member schemas are the neutral mirror).

`ordered=True` makes a turn's calls to the toolset run one at a time, in the
order the model emitted them, and stop at the first failure: the rest come
back as `Toolset.not_executed` errors (Anthropic's exact text by default).
Other tools in the same turn still run concurrently.

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

A sub-agent's own steps forward live into the parent's stream as they happen —
the parent's consumer sees them as they occur, not just as one tool result at
the end. Each forwarded `StepEvent` gets its `agent` field set to the
sub-agent's name and its `id` prefixed with the parent step's id, so ids stay
unique when several sub-agents run.

### Delegation: one tool over a roster

`as_tool()` gives each sub-agent its own tool. `DelegateToolSource` instead
puts a **roster** of sub-agents behind a single `delegate` tool — the parent
sees one tool whose description lists what each agent is for, and picks by
name:

```python
from shankit import Agent, AgentDelegate, DelegateToolSource

triage = Agent(name="triage", model="anthropic:claude-sonnet-4-5", instructions="...")
billing = Agent(name="billing", model="anthropic:claude-sonnet-4-5", instructions="...")

parent = Agent(
    name="support",
    model="anthropic:claude-sonnet-4-5",
    tools=[DelegateToolSource([triage, billing], max_calls_per_delegate=2)],
)
```

Pass `Agent`s directly (auto-wrapped in `AgentDelegate`) or implement
`Delegate` for anything else behind the same tool (an agent with its own
post-processing, a remote agent). The source enforces per-delegate call
budgets, dedupes an identical task to the same delegate, sanitizes a
delegate's crash into a calm "couldn't complete" result, forwards its steps
live (same `agent`-tagged `StepEvent`s as `as_tool()`), and rolls its usage,
sources, and `truncated` flag into the parent run.

`DelegateToolSource.from_directory("agents/", agent_kwargs={"workspace": ...})`
builds the roster straight from a directory of `agents/*.md` files (see
[File-first definitions](file-first.md)); `agent_kwargs` passes runtime wiring
the file format can't express, like a shared workspace, through to every
agent it loads.

### Clones: the agent splits its work across copies of itself

A specialist roster pays for context isolation with lossy summaries and a
second prompt to maintain. `CloneToolSource` gives the running agent one
`spawn` tool instead: each part of the work runs as a **clone** — the same
`Agent` (model, instructions, tools, reasoning), with a fresh history that
starts from a brief the parent wrote — and the tool returns each clone's
report. Because a clone is the same agent, its first request shares the
parent's prompt prefix, so the provider's prompt cache serves it.

```python
from shankit import Agent, Budget, CloneToolSource

assistant = Agent(
    name="assistant",
    model="anthropic:claude-sonnet-4-5",
    instructions=SYSTEM,
    tools=[
        mail_tools,
        ask_user_tool,
        CloneToolSource(
            refuse=["ask_user"],            # one voice: only the parent asks
            budget=Budget(max_passes=15, timeout_s=300),
            max_per_batch=5,
        ),
    ],
)
```

What the source enforces as code:

- **Depth** — a clone can't spawn (`max_depth=1`), refused at call time.
- **Refusals** — `refuse` names tools a clone may not call (anything that
  talks to the user, say). They stay in its tool list — removing them would
  change the prompt prefix — and a call comes back as an error that points
  the clone at its report instead.
- **Caps** — `max_per_batch`, `max_per_run`; over a cap the call is refused
  whole so the model regroups.
- **Budgets** — each clone runs under a `Budget` (passes, wall clock, cost).
  One that reaches it gets a last pass with tools off and reports what it
  has (`partial`).
- **Attribution** — a clone's steps carry `worker=<clone id>` (separate from
  `agent`, which names a delegated sub-agent or app), and its lifecycle is a
  `worker` event in the parent's stream. Calls from one model response share
  a batch id.

Hooks shape a clone without breaking the cache: `prompt` (its first message
from the brief — a preamble, say), `child_context` (its per-run context),
`configure` (the agent it runs as — `agent.copy(timeout_s=None)`; a
different model loses the shared cache), `clone_properties` (extra fields per
clone).

**Where clones run** is a seam, `CloneHost`. The default `InlineCloneHost`
runs a batch concurrently inside the spawn call and rolls the clones' usage,
sources and artifacts into it. An application with a durable runner writes
its own host: start the clones there, wait as long as the parent can afford,
return `background` outcomes for the rest, and deliver their reports later
itself (`rolls_up = False` when it meters clone work separately).
`run_clone()` runs one clone to its outcome either way, including resuming
one from its saved transcript.

### Run control

`RunControl`, passed to `run()`, `stream()` or `resume()`, steers a run from
outside it — always at the loop's boundaries, never mid-call:

| | |
|---|---|
| `cancel()` | Ends the run before its next model pass, or before the tools of a pass run; `done.stopped == "cancelled"`, usage and transcript kept |
| `send(text)` | Delivered with the next tool results, or as one more turn if the model was finishing |
| `budget=Budget(...)` | Soft limits (passes, wall clock, priced cost): a last pass with tools off, then `stopped == "budget"` |
| `refuse=fn` | A call-time veto `(name, args) -> text or None`; the tool list is unchanged |
| `checkpoint=fn` | Called with the transcript at every boundary; `agent.resume(transcript)` continues it, closing any call left in flight with an "interrupted" result |
| `worker`, `depth`, `parent` | Who this run is among several; the spawn tool's depth guard reads `depth`; cancelling `parent` cancels this run |

`cancel()` and `send()` are thread-safe. `agent.copy(**changes)` makes a
variant (a longer timeout for background work) that shares everything it
doesn't change.

For the case agents-as-tools does _not_ cover — when **code, not the model**,
controls flow — reach for the experimental [network](durability.md#networks-experimental).

## Workspaces

An agent's `workspace` gives it a computer: a filesystem, and optionally a
shell. `Workspace` is a contract, not a vendor — a hosted microVM, a
container, or `LocalWorkspace` (a directory + subprocess) all implement the
same small surface (`exec` optional, `read_file`/`write_file` required,
`home`/`tmp`, and `resolve()` for `~`/relative paths). Name one per run the
same way a connector names *whose* credentials to use:

```python
from shankit import Agent, LocalWorkspace, WorkspaceTools

def sandbox_for(ctx):
    return LocalWorkspace(f"/var/sandboxes/{ctx['user_id']}")

agent = Agent(
    name="coder",
    model="anthropic:claude-sonnet-4-5",
    workspace=sandbox_for,
    tools=[WorkspaceTools(sandbox_for)],  # same spec, so both resolve the same workspace
)
```

`WorkspaceTools` ships `Bash`/`Read`/`Write`/`Edit` in the shapes coding
agents converge on, as plain JSON-schema tools any provider can call (no
Glob/Grep — `rg`/`ls` via `Bash` cover both). `Bash` drops itself
automatically for a workspace that can't `exec`. A non-zero exit is
information, not a tool failure; only a timeout is. `Read` of an image (PNG,
JPEG, GIF, WebP - found by its bytes, up to 3.75 MB) returns the image itself
as an `ImageBlock`, so an agent can look at a chart it rendered or a photo it
was given, the way coding agents read screenshots. `describe_workspace_step`
is a ready-made step-describer for these four tools (compose it with your own:
`describe_workspace_step(...) or my_describer(...)`), and
`WorkspaceTools.after_tool` is the hook for syncing files out, logging work,
or attaching artifacts.

With a workspace, a tool result over `spill_threshold_chars` (default 25K) is
spilled — written whole to `<workspace.tmp>/tool-output/`, with the model
getting its head, tail, and path instead of the full text. This is lossless
where `max_tool_result_chars` truncates. `context_clear_threshold_tokens`
does the same for the whole run: once a pass's prompt crosses the threshold,
older tool results are stubbed (full text saved to the workspace first,
falling back to the cap without one) and reasoning blocks are dropped,
keeping the newest `context_keep_recent_results` results intact.

> **Security note:** `LocalWorkspace` is for development and tests, not a
> security boundary — commands run as the current OS user with full access
> to the host. Give an agent that handles untrusted content, or acts for
> other people, a real sandbox (a hosted microVM or container) behind the
> same `Workspace` contract.

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

### Testing with a scripted model

`shankit.testing.ScriptedModel` is a provider-neutral fake `ModelClient` for
deterministic tests, with no network and no spend. It plays a script, one turn
per model call, and records every request the loop sends.

```python
# test_support.py (pytest + pytest-asyncio, asyncio_mode = "auto")
from shankit import Agent, tool
from shankit.testing import ScriptedModel, ToolCall, Turn


@tool
def order_status(order_id: int) -> str:
    """Look up an order's shipping status."""
    return "shipped"


async def test_support_agent_checks_the_order():
    model = ScriptedModel([
        Turn("Let me check.", tool_calls=[ToolCall("order_status", {"order_id": 7})]),
        Turn(
            "Order 7 has shipped.",
            expect=lambda request: request.messages[-1].content[0].content == "shipped",
        ),
    ])
    agent = Agent(name="support", model="any-id", model_client=model, tools=[order_status])

    result = await agent.run("Where is order 7?", output_type=str)

    assert result.output == "Order 7 has shipped."
    assert model.requests[0].tools[0].name == "order_status"
    model.assert_exhausted()
```

- **Turns.** A `Turn` composes any of `text` (streamed as word-sized deltas),
  `tool_calls`, `reasoning`, `output` (the value `run(output_type=...)`
  returns), `usage`, `stop_reason`, raw `blocks`, and `error`. Shorthands: a
  `str` is a text-only turn, an exception is raised, and a `ModelResponse` is
  returned verbatim.
- **Errors.** Script `model_error_for_status("anthropic", 429)` (or `529`) for
  exactly what the shipped clients raise on a rate limit (or an overload):
  `run()` raises it, and `stream()` ends with a retryable `error` event.
- **Requests.** `model.requests[i]` is a snapshot of the request `script[i]`
  answered: messages, tools, system, model id, tool choice, and params. A
  turn's `expect=` checks its request before the turn plays; an `assert`
  inside it, or a `False` return, fails the test with a summary of the request.
- **Loud failures.** A failed `expect`, or a call past the end of the script,
  raises `ScriptFailure`. It is a `BaseException`, so neither the loop's
  model-error handling nor a sub-agent's sanitizing can turn a broken script
  into a calm `error` event. `assert_exhausted()` catches the opposite case:
  turns that never ran.
- **Behind a spec.** For code that builds its own agents,
  `register_provider("anthropic", lambda: model)` routes every
  `"anthropic:..."` agent to the script. One model plays one script in call
  order, so give agents that run concurrently their own.

## The event stream

`stream()` (and `run(on_event=...)`) emit a closed set of typed events. Their
types are the source of truth the [TypeScript client](../packages/client-ts)
generates from.

| `event.type` | Payload |
|---|---|
| `text_delta` | `text` — an incremental chunk of the assistant's reply |
| `step` | `id`, `title`, `detail`, `phase`, `status` (`running`/`done`/`error`), `agent`, `worker` — user-facing narration; `agent` names the delegated sub-agent the step belongs to, `None` for the running agent itself; `worker` is the clone that took it, `None` for the top-level run |
| `worker` | `id`, `title`, `status` (`running`, then `done`/`partial`/`failed`/`stopped`/`background`), `batch` — a clone's lifecycle in the stream of the run that spawned it |
| `source` | `source` — a citation surfaced by a tool |
| `usage` | `usage` — token usage for one model pass or sub-agent (they sum to `done.usage`) |
| `done` | `text`, `output`, `usage`, `truncated`, `stopped` (`budget`/`cancelled` when a `RunControl` ended it early) — terminal success. `messages` also rides the event (every message the run added) but is in-process only — excluded from serialization, never on the SSE wire or in the generated TS types |
| `error` | `message`, `code`, `retryable` — terminal failure (human-safe message) |

A stream **always** terminates with exactly one of `done` or `error`, so an SSE
consumer can tell "finished" from "the connection just dropped." `code` is an
open string; codes emitted today are `model_error`, `max_iterations`,
`output_validation`, `timeout` (the run passed its `timeout_s`), `error`, and
`unexpected`.

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
- Exceeding `timeout_s` → `RunTimeoutError` (error code `timeout`), checked
  between passes, beside `max_iterations`.

## Where to go next

- **[File-first definitions](file-first.md)** — author agents as Markdown.
- **[Durability & networks](durability.md)** — checkpointers and code-controlled
  flow.
- **[Design decision record](design.md)** — the reasoning behind every primitive
  above.
</content>
