# Framework Design

> **Status:** v1 implemented (see §14 for implementation notes and small
> corrections made during the build). This is the living design spec for
> an open-source AI agent framework being harvested from throu.
>
> **How to read this doc.** It is a **decision record, not an implementation
> spec.** It pins the decisions we've committed to and the reasoning behind
> them, and it deliberately leaves implementation shape — exact type signatures,
> module layout, naming, library choices, file discovery mechanics — to the
> implementer. Where something is intentionally left open, it says so. If a
> detail isn't stated here, that's latitude, not an oversight: make the choice
> that best serves the principles in §1.
>
> **Working name:** *unresolved* — "AgentKit" is taken (Inngest and OpenAI both
> ship products by that name) and is retired even as a placeholder. Pick a real
> name before first release (§13).

---

## 1. Thesis

Take the **integration seam** — the boundary between an agent and the real,
authenticated external tools it acts on — as seriously as the agent loop itself,
**without imposing how you authenticate.** Every leading framework treats "where
do the tools come from" as the developer's problem and makes the agent loop the
star; the connection to real systems is an afterthought. We invert that.

Three commitments follow, and every downstream decision should be checked
against them:

1. **Minimal core, escape hatches over abstraction.** A small set of primitives
   with concrete default behavior. When power is needed, add an escape hatch —
   never a heavier abstraction imposed on everyone.
2. **Definitions are prose-first.** Behavior is authored as prose + a few
   declarative knobs (the Claude Code ergonomic), not builder-pattern code. Code
   appears only where code is genuinely required: tool/connector implementations.
3. **The framework defines contracts, not systems.** We own the *shape* of the
   seam (tool format, error handling, observability, per-run context) and never
   the *mechanism* (auth, credential storage, vendor choice).

**Lightweight but powerful** is the felt experience we're optimizing for. Prefer
borrowing the ecosystem's existing conventions over inventing proprietary ones;
a surface that feels native is the point.

### What this is NOT

- Not a general-purpose "better LangChain." On raw agent-core features we are at
  parity with Pydantic AI; we do not differentiate there and will not try.
- Not a platform. No hosted runtime or deploy product. Durability is a protocol
  you back with your own storage.
- Not opinionated about auth, memory, RAG, or scheduling. Those are the app's
  concern; the framework leaves room for them without shipping an opinion.

---

## 2. Competitive landscape (why this shape)

| Framework | Identity | What it means for us |
|---|---|---|
| **LangGraph** | Low-level graph runtime; state/checkpointing/HITL are the product | Durable execution + shared state is their moat and our #1 gap. Verbose by design — we avoid that. |
| **Agno** | Batteries-included multi-agent framework + runtime | Memory/knowledge/RAG in-box. We deliberately punt these to the app. |
| **Pydantic AI** | Type-safe, Python-first, "the FastAPI way" | **Closest neighbor.** Already ships our core feature list + durability + evals. Our core is *not* a differentiator vs them. Typed per-run dependency pattern is worth borrowing. |
| **Google ADK** | Code-first multi-agent wired to Gemini/Vertex | Built-in eval + cloud deploy, with platform gravity. |
| **Inngest AgentKit** | TS-first multi-agent *networks*, deterministic routing, welded to its own durable engine | Its router-based network model is our graph escape hatch, already mature — worth learning from over LangGraph's node/edge verbosity. Platform-coupled; owns the "AgentKit" name. |

**Honest conclusion:** our differentiation is *not* the primitives. It is
**orientation** — four things none of the leaders optimize for together:

1. Connectors as a first-class, vendor-neutral contract.
2. Observability as a *product* surface (user-facing step narration), not just
   dev tracing.
3. File-first (Claude Code–style) definitions as the headline authoring surface.
4. A paired Python core + TypeScript *consumer* SDK.

---

## 3. Core primitives

The whole core is a small set. Everything else is a satellite.

### 3.1 The tool seam

The single abstraction the agent loop knows about for tools exposes exactly two
capabilities: **list the tools it offers** and **execute one of them.** Both
receive the opaque per-run context (§3.3) and nothing more. Tool definitions are
in **Anthropic tool-use shape** at the boundary — one format everywhere; model
clients re-convert internally.

That two-capability contract is the whole seam. It is deliberately too small to
over-abstract. The core ships a handful of implementations of it — at minimum:
local Python functions, any MCP server, and a way to combine several sources
into one. *Which* implementations and their exact interface are the
implementer's call; the fixed decisions are (a) exactly two capabilities, (b)
Anthropic tool shape at the boundary, (c) MCP and local functions both
first-class in core.

### 3.2 The agent

One agent primitive, runnable two ways:

- **Structured** — returns a typed, validated deliverable (this is throu's
  `run_agent_loop` / skill path).
- **Streaming** — returns a live event stream: text deltas, steps, sources, and
  usage (this is throu's `run_chat_loop` / chat path).

Both are the **same agent over the same underlying tool-use loop.** They are
**not merged** (throu's standing rule) and they split on **output shape**
(typed deliverable vs. streamed conversation), *not* on sync-vs-async. An agent
is configured by: a model, instructions (static prose or a function of the run
context), one or more tool sources, and an optional step-describer for
observability.

Whether structured output is requested is the **caller's** choice at call time.
There is no "chat vs. skill" mode baked into an agent — an agent that has a
declared default schema *can* be run for structured output; one without can only
stream. (This is why the file format needs no `mode` concept — see §5.)

### 3.3 Per-run context (dependencies)

A typed, opaque per-run container is threaded to tools and to dynamic
instructions. It carries whatever the caller needs at runtime (e.g. the acting
user's id, timezone). **The framework never reads inside it.** This is how a
tool source resolves *whose* credentials to use without the loop knowing what a
"user" is.

Hard guardrail against over-abstraction:
- It is the caller's plain data (or nothing). No base class to inherit, no
  lifecycle, no resolution magic, no framework-owned context object with hooks.
- **Litmus:** if you deleted it, the only thing that should break is passing
  per-run data to tools — nothing about the agent's core behavior. If more
  breaks, we leaked.

### 3.4 Multi-agent

- **Default — agents as tools.** An agent can be adapted into a tool source, so
  handing sub-agents to a parent agent *is* orchestration. This absorbs throu's
  entire `consult_<domain>` machinery (routing, dedup, sanitized sub-agent
  failures, token accounting) into one reusable place.
- **Escape hatch — a graph/network.** For deterministic routing, cycles, and
  shared state. Marked **experimental** in v1 (§9).

---

## 4. Connector philosophy

**The framework defines the integration *contract*, not the integration
*system*.** Three buckets, strictly enforced:

**Framework OWNS (imposes):** the tool seam plus the small contracts that make it
work in a loop — tool shape, a uniform error contract (how a tool failure
returns so the loop handles it consistently; this promotes throu's hand-rolled
sanitized-failure path to the standard), the observability hook, and the opaque
per-run context. **None of these mention auth.**

**Framework NEVER imposes:** how you authenticate, where credentials live,
whether OAuth exists at all, which vendor.

**Framework OFFERS (opt-in, a separate package):** a connector layer that wraps
a tool source and owns the connection lifecycle (initiate / check-status /
adopt-existing) for sources that need OAuth, plus one official connector
implementation (Composio, which throu migrates onto). Local-function and MCP
users never touch this layer.

**Rationale — split by change-rate.** throu's current `Provider` conflates the
runtime seam (list/execute — all the loop touches) with a vendor-shaped
connection lifecycle (consumed only by connection UI). Splitting them keeps the
hot seam tiny and keeps OAuth methods out of the face of anyone not doing OAuth.

**Litmus for the whole design:** *could someone swap Composio for a completely
different system without the framework noticing?* If yes, we didn't over-impose.
If the framework must change, we leaked.

---

## 5. Signature ergonomic: file-first definitions

The headline way to define agents and skills, borrowing the Claude Code mental
model the target audience already knows. The governing split:

> **Behavior** (prose + a few knobs) → a declarative file.
> **Capability** (tools/connectors) → Python code.
> The file references the code by name.

This works *because* our tools are already code the file can name — exactly how
Claude Code's markdown references harness-provided tools.

**Committed decisions:**

- **Frontmatter mirrors Claude Code wherever the concepts overlap** (identity,
  description, model, the set of tools available), so an engineer who has
  written a CC subagent authors one of these with no new concepts. We diverge
  only to *add*, never to rename.
- **The one addition is a default output schema.** Its presence lets the agent
  be run for structured output; its absence means the agent can only stream. It
  is *not* a mode flag — the caller still chooses how to run (§3.2). So there is
  no "chat vs. skill" file type; a file is just an agent.
- **The body is a prompt with fill-in-the-blank substitution only.** Runtime
  values (from the per-run context or config) can be interpolated. Logic —
  conditionals, loops — is deliberately **not** supported: the moment a prompt
  needs an `if`, that is the signal to define the agent in Python instead, where
  instructions can be a function. This keeps files declarative instead of
  quietly becoming code.
- **References use standard Python, not invented syntax.** Anything importable
  (e.g. an output schema) is named by its ordinary Python object path and
  resolved the way Python itself resolves it — no custom loader, no proprietary
  path format. The ecosystem's existing `module:attribute` convention (as used
  by uvicorn, entry points, etc.) is the reference style.
- **Explicit over magic for live objects.** A connector is a live, configured
  object that can't be reached by import, so it is referenced by a short name
  that the developer **explicitly registers**. No auto-scanning, no
  registration-by-bare-name guessing. The registry is load-bearing only for
  things that genuinely can't be imported.
- **The Python constructor is the escape hatch.** A file produces the *same*
  agent object the constructor would; the file is sugar. Rich typing (typed
  output models, typed per-run context) lives in the Python path — files cover
  the common 80%, Python the typed 20%. Nobody is forced onto either.

**Left to the implementer:** the exact frontmatter key names and how they
serialize, file discovery mechanics (directories, precedence, uniqueness), and
the substitution/templating implementation. Honor the decisions above; choose
the rest.

---

## 6. Observability

First-class, and *user-facing* — not just dev tracing.

- A pluggable **step-describer** translates a raw tool call into a user-language
  step (a short past-tense title, optional detail, a phase). throu's
  `describe_step` becomes an instance of this contract — same behavior, now
  pluggable.
- The streaming run mode emits a typed **event stream** (text deltas, steps,
  sources, artifacts, usage, done). This is the single source of truth the TS
  consumer SDK renders and that throu's SSE maps onto directly.
- **Artifacts** are the structured-content half of that stream: a tool result
  may carry `Artifact {type, data}` payloads that the loop emits as
  `ArtifactEvent`s, accumulates on `RunResult.artifacts`, and passes through
  the `as_tool` seam intact. The framework owns the *carrying* (mechanism);
  what an artifact means — an email card, a calendar event — and what the app
  does with it (persist, render) is app policy, same stance as the per-run
  context (§3.3). `data` is never inspected by the framework.
- OpenTelemetry tracing is available and orthogonal to the user-facing steps.

---

## 7. Cross-cutting features

### 7.1 Durability (built in v1)

Our #1 competitive gap *and* #1 use case (scheduled / long-running agents).
Committed: a **checkpointer contract** with per-thread identity, backed by
**your** storage (no platform gravity), shipping with at least an in-memory and
a durable-store implementation. The checkpointed state is the *same* state the
graph shares across steps (§9) — one concept, not two — and the natural
checkpoint boundary is a router decision. Human-in-the-loop interrupts are in v1
scope (they ride on this + the graph). Deferred to fast-follow: full
resume-after-restart semantics.

**Constraint on the core loop:** it must not hardcode assumptions that make a
checkpointer impossible to add later. Cheap now, expensive to retrofit.

### 7.2 Evaluation (built in v1)

A lightweight harness for datasets, scorers, and trajectory assertions. Low API
risk, high adoption value.

### 7.3 Models

A provider-neutral model-client contract (throu already has this seam).
Anthropic and OpenAI at launch; broader provider coverage is easy to add later.

---

## 8. Language & TypeScript SDK

- **Python core.** Agents author and run in Python.
- **TypeScript consumer SDK.** Typed consumption of the event stream over
  SSE/HTTP, with schemas generated from the Python source of truth so the two
  can't drift. It consumes agent runs; it does **not** run agents. This is a
  strict subset of a future native-TS or author-in-TS direction, so nothing
  built here is throwaway if we later grow the TS side.

---

## 9. Graph / network escape hatch (experimental)

The one genuinely shape-uncertain API. Build it in v1, ship it **flagged
experimental**, and freeze the contract only once throu exercises it (see the
validation path below). It exists for the case agents-as-tools does *not* cover:
when **code, not the model, controls flow** — deterministic routing, cycles,
state accumulated across steps, and pauses for human approval. Reaching for it
merely to chain two agents is a smell; keeping that line bright is what stops it
becoming a heavyweight graph DSL.

**Committed decisions:**

- **Router-driven loop over shared state, not a declared node/edge graph.** A
  loop repeatedly calls a *router* that reads the shared state and returns the
  next step to run (or "stop"). Much lighter than declaring nodes and edges up
  front, and honest about being imperative control flow. (Learning from
  Inngest's network model over LangGraph's node/edge verbosity.)
- **One state concept, shared with durability.** The state the router reads and
  writes across steps **is** what the checkpointer (§7.1) persists — not a second
  state system. A router decision boundary is the natural checkpoint / resume /
  interrupt point, which is what makes human-in-the-loop fall out of the design
  rather than being bolted on.
- **A step is anything callable that touches state** — an agent (the common
  case), a plain function, or a nested network.
- **A network is itself runnable like an agent and exposable as a tool**, so
  deterministic sub-flows and normal agent conversations compose in both
  directions.
- **Code routing is primary; LLM routing is merely possible.** The router is
  just a function; call a model inside it if you want. We do not build a special
  "LLM router" abstraction — the whole reason to drop here is deterministic
  control.

**In experimental v1 scope:** router loop, shared state, cycles, **and
human-in-the-loop interrupts** (approval pauses). **Fast-follow:** full
resume-after-restart semantics.

**Validation path.** throu's orchestrator is agents-as-tools and has no reason
to touch the graph, so to avoid shipping it vacuum-designed, it is dogfooded via
a real throu need: **a skill that takes a consequential action** (send an email,
create a PR, transition a ticket) routed as `gather → draft → pause for user
approval → act`. That is exactly deterministic routing + shared state +
human-in-the-loop, and it gives throu safe write-actions as a byproduct. This
"act-with-approval" slice is what freezes the graph contract (§12).

---

## 10. Framework vs. throu boundary

| throu today | becomes | lives in |
|---|---|---|
| `Provider` (Composio) | the official connector → a tool source | connector package |
| `DomainAgent` (e.g. `GmailAgent`) | an agent bound to a tool source + a step-describer | framework |
| `describe_step` override | a step-describer instance | framework contract, throu-authored |
| `Skill` | a structured agent run **+ throu's** schedule id + post-processor to `tracked_items` | split |
| `RootAgent` / `consult_<domain>` | a parent agent whose tools are other agents | framework |
| lookup/write intent split | throu policy: picks model + output shape per call | throu only |
| cron, `tracked_items`, memory, preferences, quotas | unchanged | throu only |

**Stays in throu because it's a product opinion, not a reusable primitive:**
scheduling, persistence, the lookup/write model policy, memory/preferences.

---

## 11. Scope by commitment confidence

The binding constraint is **API-commitment risk** (un-shipping a public API is
expensive), *not* build effort.

| Area | Call |
|---|---|
| Agent (two run modes), tool seam, per-run context, models, event stream | **v1, committed** |
| Core tool sources (local functions, MCP, aggregation) | **v1, committed** |
| Official connector + connector layer | **v1, committed** |
| Step-describer observability | **v1, committed** |
| File-first definitions (headline authoring surface) | **v1, committed** |
| Durability contract | **v1, build it** (#1 gap + #1 use case) |
| Evaluation harness | **v1, build it** |
| Graph/network + shared state + HITL | **v1 API, experimental** — freeze later |
| TS consumer SDK | **v1, build it** (throu needs it) |
| Native-TS runtime, hosted platform, memory/RAG batteries | **out of scope** |

---

## 12. Rollout plan

Harvest the framework from throu; do not design it in a vacuum and port throu
last. **The framework API should never be more than one real throu usage ahead
of itself.**

1. Finalize this design doc.
2. Scaffold the framework repo, wired into throu as a local/editable dependency
   from the first line of the refactor (not a published package).
3. Refactor throu **one vertical slice first** (a single connector, end-to-end,
   through both a structured run and a streaming chat) before porting the rest.
4. Port the remaining domains once the slice is green.
5. Dogfood the experimental graph via the **act-with-approval** slice (§9): one
   throu skill that takes a consequential action behind a human-approval pause.
   This is what exercises — and ultimately freezes — the graph contract.
6. Cut a versioned release / freeze the public API only after throu fully runs
   on the framework. throu's green test suite is the acceptance criteria.

---

## 13. Open threads

1. **Evaluation harness detail** — dataset/scorer/trajectory-assertion surface.
   Lighter; can be settled during implementation.
2. **TypeScript consumer SDK detail** — event-type generation, transport, the
   published API. Can come after the Python core stabilizes.
3. **Name** — "AgentKit" retired; choose before first release.

*Resolved and folded in:* core primitives (§3), connector contract (§4),
file-first format (§5), graph/network escape hatch + validation path (§9),
durability contract shape (§7.1).

---

## 14. v1 implementation notes (decision record addendum)

The v1 build honors every committed decision above. Where the doc left
latitude, these are the load-bearing choices — plus a few small corrections
where the original text had a gap (flagged ⚠):

1. **⚠ The event stream gained an `error` terminal event.** §6 listed
   "text deltas, steps, sources, usage, done" — but a stream with no failure
   signal can't be consumed safely over SSE (the transport just ends). A
   stream now terminates with exactly one of `done` | `error`.
2. **⚠ `str` counts as an output type.** §3.2/§5 read as if absence of a
   schema forbids `run()` entirely. v1 allows `run(output_type=str)`: the
   final assistant text as a (trivially) typed deliverable. Without this,
   agents-as-tools (§3.4) could not obtain a sub-agent's text answer through
   the structured path, and evals couldn't target schema-less agents. The
   spirit is kept: no schema and no `output_type=` ⇒ `run()` refuses and
   points to `stream()`.
3. **Model specs require a provider prefix** (`anthropic:...`, `openai:...`);
   there is no silent default vendor — neutrality (§1.3) over convenience.
   `register_provider()` extends the prefix set; passing a `ModelClient`
   bypasses it entirely.
4. **File refs disambiguate on the colon**: `module:attribute` ⇒ Python
   import; bare `name` ⇒ explicit registry (§5). `output_schema` must be an
   import ref (a type is always importable).
5. **The experimental flag is structural**: the graph lives at
   `shankit.experimental.graph`; stabilizing it *is* moving it out of that
   namespace.
6. **Uniform error contract**: tools raise `ToolError("model-visible text")`
   for controlled failures; any other exception is sanitized to a generic
   message (full traceback logged) — the loop applies this in one choke
   point regardless of tool source. Sub-agent failures surface as the same
   sanitized shape.
7. **Checkpoint state is JSON** (enforced by the shipped stores on save), and
   the network checkpoints after every completed step, at interrupts, and at
   done. Durable stores shipped: in-memory and SQLite. `resume()` handles
   `interrupted` threads (HITL); `recover()` re-enters `running` threads
   after a crash or step failure, using a write-ahead `in_flight` marker to
   distinguish "died between steps" (safe — the router re-derives the next
   step from state) from "died mid-step" (side effects ambiguous — refused
   unless the caller asserts idempotency with `retry_in_flight=True`). This
   is the minimal principled slice of §7.1's resume-after-restart
   fast-follow; richer semantics (per-step idempotency keys, retry policies)
   stay open until real usage demands them.
8. **TS types are generated, not mirrored**: `scripts/generate_ts_events.py`
   emits `@shankit/client`'s event types from the pydantic models; CI fails
   on drift (§8).
9. **Naming (§13) remains open** — everything ships under the working name
   `shankit` / `@shankit/client`; renaming before first release is a
   find-replace plus package metadata.

10. **⚠ `text` is the transcript; `output` is the answer.** The original
    build kept only the final iteration's text, so an agent that speaks
    before its tool calls lost those sentences from the result — the
    streamed transcript and the persisted text disagreed, violating §6's
    "single source of truth," and the first consumer coupled to UsageEvent
    ordering to reassemble it. `text` (on `DoneEvent` and `RunResult`) is
    now every per-pass text joined with blank lines. The *deliverable*
    deliberately did not follow: for `output_type=str` runs, `output` (and
    therefore `as_tool()` text results, graph `agent_step` state, and what
    the eval scorers score) remains the final pass — the answer, not the
    narration; joining the transcript into the deliverable would leak
    "Let me check that." into drafts, eval haystacks, and sub-agent tool
    results. Corollary: usage a tool reports through the seam (a
    sub-agent's spend) now also emits a `UsageEvent`, giving the stream
    the invariant `sum(UsageEvents) == DoneEvent.usage`. A sticky
    `truncated` flag on `DoneEvent`/`RunResult` surfaces max-tokens
    cutoffs (any pass, including a sub-agent's via
    `ToolResult.truncated`) that were previously only logged.
11. **⚠ The uniform error contract (§4) now covers models, not just
    tools.** The v1 build sanitized tool failures but let model-provider
    exceptions cross the "provider-neutral" boundary raw, so retry/backoff
    meant importing provider SDK exception types. Failed model calls now
    surface as `ModelError` (`provider` / `status` / `retryable`, original
    exception chained): shipped clients map their own SDK's exceptions —
    the client knows which of its failures are transient — and the loop
    wraps anything a custom client leaks. `ErrorEvent` carries `code` (an
    open string, so new codes are additive) and `retryable`. Messages stay
    human-safe; detail lives on `__cause__`.
12. **Connector conveniences stay escape-hatch-shaped.** The official
    Composio connector gained an opt-in `tools_cache_ttl` (the catalog
    doesn't depend on per-run context; a warmup hook was rejected because
    warming *is* calling `list_tools` once) and a `transform_result`
    subclass point for slimming vendor payloads. `ConnectionStatus` gained
    a vendor-neutral `account_id`. Nothing new was added to the base
    `Connector` contract's method set.
13. **⚠ Toolkit versioning is the consumer's explicit choice.** The first
    real consumer surfaced that `ComposioConnector.execute` couldn't run
    tools at all against a client that pins no `toolkit_versions` (the SDK
    default): newer composio SDKs resolve the version to `"latest"` and
    raise `ToolVersionRequiredError` — invisible to the fake-SDK tests,
    which didn't model the check. The connector gained an opt-in
    `skip_version_check: bool = False`, forwarded as
    `dangerously_skip_version_check`. Default `False` on purpose: silently
    following `"latest"` means a breaking toolkit release changes behavior
    with no code change, so the consumer must choose — pin versions on the
    client, or opt in. The kwarg is only sent when opted in, because the
    declared floor (`composio>=0.8.0`) predates it and an unconditional
    forward would break every default-path user on an older SDK.
14. **Tool-result size is a loop safety bound.** A throu production incident
    (July 2026): a lookup sub-agent deep-fetched nine un-slimmed HTML emails
    and the run died mid-turn at 244,716 tokens > the 200K context window —
    a non-retryable provider 400 the end user saw as "Gmail is unavailable".
    The overflow happened in the loop's own `messages` list, so the guard
    lives with the loop's other runaway bounds (`max_iterations`,
    `max_tokens`), not in any one tool source: `Agent(max_tool_result_chars=)`
    caps every source uniformly — local functions, MCP, connectors,
    sub-agents, and `ToolError` text — at the `_execute_tool` choke point,
    appends a marker, and marks the result `truncated` (composing with the
    existing sticky flag rather than adding new surface). Default `None`:
    a silent default cap could corrupt structured pipelines that legitimately
    move large payloads, so the consumer opts in. Deliberately per-result,
    not per-turn: bounding the *sum* would need request-size awareness in
    the loop (a token estimator, provider-specific limits) — over-engineering
    until a consumer needs it. Semantic slimming (which fields matter, what a
    "preview" is) stays the consumer's job via `transform_result`; the cap is
    only the backstop that turns "the whole turn dies" into "one result is
    cut".
