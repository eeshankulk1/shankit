# Deferred improvements

Findings from the July 2026 analysis pass (throu PR agentkit#29, framework PR #3)
that were deliberately NOT implemented because they are design decisions - new
event types, new exception contracts, or new API surface - rather than bug
fixes. Each one changes what consumers can depend on, so they deserve a
deliberate call instead of riding a hardening PR. Listed roughly by value.

The guiding constraint stays the same as the framework's founding rule: never
more than one real usage ahead, and no abstraction without a consumer that
needs it today. throu is the consumer motivating every item below.

---

## 1. Iteration-boundary semantics for streamed text

**Problem.** `DoneEvent.text` (and `RunResult.text`) carry only the LAST
model pass's text - `agent.py` overwrites `final_text` each iteration. An
agent prompted to emit interim acknowledgments before its tool calls (a
pattern throu deliberately uses in the orchestrator) loses those sentences
from the final result. throu works around it by reassembling text from
`TextDeltaEvent`s and treating each `UsageEvent` as a pass boundary
(`orchestrator.py`, comment at ~line 370) - coupling to an accident of event
ordering.

**Options.**
- (a) Accumulate: `DoneEvent.text` = non-empty per-iteration texts joined
  with `"\n\n"`. Simplest; changes the meaning of `.text` for every consumer.
- (b) New `IterationEvent(index)` emitted at each pass boundary, keeping
  `DoneEvent.text` as-is but giving stream consumers a real boundary signal.
- (c) Both: accumulated `text` + `texts: list[str]` per iteration on
  `DoneEvent`.

**Notes.** Any new event type must be regenerated into the TS client
(`scripts/generate_ts_events.py`; CI enforces sync). Option (a) or (c) lets
throu delete its reassembly entirely - the streamed transcript and the
persisted text become the same string by construction.

## 2. Typed model-error contract

**Problem.** The "provider-neutral" boundary leaks raw provider exceptions:
`Agent.run()` propagates `anthropic.RateLimitError` / `openai.APIStatusError`
as-is, so callers wanting retry/backoff must import provider SDK exception
types. `Agent.stream()` sanitizes into `ErrorEvent` but keeps only a message
string - consumers cannot distinguish "rate limited, retry shortly" from "bug,
don't retry." throu re-raises `RuntimeError(event.message)` to preserve its
route contract (`orchestrator.py` W3), losing everything else.

**Sketch.**
- `ModelError(ShankitError)` with `provider`, `status: int | None`,
  `retryable: bool`. Each `ModelClient` maps its own SDK's exceptions (it
  knows them); the loop wraps as a fallback for custom clients.
- `ErrorEvent` gains `code: str` (e.g. `model_error`, `max_iterations`,
  `output_validation`, `unexpected`) and `retryable: bool = False`.
- Optional follow-on: a per-request retry policy on `ModelRequest` for
  transient mid-loop failures (today a 529 on iteration 5 of a structured
  run loses the whole run; only the SDKs' built-in request retries apply).

**Notes.** Keep `ErrorEvent.message` human-safe (no request ids/keys) - the
message is what consumers show users. TS regen required.

## 3. ComposioConnector: caching + result hooks (delete throu's override)

**Problem.** throu's `ThrouComposioConnector` overrides essentially the whole
seam of the official connector (`list_tools`, `execute`, `initiate`,
`check_status`) - it inherits shape, not behavior. PR #3 fixed the worst part
(slug-based fetch instead of whole-toolkit). What remains, each a small API
decision:

- **`tools_cache_ttl: float | None` on the connector.** throu measured
  ~200-1000ms/turn for uncached catalog fetches and caches with a 10-min TTL.
  An opt-in TTL belongs on the official connector so every consumer gets it.
- **Schema-cache-aware execute.** The Composio SDK's `tools.execute()`
  re-fetches the tool schema via `GET /v3/tools/<slug>` on EVERY call
  (~150-400ms), even when a bulk fetch already cached it in
  `Tools._tool_schemas`. throu monkey-patches
  `Tools.get_raw_composio_tool_by_slug` on its client
  (`integrations/composio_client.py`). Decide whether the framework should
  (a) apply the same instance-scoped patch behind an opt-in flag, (b) keep a
  connector-level slug->schema cache populated by `list_tools` and route
  execute through it, or (c) leave it to consumers and document the patch.
  Re-verify against the current SDK before building - an SDK release may fix
  the redundant fetch and moot this.
- **Typed `connected_account_id` on `ConnectionStatus`.** throu smuggles it
  through `extra="allow"` today; it's clearly part of the lifecycle contract.
- **Result post-processing hook.** throu slims Gmail payloads before they hit
  the model. A `transform_result` hook (or documented subclass point) on the
  connector would make that supported rather than override-everything.
- **Warmup hook.** throu adds `warm_tools()` / `find_active_account()`
  extensions; decide if a generic warmup belongs on the `Connector` contract.

**Payoff.** With the first three, `ThrouComposioConnector` shrinks to curated
slugs + Gmail slimming - inherit behavior, not just shape.

## 4. UsageEvent completeness (streamed usage == final usage)

**Problem.** Sub-agent usage arriving via `ToolResult.usage` is folded into
the run total silently (`agent.py` ~line 434) - no `UsageEvent` is emitted.
A streaming consumer summing `UsageEvent`s undercounts vs `DoneEvent.usage`.

**Blocker.** throu currently uses `UsageEvent` as its pass-boundary signal
(item 1's workaround), so emitting MORE usage events would break it. Do item
1 first; then this is a two-line fix plus a test asserting
`sum(UsageEvents) == DoneEvent.usage`.

## 5. `Agent.as_tool()` parameter passthrough + event hook

**Problem.** throu's `ConsultToolSource` exists mostly for policy (intent
routing, per-turn dedup, live sub-step forwarding, source pills) - that's
fine and stays in throu. But two generic pieces are duplicated from
`_AgentToolSource`: sanitized sub-agent failure text and usage-on-ToolResult.
If `as_tool()` grew (a) extra tool-input fields forwarded as run kwargs
(e.g. an enum param the parent model fills in, mapped to model/instructions
selection) and (b) an `on_event` hook for live sub-agent step forwarding,
throu could shed most of the class.

**Caution.** This is the item most at risk of over-abstraction - only worth
it if the passthrough design stays dead simple. Fine to reject.

## 6. Smaller notes (grab-bag, no design weight)

- `DoneEvent` truncation flag: PR #3 logs a warning on
  `stop_reason == "max_tokens"`; consumers still can't see it. A
  `truncated: bool` on `DoneEvent` is one field + TS regen - held back only
  to keep PR #3 schema-neutral.
- `models/registry.py` `_CLIENT_CACHE` clients are never `aclose()`d - fine
  for servers, leaks warnings in short-lived scripts. A `shutdown()` helper
  or docs note.
- `AnthropicModel.parse_message` silently drops thinking / server-tool
  blocks; worth stating in the ModelClient boundary contract docstring.
- Token estimation: throu keeps a `_CHARS_PER_TOKEN = 4` heuristic in two
  places (orchestrator, compaction). If the framework ever grows a token
  estimator, both adopt it; until then this is throu's to unify.
- Test gaps in the core suite: non-object output schemas beyond the PR #3
  cases, provider-exception behavior in `run()` (blocked on item 2), stream
  `aclose()` (vs task-cancel) semantics.
