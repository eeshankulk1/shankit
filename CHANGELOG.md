# Changelog

Each release lists what changed for people using Throukit. Throukit follows
[Semantic Versioning](https://semver.org/) once it reaches 1.0. Before then, a
minor release (0.x.0) may break the API and says so under **Breaking**, and
anything under `throukit.experimental` may change in any release.

## Unreleased

### Breaking

- **Renamed from shankit to Throukit.** The packages are now `throukit`,
  `throukit-connectors` and `@throukit/client`, imported as `throukit` and
  `throukit_connectors`. `ShankitError` is now `ThroukitError`. Logger names
  (`throukit`, `throukit.connectors`) and tracing span names
  (`throukit.agent.run`, `throukit.agent.stream`, `throukit.tool.execute`)
  follow the new name, so update any log filters or dashboards that matched
  the old ones. `SqliteCheckpointer` now uses a table named
  `throukit_checkpoints`, so checkpoints saved under the old name are not read.
  The old names have no aliases.
- `AnthropicModel(prompt_cache="5m" | "1h" | None)` replaces
  `cache_system_and_tools`.
- `throukit.evals` has a new runner. `Case(name=...)` is now `Case(id=...)`,
  `Case.metadata` is now `Case.tags`, `Case.input` may be any value,
  `Dataset` is gone (cases are a list; `load_cases()` reads JSONL), and a
  `CaseResult` holds `metrics` and `details` rather than `case`, `result` and
  `scores`. `evaluate()` takes the list of cases and returns an `EvalReport`
  as before; `scorers` and `trajectory` are unchanged.

### Added

- Evals for real spend: `Suite` subclasses with repeated trials, pass bars
  (`Bar.parse("rate(correct) >= 0.9")`), a hard budget (`Meter`,
  `BudgetExceeded`), a cap across runs (`SessionCap`, POSIX only), Markdown and JSON
  reports, and a command line (`EvalCLI`) with dry runs and exit codes.
  `evaluate()` takes `bars=`, `trials=`, `budget_usd=` and `price=`.
- `AnthropicModel` places its own prompt-cache breakpoints (the system
  prompt, the newest prompts, the last block and the shared prefix);
  `volatile=True` on a `TextBlock` or `ImageBlock` keeps per-turn content
  out of the cached prefix, `Usage.cache_write_1h_tokens` counts hour-long
  cache writes, and `default_reasoning` fills in when a request sets none.
- Per-model usage: `Usage.by_model`, `per_model()` and `for_model()`, so a run
  that mixes models can be priced. `ModelRequest.agent` names the agent that
  made each call.
- `get_provider()` reads a registered provider, and `register_provider()`
  returns the factory it replaced, so a provider can be wrapped.
- `throukit.testing.use_model()` and `tool_call()` for tests.
- `ObservedModel` (from `throukit.models`) wraps a model client: a `before`
  hook can refuse a call (a spend cap), and an `after` hook gets one
  `ModelCall` per call with timing, time to first text, and the response or
  error.
- `throukit.testing.Cassette` and `CassetteModel` record real model calls once
  and replay them in tests, with ids scrubbed from the keys.
- `RunControl.send(on_delivered=)` and `RunControl.undelivered()`.
- `WorkspaceTools.bash_def`, `resolve_env` and `command_timeout()` are public.

### Fixed

- `stream()`'s tracing span is current only while the run works, never across
  a `yield`, so closing an abandoned stream from another task no longer breaks
  OpenTelemetry context.
- Whitespace-only text blocks are dropped before an Anthropic request, which
  the API rejects.

### Project

- A release workflow that publishes to PyPI and npm from a version tag, and a
  documentation site built from `docs/`.

## 0.1.0 - not yet published

The first version.

- `Agent` with two run modes over one loop: `run()` for a typed, validated
  result and `stream()` for a typed event stream, with conversation history,
  prompt caching, parallel tool calls and token accounting across sub-agents.
- The tool seam: `@tool` functions, MCP servers, aggregation, native
  toolsets, multimodal tool results, a size cap on tool results, an artifacts
  channel for structured content, and `ToolResult.end_run`.
- Multi-agent: `agent.as_tool()`, `DelegateToolSource` with per-run budgets,
  and `CloneToolSource` for copies of an agent that split its work, plus
  `RunControl` to cancel, message, budget and checkpoint a run from outside.
- Workspaces (`Workspace`, `LocalWorkspace`, `WorkspaceTools`) for agents that
  read and write files, images included.
- Models: Anthropic and OpenAI clients behind one `ModelClient` contract,
  per-model request rules so one request works on every Claude model,
  `register_provider()`, and `ScriptedModel` for tests.
- Agents defined as Markdown files (`load_agent`, `load_agents`).
- Durability: the `Checkpointer` contract with in-memory and SQLite stores,
  and an experimental router-driven graph with human approval pauses.
- Evals: datasets, scorers, model-graded judging and trajectory assertions.
- `throukit-connectors`: the OAuth connection lifecycle and a Composio
  connector.
- `@throukit/client`: a typed TypeScript consumer for event streams over SSE,
  with event types generated from the Python models.
