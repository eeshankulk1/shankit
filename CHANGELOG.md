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

### Added

- A release workflow that publishes to PyPI and npm from a version tag, and a
  documentation site built from `docs/`.

## 0.1.0 - not yet published

The first version, built inside [Throu](https://throu.ai) and used there in
production before any public release.

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
