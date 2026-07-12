# shankit documentation

The complete guide to shankit — an AI agent framework that takes the
integration seam as seriously as the agent loop itself, without imposing how
you authenticate.

New here? Start with **[Getting started](getting-started.md)**, then read
**[Core concepts](concepts.md)** to understand the model. Everything else is
reference you can reach for when you need it.

## Guides

| Doc | What it covers |
|---|---|
| **[Getting started](getting-started.md)** | Install, define your first agent, run it structured and streamed. |
| **[Core concepts](concepts.md)** | The tool seam, the opaque per-run context, the two run modes, multi-agent, models, and the uniform error contract. |
| **[File-first definitions](file-first.md)** | Authoring agents as Markdown files: the full frontmatter reference, prompt substitution, and import-vs-registry references. |
| **[Connectors](connectors.md)** | The opt-in OAuth integration layer: the connector contract, the connection lifecycle, and the official Composio implementation. |
| **[Durability & networks](durability.md)** | The `Checkpointer` contract, its shipped stores, and the experimental router-driven network for code-controlled flow and human-in-the-loop pauses. |
| **[Evals](evals.md)** | Datasets, scorers, model-graded judging, and trajectory assertions over `RunResult`. |

## Reference

- **[Examples](../examples)** — a runnable script for every major surface.
- **[Design decision record](design.md)** — the living spec: the thesis, the
  competitive landscape, and _why_ every primitive is shaped the way it is.
- **[Deferred improvements](deferred-improvements.md)** — design-level changes
  considered and consciously deferred, with the reasoning.

## Packages

| Package | Role |
|---|---|
| [`shankit`](../packages/shankit) | The Python core: the agent, the tool seam, models, files, durability, evals, and the experimental graph. |
| [`shankit-connectors`](../packages/shankit-connectors) | The opt-in OAuth connector layer + the official Composio connector. |
| [`@shankit/client`](../packages/client-ts) | The typed TypeScript consumer for agent event streams over SSE. |

---

Something unclear or missing? Open an issue — docs gaps are bugs.
</content>
