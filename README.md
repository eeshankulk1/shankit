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

## Status

Planning / pre-implementation. The framework is being harvested from an existing
product ([throu](https://throu.ai)) rather than designed in a vacuum.

## Docs

- [Framework Design](docs/design.md) — the living design spec and decision record.
