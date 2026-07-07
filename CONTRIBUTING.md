# Contributing to shankit

Thanks for your interest. shankit is early and deliberately small — the most
useful contributions right now are bug reports, docs improvements, sharp
questions about the API, and connectors/model clients that exercise the existing
contracts.

## The one rule that governs new API

shankit is being **harvested from a production product** ([throu](https://throu.ai))
rather than designed in a vacuum. The binding constraint is **API-commitment
risk**, not build effort — un-shipping a public API is expensive.

> **The framework is never more than one real usage ahead of itself.**
> No abstraction ships without a consumer that needs it _today_.

So a PR that adds a primitive "because someone might want it" will be asked for
the consumer that needs it now. A PR that adds an _escape hatch_ over a heavier
abstraction is the shape we prefer (see [design §1](docs/design.md#1-thesis)).
When in doubt, open an issue to discuss the shape before writing code.

## Development setup

```bash
uv venv
uv pip install -e packages/shankit -e packages/shankit-connectors \
    pytest pytest-asyncio anthropic openai ruff
```

The test suite fakes the LLM, Composio, and network boundaries, so it runs
anywhere without credentials or network access.

## The checks every PR must pass

CI runs exactly these — run them locally before opening a PR:

```bash
# Python core + connector tests (~150 tests, no network)
uv run pytest packages/shankit/tests packages/shankit-connectors/tests

# Lint (ruff, line length 100)
uv run ruff check packages scripts

# TypeScript event types must match the Python source of truth
python scripts/generate_ts_events.py --check

# TypeScript client build + tests
cd packages/client-ts && npm install && npm test
```

### Event types are generated, not hand-written

`@shankit/client`'s event types are generated from the Python pydantic models by
`scripts/generate_ts_events.py`. If you change an event shape in Python,
regenerate (drop the `--check` flag) and commit the result — CI fails on drift.

## Conventions

- **Tests target contracts, not internals.** Add tests against documented
  behavior so a refactor that preserves behavior doesn't break them.
- **Keep public signatures readable.** `typing.Optional`/`Union` are kept in
  public signatures on purpose (see the ruff config); match the surrounding
  style.
- **Errors stay human-safe.** Anything that can reach a user (an `ErrorEvent`
  message, a sanitized tool failure) must never carry secrets, keys, or raw
  tracebacks — those go to the logs.
- **Experimental means experimental.** New unstable surface lands under
  `shankit.experimental.*`; stabilizing it is a deliberate move out of that
  namespace, not a silent promotion.

## Reporting bugs

Open an issue with a minimal reproduction and what you expected. Docs gaps count
as bugs — if something in [`docs/`](docs/README.md) is wrong or missing, say so.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
</content>
