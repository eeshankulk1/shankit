# Connectors

The opt-in integration layer for tools that need OAuth-style account
linking. Local-function and MCP users never touch this; everything here
lives in the separate [`shankit-connectors`](../packages/shankit-connectors)
package.

```bash
pip install shankit-connectors[composio]
```

The litmus for this layer (design §4): **you could swap Composio for a
completely different system without the framework noticing.** Nothing
Composio-shaped leaks into the core; a connector is just a `ToolSource`
with a connection lifecycle bolted on.

## The contract

A `Connector` is a `shankit.ToolSource` — the same two-method seam the agent
loop already speaks (`list_tools` / `execute`) — **plus** three lifecycle
methods consumed only by your connection UI:

| Method | Purpose |
|---|---|
| `initiate(context, **params)` | Start connecting an account; returns a `ConnectionRequest` (usually carrying a `redirect_url` for your UI) |
| `check_status(context, connection_id=...)` | Poll whether a pending connection became `active` |
| `adopt(context, connection_id=...)` | Adopt a connection created outside this app (vendor dashboard, previous system) |

`ConnectionStatus.status` is one of `pending`, `active`, `failed`,
`expired`. Statuses a connector doesn't recognize report as `failed` —
terminal — so your connect UI never polls a dead connection forever.

The split exists because the runtime seam (what the loop touches) changes
at a different rate than the vendor-shaped lifecycle (what your connect
flow touches). The loop never sees these methods; your connection UI never
sees the loop.

## Identity: the `user_id` extractor

The framework never learns what a "user" is. A connector resolves *whose*
credentials to use by reading your plain data off the opaque per-run
context, through an extractor **you** supply:

```python
from shankit_connectors import ComposioConnector

gmail = ComposioConnector(
    toolkit="GMAIL",
    user_id=lambda ctx: ctx["user_id"],   # you decide what identity means
)
agent = Agent(name="mail", model="anthropic:claude-sonnet-4-5", tools=[gmail])

await agent.run("Summarize today's inbox.", context={"user_id": "u42"}, output_type=str)
```

Omit `user_id` for single-tenant setups (Composio's `"default"` user is
used).

When an extractor **is** configured, the lifecycle methods are scoped to
it: `check_status`/`adopt` verify the connection actually belongs to the
user the context resolves to, and raise `PermissionError` otherwise — a
leaked or guessed connection id can't be linked cross-tenant.

## The connection flow

```python
# 1. Your connect endpoint: start the OAuth dance.
request = await gmail.initiate(ctx, auth_config_id="ac_...")   # vendor-specific params
redirect_user_to(request.redirect_url)

# 2. Your callback/polling endpoint.
status = await gmail.check_status(ctx, connection_id=request.connection_id)
if status.status == "active":
    ...  # store status.account_id against your user

# 3. Or adopt an account connected elsewhere (dashboard, migration).
status = await gmail.adopt(ctx, connection_id="conn_existing")
```

## ComposioConnector options

| Option | What it does |
|---|---|
| `toolkit` | The Composio toolkit slug (e.g. `"GMAIL"`) |
| `tools=[...]` | Allowlist of tool slugs to expose — **recommended**, toolkits can run to hundreds of tools |
| `tools_cache_ttl=600` | Cache the tool catalog for N seconds. The loop lists tools every run; an uncached catalog costs a vendor round-trip per turn. Identity only matters at `execute`, so one cache per connector is safe |
| `client=` / `api_key=` | Bring an existing `composio.Composio` client, or construct one |
| `skip_version_check=True` | Follow the latest toolkit version when the client pins none. Left off, an unpinned client surfaces an actionable `ToolError` telling you to pin or opt in — an explicit choice, because a breaking toolkit release can change behavior without a code change |

### Slimming vendor responses

Vendor payloads are often bulky. Subclass and override `transform_result`
to shape what the model sees:

```python
class SlimGmail(ComposioConnector):
    def transform_result(self, name, data, context=None):
        if name == "GMAIL_SEARCH":
            return [{"id": m["id"], "subject": m["subject"]} for m in data["messages"]]
        return data
```

## Error behavior at the seam

Connectors follow the framework's uniform tool-error contract
([concepts](concepts.md)): a vendor-reported failure raises `ToolError`
(model-visible, recorded on the trajectory); anything else is sanitized by
the loop. Known, actionable vendor failures — like Composio's
version-pinning requirement — are translated into `ToolError`s that state
the remedy rather than an opaque "failed unexpectedly".

## Writing your own connector

Subclass `shankit_connectors.Connector`, implement the two seam methods and
the three lifecycle methods, and keep every vendor concept inside your
module. `ComposioConnector`
([source](../packages/shankit-connectors/src/shankit_connectors/composio.py))
is the reference implementation; its test suite
([tests](../packages/shankit-connectors/tests/test_composio.py)) shows how
to verify one against a mocked SDK, including the agent-loop integration
test proving a connector is a plain `ToolSource`.

## See also

- [Core concepts](concepts.md) — the tool seam and the opaque per-run context.
- [Design §4](design.md) — why the connector layer is a separate package.

---

[Back to docs](README.md)
