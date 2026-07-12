# shankit-connectors

The **opt-in** connector layer for [shankit](https://github.com/eeshankulk1/shankit).

A connector *is* a shankit tool source, plus ownership of the connection
lifecycle (initiate / check-status / adopt-existing) for sources that need
OAuth. Local-function and MCP users never touch this package — that's the
point (design §4): the framework defines the integration **contract**, never
the **system**. Auth mechanism, credential storage, and vendor choice remain
yours.

Ships the connector contract and one official implementation: **Composio**
(`pip install shankit-connectors[composio]`).

```python
from shankit_connectors import ComposioConnector

gmail = ComposioConnector(
    toolkit="GMAIL",
    user_id=lambda ctx: ctx["user_id"],  # you decide what identity means
)

# connection lifecycle (drive your own connect UI with it)
request = await gmail.initiate(context={"user_id": "u1"}, auth_config_id="ac_...")
print(request.redirect_url)
status = await gmail.check_status(context={"user_id": "u1"}, connection_id=request.connection_id)

# and it's just a tool source
agent = Agent(name="mail", model="anthropic:claude-sonnet-4-5", tools=[gmail])
```

The litmus for this layer: you could swap Composio for a completely different
system without the framework noticing.

Full guide — the contract, the lifecycle, multi-tenant identity scoping, and
every `ComposioConnector` option: [docs/connectors.md](../../docs/connectors.md).
