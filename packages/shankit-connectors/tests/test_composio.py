"""Composio connector tests against a mocked SDK client."""

from types import SimpleNamespace

import pytest
from shankit import Agent, ToolError, ToolNotFoundError
from shankit.messages import TextBlock, ToolUseBlock
from shankit.models.base import ModelClient, ModelResponse, ModelResponseComplete, ModelTextDelta
from shankit.usage import Usage
from shankit_connectors import ComposioConnector, ConnectionRequest, ConnectionStatus, Connector


class FakeComposio:
    """Mimics the composio v3 SDK surface the connector touches."""

    def __init__(self):
        self.executed = []
        self.raw_tool_queries = []
        self.tools = SimpleNamespace(
            get_raw_composio_tools=self._get_raw_tools, execute=self._execute
        )
        self.connected_accounts = SimpleNamespace(
            initiate=self._initiate, get=self._get_account
        )
        self.account_status = "INITIATED"

    def _get_raw_tools(self, tools=None, toolkits=None):
        self.raw_tool_queries.append({"tools": tools, "toolkits": toolkits})
        catalog = self._catalog()
        if tools is not None:
            by_slug = {(t["slug"] if isinstance(t, dict) else t.slug): t for t in catalog}
            return [by_slug[slug] for slug in tools if slug in by_slug]
        assert toolkits == ["GMAIL"]
        return catalog

    def _catalog(self):
        return [
            {
                "slug": "GMAIL_SEND_EMAIL",
                "description": "Send an email via Gmail.",
                "input_parameters": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"],
                },
            },
            SimpleNamespace(  # SDKs sometimes return objects, not dicts
                slug="GMAIL_SEARCH",
                description="Search the mailbox.",
                input_parameters={"type": "object", "properties": {}},
            ),
        ]

    def _execute(self, slug, user_id, arguments):
        self.executed.append((slug, user_id, arguments))
        if slug == "GMAIL_SEND_EMAIL" and not arguments.get("to"):
            return {"successful": False, "error": "Missing recipient", "data": None}
        return {"successful": True, "error": None, "data": {"id": "msg_1"}}

    def _initiate(self, user_id, auth_config_id):
        assert user_id == "u42"
        assert auth_config_id == "ac_1"
        return SimpleNamespace(id="conn_1", redirect_url="https://connect.example/oauth")

    def _get_account(self, connection_id):
        assert connection_id == "conn_1"
        return {"status": self.account_status}


@pytest.fixture
def connector():
    return ComposioConnector(
        toolkit="GMAIL",
        user_id=lambda ctx: ctx["user_id"],
        client=FakeComposio(),
    )


async def test_list_tools_converts_to_anthropic_shape(connector):
    defs = await connector.list_tools(context={"user_id": "u42"})
    assert [d.name for d in defs] == ["GMAIL_SEND_EMAIL", "GMAIL_SEARCH"]
    assert defs[0].input_schema["required"] == ["to"]
    assert defs[0].description == "Send an email via Gmail."


async def test_allowlist_filters_tools():
    connector = ComposioConnector(
        toolkit="GMAIL", tools=["gmail_search"], client=FakeComposio()
    )
    defs = await connector.list_tools()
    assert [d.name for d in defs] == ["GMAIL_SEARCH"]
    # An allowlisted connector fetches exactly its slugs, not the whole toolkit.
    assert connector._client.raw_tool_queries == [
        {"tools": ["GMAIL_SEARCH"], "toolkits": None}
    ]
    with pytest.raises(ToolNotFoundError):
        await connector.execute("GMAIL_SEND_EMAIL", {"to": "x"})


async def test_execute_resolves_user_from_context(connector):
    result = await connector.execute(
        "GMAIL_SEND_EMAIL", {"to": "a@b.c"}, context={"user_id": "u42"}
    )
    assert result.content == '{"id": "msg_1"}'
    assert connector._client.executed == [("GMAIL_SEND_EMAIL", "u42", {"to": "a@b.c"})]


async def test_execute_default_user_when_no_extractor():
    connector = ComposioConnector(toolkit="GMAIL", client=FakeComposio())
    await connector.execute("GMAIL_SEARCH", {})
    assert connector._client.executed[0][1] == "default"


async def test_vendor_failure_is_tool_error(connector):
    with pytest.raises(ToolError, match="Missing recipient"):
        await connector.execute("GMAIL_SEND_EMAIL", {}, context={"user_id": "u42"})


async def test_connection_lifecycle(connector):
    ctx = {"user_id": "u42"}
    request = await connector.initiate(ctx, auth_config_id="ac_1")
    assert request == ConnectionRequest(
        connection_id="conn_1", redirect_url="https://connect.example/oauth"
    )

    status = await connector.check_status(ctx, connection_id="conn_1")
    assert status == ConnectionStatus(connection_id="conn_1", status="pending")

    connector._client.account_status = "ACTIVE"
    adopted = await connector.adopt(ctx, connection_id="conn_1")
    assert adopted.status == "active"


async def test_connector_is_a_plain_tool_source(connector):
    """The agent loop uses a connector without knowing the lifecycle exists."""

    class ScriptedModel(ModelClient):
        def __init__(self):
            self.turn = 0

        async def complete(self, request):
            self.turn += 1
            if self.turn == 1:
                return ModelResponse(
                    content=[
                        ToolUseBlock(
                            id="t1", name="GMAIL_SEND_EMAIL", input={"to": "a@b.c"}
                        )
                    ],
                    stop_reason="tool_use",
                    usage=Usage(requests=1),
                )
            return ModelResponse(
                content=[TextBlock(text="sent")], stop_reason="end_turn", usage=Usage(requests=1)
            )

        async def stream(self, request):
            response = await self.complete(request)
            if response.text:
                yield ModelTextDelta(text=response.text)
            yield ModelResponseComplete(response=response)

    agent = Agent(
        name="mailer", model="fake", model_client=ScriptedModel(), tools=[connector]
    )
    result = await agent.run("send it", context={"user_id": "u42"}, output_type=str)
    assert result.text == "sent"
    assert result.trajectory[0].tool == "GMAIL_SEND_EMAIL"
    assert isinstance(connector, Connector)
