"""ObservedModel and the record/replay cassette: the two clients that wrap
another one, exercised through a real Agent where the loop matters."""

import json

import pytest
from throukit import Agent, ModelError, Usage, tool
from throukit.messages import Message, TextBlock
from throukit.models import ModelCall, ModelRequest, ObservedModel
from throukit.models.base import ModelResponseComplete, ModelTextDelta
from throukit.testing import Cassette, CassetteMiss, CassetteModel, ScriptedModel, ToolCall, Turn


@tool
def order_status(order_id: int) -> str:
    """Look up an order's shipping status."""
    return f"order {order_id}: shipped"


def _request(text: str = "hi", **extra) -> ModelRequest:
    return ModelRequest(
        model="m", messages=[Message(role="user", content=[TextBlock(text=text)])], **extra
    )


# -- ObservedModel ----------------------------------------------------------


async def test_observed_reports_each_call_of_a_run():
    calls: list[ModelCall] = []
    inner = ScriptedModel(
        [
            Turn(
                "Checking.",
                tool_calls=[ToolCall("order_status", {"order_id": 7})],
                usage=Usage(input_tokens=5),
            ),
            "Shipped.",
        ]
    )
    agent = Agent(
        name="support",
        model="m",
        model_client=ObservedModel(inner, after=calls.append),
        tools=[order_status],
    )
    events = [e async for e in agent.stream("Where is order 7?")]
    assert events[-1].type == "done"
    assert [c.ok for c in calls] == [True, True]
    assert all(c.streamed and c.request.agent == "support" for c in calls)
    assert calls[0].response.usage.input_tokens == 5
    assert calls[0].first_text is not None
    assert calls[0].duration >= calls[0].first_text


async def test_observed_reports_a_failed_call_and_reraises():
    calls: list[ModelCall] = []
    model = ObservedModel(ScriptedModel([ModelError("boom", provider="x")]), after=calls.append)
    with pytest.raises(ModelError):
        await model.complete(_request())
    (call,) = calls
    assert not call.ok
    assert not call.cancelled
    assert isinstance(call.error, ModelError)
    assert not call.streamed
    assert call.first_text is None


async def test_observed_stream_closed_after_response_counts_as_ok():
    calls: list[ModelCall] = []
    stream = ObservedModel(ScriptedModel(["done"]), after=calls.append).stream(_request())
    async for event in stream:
        if isinstance(event, ModelResponseComplete):
            break
    await stream.aclose()
    assert [c.ok for c in calls] == [True]


async def test_observed_stream_failing_after_response_counts_as_ok():
    class _ThenFails(ScriptedModel):
        async def stream(self, request):
            async for event in super().stream(request):
                yield event
            raise RuntimeError("teardown failed")

    calls: list[ModelCall] = []
    with pytest.raises(RuntimeError):
        [
            e
            async for e in ObservedModel(_ThenFails(["done"]), after=calls.append).stream(
                _request()
            )
        ]
    (call,) = calls
    assert call.ok
    assert call.error is None


async def test_observed_stream_closed_before_response_is_cancelled():
    calls: list[ModelCall] = []
    stream = ObservedModel(ScriptedModel(["one two three"]), after=calls.append).stream(_request())
    async for event in stream:
        assert isinstance(event, ModelTextDelta)
        break
    await stream.aclose()
    (call,) = calls
    assert not call.ok
    assert call.cancelled


async def test_observed_before_can_refuse_and_after_errors_are_swallowed():
    def refuse(request: ModelRequest) -> None:
        raise ModelError("over budget", provider="cap", retryable=False)

    inner = ScriptedModel(["never"])
    with pytest.raises(ModelError, match="over budget"):
        await ObservedModel(inner, before=refuse).complete(_request())
    with pytest.raises(ModelError, match="over budget"):
        [e async for e in ObservedModel(inner, before=refuse).stream(_request())]
    assert inner.requests == []

    def broken(call: ModelCall) -> None:
        raise RuntimeError("observer bug")

    response = await ObservedModel(inner, after=broken).complete(_request())
    assert response.text == "never"


# -- Cassette ---------------------------------------------------------------


async def test_cassette_records_then_replays_without_the_live_client(tmp_path):
    tape = Cassette(tmp_path)
    live = ScriptedModel(["Hello there friend."])
    recorder = CassetteModel("p", lambda: live, tape, "record")
    events = [e async for e in recorder.stream(_request())]
    assert events[-1].response.text == "Hello there friend."
    assert len(list(tmp_path.glob("p-*.json"))) == 1

    def no_live():
        raise AssertionError("replay built the live client")

    player = CassetteModel("p", no_live, tape, "replay")
    events = [e async for e in player.stream(_request())]
    assert [e.text for e in events if isinstance(e, ModelTextDelta)] == [
        "Hello ",
        "there ",
        "friend.",
    ]
    assert (await player.complete(_request())).text == "Hello there friend."


async def test_cassette_keys_ignore_ids_agent_and_scrubbed_text(tmp_path):
    tape = Cassette(tmp_path, scrub=[(r"It is \d+ o'clock", "It is <now>")])
    await CassetteModel("p", lambda: ScriptedModel(["ok"]), tape, "record").complete(
        _request(
            "It is 3 o'clock. Order 3f2b8c1e-0000-4000-8000-000000000001, call toolu_abc123",
            agent="a",
        )
    )
    replayed = await CassetteModel("p", lambda: None, tape, "replay").complete(
        _request(
            "It is 9 o'clock. Order 11111111-2222-4333-8444-555555555555, call toolu_zzz9",
            agent="b",
        )
    )
    assert replayed.text == "ok"


async def test_cassette_miss_raises_and_writes_a_diff(tmp_path):
    tape = Cassette(tmp_path)
    await CassetteModel("p", lambda: ScriptedModel(["ok"]), tape, "record").complete(
        _request("first")
    )
    with pytest.raises(CassetteMiss) as caught:
        await CassetteModel("p", lambda: None, tape, "replay").complete(_request("second"))
    assert not caught.value.retryable
    (report,) = (tmp_path / "_misses").glob("*.json")
    body = json.loads(report.read_text())
    assert body["closest"].startswith("p-")
    assert any("second" in line for line in body["diff"])


async def test_cassette_rekey_moves_recordings_to_new_keys(tmp_path):
    await CassetteModel("p", lambda: ScriptedModel(["ok"]), Cassette(tmp_path), "record").complete(
        _request("ticket T-42")
    )
    scrubbed = Cassette(tmp_path, scrub=[(r"T-\d+", "<ticket>")])
    assert scrubbed.rekey() == 1
    assert scrubbed.rekey() == 0
    replayed = await CassetteModel("p", lambda: None, scrubbed, "replay").complete(
        _request("ticket T-7")
    )
    assert replayed.text == "ok"
