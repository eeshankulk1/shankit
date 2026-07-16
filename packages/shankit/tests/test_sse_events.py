import json

import pytest
from shankit import DoneEvent, ShankitError, StepEvent, TextDeltaEvent, format_sse, sse_stream
from shankit.events import Artifact, ArtifactEvent, agent_event_adapter


def test_format_sse():
    message = format_sse(TextDeltaEvent(text="hi"))
    assert message.startswith("event: text_delta\ndata: ")
    assert message.endswith("\n\n")
    payload = json.loads(message.split("data: ", 1)[1])
    assert payload == {"type": "text_delta", "text": "hi"}


def test_events_roundtrip_via_discriminator():
    adapter = agent_event_adapter()
    for event in [
        TextDeltaEvent(text="x"),
        StepEvent(id="s1", title="Did a thing", status="done"),
        ArtifactEvent(artifact=Artifact(type="email_card", data={"subject": "s"})),
        DoneEvent(text="fin", output={"k": 1}),
    ]:
        parsed = adapter.validate_json(event.model_dump_json())
        assert parsed == event


async def test_sse_stream_emits_error_event_for_framework_errors():
    async def events():
        yield TextDeltaEvent(text="a")
        raise ShankitError("run exploded politely")

    chunks = [chunk async for chunk in sse_stream(events())]
    assert "text_delta" in chunks[0]
    assert "event: error" in chunks[1]
    assert "run exploded politely" in chunks[1]


async def test_sse_stream_sanitizes_unexpected_errors():
    async def events():
        yield TextDeltaEvent(text="a")
        raise RuntimeError("secret stack details")

    chunks = [chunk async for chunk in sse_stream(events())]
    assert "secret" not in chunks[1]
    assert "unexpectedly" in chunks[1]


async def test_sse_stream_single_terminal_error():
    """Agent.stream yields its own error event before re-raising unexpected
    exceptions; sse_stream must not append a second terminal error."""
    from shankit import ErrorEvent

    async def events():
        yield TextDeltaEvent(text="a")
        yield ErrorEvent(message="already terminal", code="unexpected")
        raise RuntimeError("re-raised after the event")

    chunks = [chunk async for chunk in sse_stream(events())]
    assert len(chunks) == 2
    assert sum("event: error" in c for c in chunks) == 1


async def test_sse_stream_raise_mode():
    async def events():
        raise RuntimeError("boom")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError):
        async for _ in sse_stream(events(), errors="raise"):
            pass
