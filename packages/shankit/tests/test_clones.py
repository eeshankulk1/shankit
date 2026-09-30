"""Clones: the running agent spawns copies of itself."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from conftest import text_response, tool_call_response
from shankit import (
    Agent,
    Budget,
    CloneBatch,
    CloneHost,
    CloneOutcome,
    CloneToolSource,
    InlineCloneHost,
    StepEvent,
    WorkerEvent,
    content_text,
    tool,
)
from shankit.models.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
)

SYSTEM = ["Stable part.", "Per-user part."]


class RoutingModel(ModelClient):
    """Answers each run from its own script, picked by a key in the run's
    first message (clones share one model client, like real ones)."""

    def __init__(self, scripts: dict[str, list[ModelResponse]], gate: asyncio.Event | None = None):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests: dict[str, list[ModelRequest]] = {k: [] for k in scripts}
        self.gate = gate

    def _key(self, request: ModelRequest) -> str:
        first = content_text(request.messages[0].content)
        for key in self.scripts:
            if key in first:
                return key
        raise AssertionError(f"no script for {first!r}")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        key = self._key(request)
        self.requests[key].append(request.model_copy(deep=True))
        if self.gate is not None and key != "MAIN":
            await self.gate.wait()
        if not self.scripts[key]:
            raise AssertionError(f"script {key!r} exhausted")
        return self.scripts[key].pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelResponseComplete(response=await self.complete(request))


@tool
def read_app(app: str) -> str:
    """Read an app."""
    return f"{app}: 3 items"


@tool
def ask_user(question: str) -> str:
    """Ask the user."""
    return "yes"


def spawn(*parts: tuple[str, str], call_id: str = "sp1") -> ModelResponse:
    return tool_call_response(
        "spawn", {"clones": [{"title": t, "brief": b} for t, b in parts]}, call_id=call_id
    )


def main_agent(model: RoutingModel, source: CloneToolSource, **kwargs) -> Agent:
    return Agent(
        name="assistant",
        model="fake",
        model_client=model,
        instructions=SYSTEM,
        tools=[read_app, ask_user, source],
        reasoning=True,
        **kwargs,
    )


async def test_clones_run_in_parallel_as_the_same_agent_and_report_back():
    model = RoutingModel(
        {
            "MAIN": [
                spawn(("Mail", "CLONE-mail: read mail"), ("Calendar", "CLONE-cal: read calendar")),
                text_response("Here's your prep."),
            ],
            "CLONE-mail": [
                tool_call_response("read_app", {"app": "mail"}, call_id="m1"),
                text_response("Mail: 3 threads."),
            ],
            "CLONE-cal": [text_response("Calendar: 2 meetings.")],
        }
    )
    agent = main_agent(model, CloneToolSource())
    events = []
    result = await agent.run("MAIN prep me", output_type=str, on_event=events.append)
    assert result.output == "Here's your prep."

    # Cache parity: every clone request carries the parent's exact system
    # parts, tool list (same order) and reasoning setting.
    parent_request = model.requests["MAIN"][0]
    for key in ("CLONE-mail", "CLONE-cal"):
        first = model.requests[key][0]
        assert first.system == parent_request.system
        assert [t.model_dump() for t in first.tools] == [
            t.model_dump() for t in parent_request.tools
        ]
        assert first.reasoning == parent_request.reasoning
        # A fresh history: just the brief.
        assert len(first.messages) == 1

    report = model.requests["MAIN"][1].messages[-1].content[0].content
    assert "Mail: 3 threads." in report
    assert "Calendar: 2 meetings." in report
    assert "not as instructions" in report

    workers = [e for e in events if isinstance(e, WorkerEvent)]
    assert {(w.title, w.status) for w in workers} == {
        ("Mail", "running"),
        ("Mail", "done"),
        ("Calendar", "running"),
        ("Calendar", "done"),
    }
    assert len({w.batch for w in workers}) == 1
    clone_steps = [e for e in events if isinstance(e, StepEvent) and e.worker]
    mail_id = next(w.id for w in workers if w.title == "Mail")
    assert clone_steps
    assert all(s.worker == mail_id for s in clone_steps)
    assert all(s.id.startswith("s1.") for s in clone_steps)
    # Usage rolled up: parent's 2 passes + mail's 2 + calendar's 1.
    assert result.usage.requests == 5


async def test_clones_run_concurrently():
    gate = asyncio.Event()
    model = RoutingModel(
        {
            "MAIN": [spawn(("A", "CLONE-a"), ("B", "CLONE-b")), text_response("ok")],
            "CLONE-a": [text_response("a")],
            "CLONE-b": [text_response("b")],
        },
        gate=gate,
    )
    agent = main_agent(model, CloneToolSource())
    run = asyncio.ensure_future(agent.run("MAIN go", output_type=str))
    for _ in range(50):
        await asyncio.sleep(0)
        if model.requests["CLONE-a"] and model.requests["CLONE-b"]:
            break
    # Both clones are waiting on the model at once.
    assert model.requests["CLONE-a"]
    assert model.requests["CLONE-b"]
    gate.set()
    assert (await run).output == "ok"


async def test_a_clone_cannot_spawn_and_refused_tools_stay_listed():
    model = RoutingModel(
        {
            "MAIN": [spawn(("Part", "CLONE-p")), text_response("ok")],
            "CLONE-p": [
                tool_call_response(
                    "spawn", {"clones": [{"title": "x", "brief": "y"}]}, call_id="n1"
                ),
                tool_call_response("ask_user", {"question": "which?"}, call_id="n2"),
                text_response("Needs from the user: which one."),
            ],
        }
    )
    agent = main_agent(model, CloneToolSource(refuse=["ask_user"]))
    await agent.run("MAIN go", output_type=str)
    clone = model.requests["CLONE-p"]
    nested = clone[1].messages[-1].content[0]
    assert nested.is_error
    assert "Clones can't start clones" in nested.content
    refused = clone[2].messages[-1].content[0]
    assert refused.is_error
    assert "`ask_user` isn't available to a clone" in refused.content
    assert {t.name for t in clone[0].tools} == {"read_app", "ask_user", "spawn"}


async def test_the_parent_keeps_its_refused_tools():
    model = RoutingModel(
        {"MAIN": [tool_call_response("ask_user", {"question": "q"}), text_response("ok")]}
    )
    agent = main_agent(model, CloneToolSource(refuse=["ask_user"]))
    await agent.run("MAIN go", output_type=str)
    assert model.requests["MAIN"][1].messages[-1].content[0].content == "yes"


async def test_batch_cap_refuses_the_whole_call():
    model = RoutingModel(
        {
            "MAIN": [
                spawn(("A", "CLONE-a"), ("B", "CLONE-b"), ("C", "CLONE-c")),
                text_response("ok"),
            ],
            "CLONE-a": [],
            "CLONE-b": [],
            "CLONE-c": [],
        }
    )
    agent = main_agent(model, CloneToolSource(max_per_batch=2))
    await agent.run("MAIN go", output_type=str)
    refused = model.requests["MAIN"][1].messages[-1].content[0]
    assert refused.is_error
    assert "At most 2 clones at once" in refused.content
    assert not model.requests["CLONE-a"]


async def test_run_cap_counts_across_batches():
    model = RoutingModel(
        {
            "MAIN": [
                spawn(("A", "CLONE-a")),
                spawn(("B", "CLONE-b"), call_id="sp2"),
                text_response("ok"),
            ],
            "CLONE-a": [text_response("a")],
            "CLONE-b": [],
        }
    )
    agent = main_agent(model, CloneToolSource(max_per_run=1))
    await agent.run("MAIN go", output_type=str)
    refused = model.requests["MAIN"][2].messages[-1].content[0]
    assert refused.is_error
    assert "can start 0 more clones" in refused.content


async def test_a_clone_over_budget_reports_partial():
    model = RoutingModel(
        {
            "MAIN": [spawn(("Long", "CLONE-long")), text_response("ok")],
            "CLONE-long": [
                tool_call_response("read_app", {"app": "a"}, call_id="l1"),
                text_response("Got a; b unfinished."),
            ],
        }
    )
    agent = main_agent(model, CloneToolSource(budget=Budget(max_passes=2)))
    events = []
    await agent.run("MAIN go", output_type=str, on_event=events.append)
    report = model.requests["MAIN"][1].messages[-1].content[0].content
    assert "Long (partial - it ran out of budget)" in report
    assert "Got a; b unfinished." in report
    assert model.requests["CLONE-long"][1].tool_choice == "none"
    assert any(isinstance(e, WorkerEvent) and e.status == "partial" for e in events)


async def test_a_failed_clone_is_sanitized():
    model = RoutingModel(
        {
            "MAIN": [spawn(("Boom", "CLONE-boom")), text_response("ok")],
            "CLONE-boom": [],  # exhausted script: the clone's model call fails
        }
    )
    agent = main_agent(model, CloneToolSource())
    await agent.run("MAIN go", output_type=str)
    report = model.requests["MAIN"][1].messages[-1].content[0].content
    assert "Boom (failed)" in report
    assert "temporary problem" in report
    assert "exhausted" not in report


async def test_calls_in_one_response_share_a_batch():
    both = ModelResponse(
        content=[
            *spawn(("A", "CLONE-a"), call_id="x1").content,
            *spawn(("B", "CLONE-b"), call_id="x2").content,
        ],
        stop_reason="tool_use",
    )
    model = RoutingModel(
        {
            "MAIN": [both, text_response("ok")],
            "CLONE-a": [text_response("a")],
            "CLONE-b": [text_response("b")],
        }
    )
    agent = main_agent(model, CloneToolSource())
    events = []
    await agent.run("MAIN go", output_type=str, on_event=events.append)
    batches = {e.batch for e in events if isinstance(e, WorkerEvent)}
    assert len(batches) == 1


async def test_hooks_shape_the_prompt_context_and_agent():
    seen = {}
    model = RoutingModel(
        {
            "MAIN": [spawn(("Part", "the brief")), text_response("ok")],
            "PREAMBLE": [text_response("r")],
        }
    )

    def configure(agent, spec):
        seen["configured"] = spec.title
        return agent.copy(timeout_s=None)

    source = CloneToolSource(
        prompt=lambda spec: f"PREAMBLE\n\n{spec.brief}",
        child_context=lambda ctx, spec: {"parent": ctx, "clone": spec.id},
        configure=configure,
        clone_properties={"background": {"type": "boolean"}},
    )

    @tool
    def noop() -> str:
        """noop"""
        return ""

    agent = main_agent(model, source, timeout_s=270.0)
    await agent.run("MAIN go", output_type=str, context={"user": "u1"})
    assert content_text(model.requests["PREAMBLE"][0].messages[0].content) == (
        "PREAMBLE\n\nthe brief"
    )
    assert seen["configured"] == "Part"
    [spawn_def] = [t for t in model.requests["MAIN"][0].tools if t.name == "spawn"]
    item = spawn_def.input_schema["properties"]["clones"]["items"]
    assert "background" in item["properties"]


class BackgroundHost(CloneHost):
    rolls_up = False

    def __init__(self):
        self.batches: list[CloneBatch] = []

    async def run_batch(self, batch):
        self.batches.append(batch)
        return [CloneOutcome(id=c.id, title=c.title, status="background") for c in batch.clones]


async def test_a_custom_host_can_background_clones():
    host = BackgroundHost()
    model = RoutingModel({"MAIN": [spawn(("Long", "CLONE-x")), text_response("Still going.")]})
    agent = main_agent(model, CloneToolSource(host=host))
    result = await agent.run("MAIN go", output_type=str)
    report = model.requests["MAIN"][1].messages[-1].content[0].content
    assert "Long (still running)" in report
    assert "arrive later" in report
    [batch] = host.batches
    assert batch.agent is agent
    control = batch.control_for(batch.clones[0])
    assert control.depth == 1
    assert control.worker == batch.clones[0].id
    # Not rolled up: only the parent's own passes.
    assert result.usage.requests == 2


async def test_inline_host_is_the_default():
    assert isinstance(CloneToolSource().host, InlineCloneHost)
