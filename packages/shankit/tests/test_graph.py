import pytest
from conftest import FakeModel, text_response
from shankit import Agent, ShankitError
from shankit.durability import InMemoryCheckpointer
from shankit.experimental.graph import Interrupt, Network, agent_step


def draft_step(state, context):
    return {"draft": f"Draft for: {state['task']}"}


async def act_step(state, context):
    return {"sent": True}


def approval_router(state):
    if "draft" not in state:
        return "draft"
    if "approval" not in state:
        return Interrupt(reason="Approve sending this draft?", payload=state["draft"])
    if state["approval"] and "sent" not in state:
        return "act"
    return None


def approval_network(checkpointer=None):
    return Network(
        name="act-with-approval",
        steps={"draft": draft_step, "act": act_step},
        router=approval_router,
        checkpointer=checkpointer,
    )


async def test_router_loop_runs_to_done():
    async def router(state):  # async routers are fine
        return None if "draft" in state else "draft"

    net = Network(name="n", steps={"draft": draft_step}, router=router)
    result = await net.run({"task": "email bob"})
    assert result.status == "done"
    assert result.state["draft"] == "Draft for: email bob"
    assert result.steps_run == ["draft"]


async def test_interrupt_and_resume_with_checkpointer():
    checkpointer = InMemoryCheckpointer()
    net = approval_network(checkpointer)

    result = await net.run({"task": "email bob"}, thread_id="t1")
    assert result.status == "interrupted"
    assert result.interrupt.reason == "Approve sending this draft?"
    assert result.interrupt.payload == "Draft for: email bob"

    saved = await checkpointer.load("t1")
    assert saved.status == "interrupted"

    resumed = await net.resume("t1", value=True)
    assert resumed.status == "done"
    assert resumed.state["sent"] is True
    assert resumed.steps_run == ["draft", "act"]
    assert (await checkpointer.load("t1")).status == "done"


async def test_rejection_path():
    checkpointer = InMemoryCheckpointer()
    net = approval_network(checkpointer)
    await net.run({"task": "email bob"}, thread_id="t2")
    resumed = await net.resume("t2", value=False)
    assert resumed.status == "done"
    assert "sent" not in resumed.state


async def test_checkpointer_requires_thread_id():
    net = approval_network(InMemoryCheckpointer())
    with pytest.raises(ShankitError, match="thread_id"):
        await net.run({"task": "x"})


async def test_resume_requires_interrupted_thread():
    checkpointer = InMemoryCheckpointer()
    net = Network(
        name="n", steps={"draft": draft_step}, router=lambda s: None, checkpointer=checkpointer
    )
    await net.run({"task": "x"}, thread_id="t3")  # completes immediately
    with pytest.raises(ShankitError, match="not awaiting input"):
        await net.resume("t3", value=True)
    with pytest.raises(ShankitError, match="No checkpoint"):
        await net.resume("missing", value=True)


async def test_unknown_step_and_max_steps():
    net = Network(name="n", steps={"a": draft_step}, router=lambda s: "ghost")
    with pytest.raises(ShankitError, match="unknown step"):
        await net.run({"task": "x"})

    looping = Network(name="n", steps={"a": lambda s, c: None}, router=lambda s: "a", max_steps=3)
    with pytest.raises(ShankitError, match="max_steps"):
        await looping.run({})


async def test_bare_agent_step_rejected():
    agent = Agent(name="a", model="fake", model_client=FakeModel([]))
    with pytest.raises(TypeError, match="agent_step"):
        Network(name="n", steps={"a": agent}, router=lambda s: None)


async def test_agent_step_adapts_agent():
    agent = Agent(name="writer", model="fake", model_client=FakeModel([text_response("the draft")]))
    step = agent_step(agent, prompt="Write a draft about {task}.", output_key="draft")
    net = Network(
        name="n",
        steps={"draft": step},
        router=lambda s: "draft" if "draft" not in s else None,
    )
    result = await net.run({"task": "cats"})
    assert result.state["draft"] == "the draft"


async def test_agent_step_missing_placeholder():
    agent = Agent(name="writer", model="fake", model_client=FakeModel([text_response("x")]))
    step = agent_step(agent, prompt="{missing}", output_key="out")
    net = Network(name="n", steps={"s": step}, router=lambda s: "s" if "out" not in s else None)
    with pytest.raises(ShankitError, match="missing"):
        await net.run({})


async def test_nested_network_shares_state():
    inner = Network(
        name="inner",
        steps={"draft": draft_step},
        router=lambda s: "draft" if "draft" not in s else None,
    )
    outer = Network(
        name="outer",
        steps={"sub": inner, "act": act_step},
        router=lambda s: ("sub" if "draft" not in s else ("act" if "sent" not in s else None)),
    )
    result = await outer.run({"task": "x"})
    assert result.state["sent"] is True
    assert result.state["draft"] == "Draft for: x"


async def test_network_as_tool():
    net = Network(
        name="drafting",
        description="Draft a message deterministically.",
        steps={"draft": draft_step},
        router=lambda s: "draft" if "draft" not in s else None,
    )
    source = net.as_tool(output_key="draft")
    defs = await source.list_tools()
    assert defs[0].name == "drafting"
    result = await source.execute("drafting", {"task": "email bob"})
    assert result.content == "Draft for: email bob"


async def test_network_as_tool_interrupt_reported():
    net = approval_network()
    source = net.as_tool()
    result = await source.execute("act-with-approval", {"task": "x"})
    assert result.is_error
    assert "paused for human input" in result.content


# ------------------------------------------------------------ crash recovery


async def test_recover_after_crash_between_steps():
    """A 'running' checkpoint with no in-flight step re-enters cleanly."""
    from shankit.durability import Checkpoint

    checkpointer = InMemoryCheckpointer()
    net = approval_network(checkpointer)
    # Simulate a process that died after "draft" completed but before the
    # router ran again: exactly what the post-step checkpoint records.
    await checkpointer.save(
        "t-crash",
        Checkpoint(
            state={"task": "email bob", "draft": "Draft for: email bob", "approval": True},
            status="running",
            steps_run=["draft"],
        ),
    )
    result = await net.recover("t-crash")
    assert result.status == "done"
    assert result.state["sent"] is True
    assert result.steps_run == ["draft", "act"]


async def test_step_failure_leaves_in_flight_marker_and_recover_retries():
    checkpointer = InMemoryCheckpointer()
    attempts = {"count": 0}

    async def flaky_act(state, context):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient provider outage")
        return {"sent": True}

    net = Network(
        name="flaky",
        steps={"act": flaky_act},
        router=lambda s: "act" if "sent" not in s else None,
        checkpointer=checkpointer,
    )
    with pytest.raises(RuntimeError, match="transient"):
        await net.run({"task": "x"}, thread_id="t-flaky")

    saved = await checkpointer.load("t-flaky")
    assert saved.status == "running"
    assert saved.in_flight == "act"

    # Ambiguous by default: the failed step may have had side effects.
    with pytest.raises(ShankitError, match="in ?flight|idempotent"):
        await net.recover("t-flaky")

    result = await net.recover("t-flaky", retry_in_flight=True)
    assert result.status == "done"
    assert result.state["sent"] is True
    assert attempts["count"] == 2


async def test_in_flight_cleared_after_successful_step():
    checkpointer = InMemoryCheckpointer()
    net = approval_network(checkpointer)
    await net.run({"task": "x"}, thread_id="t-clear")  # pauses at the interrupt
    saved = await checkpointer.load("t-clear")
    assert saved.in_flight is None
    assert saved.status == "interrupted"


async def test_recover_wrong_states():
    checkpointer = InMemoryCheckpointer()
    net = approval_network(checkpointer)

    with pytest.raises(ShankitError, match="No checkpoint"):
        await net.recover("missing")

    await net.run({"task": "x"}, thread_id="t-int")  # interrupted
    with pytest.raises(ShankitError, match="use resume"):
        await net.recover("t-int")

    done = await net.resume("t-int", value=False)
    assert done.status == "done"
    with pytest.raises(ShankitError, match="already finished"):
        await net.recover("t-int")

    no_cp = approval_network()
    with pytest.raises(ShankitError, match="no checkpointer"):
        await no_cp.recover("t-int")
