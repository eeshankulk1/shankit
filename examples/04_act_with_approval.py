"""The act-with-approval slice (design §9's validation path).

A consequential action routed as gather -> draft -> pause for human
approval -> act. Deterministic code routing + shared state + a durable
checkpoint at every router boundary; the human approval is just an
Interrupt at one of those boundaries.

EXPERIMENTAL API (shankit.experimental.graph).

Requires: pip install shankit[anthropic]  and  ANTHROPIC_API_KEY set.
"""

import asyncio

from shankit import Agent, SqliteCheckpointer, tool
from shankit.experimental.graph import Interrupt, Network, agent_step


@tool
def search_thread(topic: str) -> str:
    """Search recent email threads about a topic."""
    return "Bob asked for the revised contract by Thursday; you owe him a reply."


researcher = Agent(
    name="researcher",
    model="anthropic:claude-sonnet-4-5",
    instructions="Gather the facts needed to reply. Report findings tersely.",
    tools=[search_thread],
)

writer = Agent(
    name="writer",
    model="anthropic:claude-sonnet-4-5",
    instructions="Draft a short, professional email. Output only the draft.",
)


async def send_email(state: dict, context) -> dict:
    print(f"\n>>> SENDING EMAIL:\n{state['draft']}\n")
    return {"sent": True}


def router(state: dict):
    if "notes" not in state:
        return "gather"
    if "draft" not in state:
        return "draft"
    if "approval" not in state:
        # Every router decision is a checkpoint boundary, so pausing for a
        # human here is durable by construction.
        return Interrupt(reason="Approve sending this email?", payload=state["draft"])
    if state["approval"] and "sent" not in state:
        return "act"
    return None  # stop


network = Network(
    name="act-with-approval",
    steps={
        "gather": agent_step(researcher, prompt="Gather context for: {task}", output_key="notes"),
        "draft": agent_step(
            writer, prompt="Write the email for: {task}\n\nContext:\n{notes}", output_key="draft"
        ),
        "act": send_email,
    },
    router=router,
    checkpointer=SqliteCheckpointer("approval_threads.db"),
)


async def main() -> None:
    thread_id = "demo-thread-1"
    result = await network.run({"task": "reply to Bob about the contract"}, thread_id=thread_id)

    if result.status == "interrupted":
        print(f"PAUSED: {result.interrupt.reason}")
        print(f"--- draft ---\n{result.interrupt.payload}\n-------------")
        answer = input("approve? [y/N] ").strip().lower() == "y"
        # This could equally happen hours later, from another process — the
        # thread lives in SQLite, not in memory.
        result = await network.resume(thread_id, value=answer)

    print(f"finished: {result.status}, steps run: {result.steps_run}")


if __name__ == "__main__":
    asyncio.run(main())
