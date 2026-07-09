"""Agents as tools: handing sub-agents to a parent agent IS orchestration.

The parent's model decides when to consult each specialist; failures are
sanitized, sources propagate, and sub-agent token usage folds into the
parent run's accounting — all through the ordinary tool seam.

Requires: pip install shankit[anthropic]  and  ANTHROPIC_API_KEY set.
"""

import asyncio

from shankit import Agent, ToolResult, tool
from shankit.events import Source


@tool
def search_docs(query: str) -> ToolResult:
    """Search the internal docs."""
    return ToolResult(
        content=f"Docs matching {query!r}: 'Deploys run at 2pm UTC daily.'",
        sources=[Source(title="Runbook: deploys", url="https://internal/runbook#deploys")],
    )


@tool
def query_metrics(service: str) -> str:
    """Get error-rate metrics for a service."""
    return f"{service}: error rate 0.02% (last 24h), p99 latency 210ms"


docs_agent = Agent(
    name="docs",
    description="Answers questions from internal documentation.",
    model="anthropic:claude-sonnet-4-5",
    instructions="Answer strictly from the docs search results.",
    tools=[search_docs],
)

metrics_agent = Agent(
    name="metrics",
    description="Reports service health metrics.",
    model="anthropic:claude-sonnet-4-5",
    instructions="Report metrics precisely; no speculation.",
    tools=[query_metrics],
)

oncall = Agent(
    name="oncall-orchestrator",
    model="anthropic:claude-sonnet-4-5",
    instructions="You coordinate specialists to answer on-call questions.",
    tools=[docs_agent.as_tool(), metrics_agent.as_tool()],
)


async def main() -> None:
    async for event in oncall.stream(
        "Is the checkout service healthy, and when is the next deploy?"
    ):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "step":
            print(f"\n  [{event.status}] {event.title}")
        elif event.type == "source":
            print(f"\n  source: {event.source.title}")
        elif event.type == "error":
            print(f"\n[error] {event.message}")
        elif event.type == "done":
            print(f"\n(total usage across all agents: {event.usage})")


if __name__ == "__main__":
    asyncio.run(main())
