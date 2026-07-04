# shankit

An open-source AI agent framework that takes the **integration seam** — the
boundary between an agent and the real, authenticated external tools it acts
on — as seriously as the agent loop itself, **without imposing how you
authenticate**.

```python
from shankit import Agent, tool
from pydantic import BaseModel

@tool
def search_orders(ctx, query: str) -> str:
    """Search the order database."""
    return my_db.search(query, user_id=ctx["user_id"])  # your auth, not ours

class Summary(BaseModel):
    headline: str
    open_orders: int

agent = Agent(
    name="orders",
    model="anthropic:claude-sonnet-4-5",
    instructions="You help users with their orders.",
    tools=[search_orders],
)

# Structured: a typed, validated deliverable
result = await agent.run("Summarize my open orders", context={"user_id": "u1"}, output_type=Summary)

# Streaming: a typed event stream (text deltas, steps, sources, usage, done)
async for event in agent.stream("What happened to order 42?", context={"user_id": "u1"}):
    ...
```

See the [repository](https://github.com/eeshankulk1/shankit) for the full
documentation, the design decision record, file-first agent definitions,
durability, evals, the experimental graph escape hatch, the connector layer
(`shankit-connectors`), and the TypeScript consumer SDK.
