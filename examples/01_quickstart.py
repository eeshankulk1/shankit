"""Quickstart: one agent, both run modes.

Requires: pip install shankit[anthropic]  and  ANTHROPIC_API_KEY set.
Run: python examples/01_quickstart.py
"""

import asyncio

from pydantic import BaseModel, Field
from shankit import Agent, ToolError, tool

# --- capability: tools are code -------------------------------------------

FAKE_ORDERS = {
    "u1": [
        {"id": "A-100", "item": "mechanical keyboard", "status": "shipped"},
        {"id": "A-101", "item": "usb hub", "status": "processing"},
    ]
}


@tool
def list_orders(ctx) -> list:
    """List the acting user's orders."""
    # `ctx` is the opaque per-run context — the framework never reads inside
    # it. Whose orders these are is *your* concern, not the framework's.
    return FAKE_ORDERS.get(ctx["user_id"], [])


@tool
def cancel_order(ctx, order_id: str) -> str:
    """Cancel an order that has not shipped yet."""
    orders = FAKE_ORDERS.get(ctx["user_id"], [])
    for order in orders:
        if order["id"] == order_id:
            if order["status"] == "shipped":
                raise ToolError(f"Order {order_id} already shipped and cannot be cancelled.")
            order["status"] = "cancelled"
            return f"Cancelled {order_id}."
    raise ToolError(f"No order {order_id} for this user.")


# --- the agent -------------------------------------------------------------

agent = Agent(
    name="orders-assistant",
    model="anthropic:claude-sonnet-4-5",
    instructions="You help users manage their orders. Be concise.",
    tools=[list_orders, cancel_order],
)


class OrdersSummary(BaseModel):
    open_orders: int = Field(description="Orders not yet delivered or cancelled")
    headline: str


async def main() -> None:
    context = {"user_id": "u1"}

    # Structured run: a typed, validated deliverable (the caller chooses).
    result = await agent.run("Summarize my orders.", context=context, output_type=OrdersSummary)
    print("structured:", result.output)
    print("tools used:", [record.tool for record in result.trajectory])
    print("usage:", result.usage)

    # Streaming run: the typed event stream.
    async for event in agent.stream("Cancel my usb hub order.", context=context):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "step":
            print(f"\n[{event.status}] {event.title}")
        elif event.type == "done":
            print(f"\n(done — {event.usage.input_tokens} in / {event.usage.output_tokens} out)")


if __name__ == "__main__":
    asyncio.run(main())
