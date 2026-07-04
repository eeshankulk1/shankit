"""Serving the event stream over SSE for the TypeScript consumer SDK.

Requires: pip install shankit[anthropic] fastapi uvicorn
Run:      uvicorn examples.06_sse_server:app --reload
Consume:  see examples/07_ts_consumer.ts (@shankit/client)
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shankit import Agent, sse_stream, tool


@tool
def lookup_status(order_id: str) -> str:
    """Look up an order's status."""
    return f"Order {order_id} is out for delivery."


agent = Agent(
    name="support",
    model="anthropic:claude-sonnet-4-5",
    instructions="You are a friendly support agent.",
    tools=[lookup_status],
)

app = FastAPI()


class StreamRequest(BaseModel):
    prompt: str
    user_id: str


@app.post("/agents/support/stream")
async def stream(body: StreamRequest) -> StreamingResponse:
    events = agent.stream(body.prompt, context={"user_id": body.user_id})
    return StreamingResponse(sse_stream(events), media_type="text/event-stream")
