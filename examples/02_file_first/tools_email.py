"""Capability lives in code; the .md file references it by import path."""

from pydantic import BaseModel
from shankit import tool

INBOX = [
    {"id": "m1", "from": "boss@corp.com", "subject": "Need the Q3 numbers by Friday"},
    {"id": "m2", "from": "news@letter.io", "subject": "Weekly digest"},
]


class TriageItem(BaseModel):
    message_id: str
    priority: str
    next_action: str


class TriagePlan(BaseModel):
    items: list[TriageItem]
    summary: str


@tool
def search_email(query: str = "") -> list:
    """Search the inbox; empty query lists everything."""
    return [m for m in INBOX if query.lower() in (m["subject"] + m["from"]).lower()]


@tool
def read_email(message_id: str) -> str:
    """Read one message body."""
    return f"(body of {message_id})"
