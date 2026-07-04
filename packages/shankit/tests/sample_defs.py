"""Importable objects for the file-first definition tests."""

from pydantic import BaseModel
from shankit import tool


class Triage(BaseModel):
    priority: str
    reply: str


@tool
def search_email(query: str) -> str:
    """Search the mailbox."""
    return f"results for {query}"


@tool
def archive(message_id: str) -> str:
    """Archive a message."""
    return "archived"


toolbox = [search_email, archive]
