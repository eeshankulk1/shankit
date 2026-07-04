"""Internal message representation.

Messages use the Anthropic content-block shape (design §3.1: one format at the
boundary). Model clients for other providers convert internally; the agent
loop and tool seam only ever see these types.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

__all__ = [
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    "Message",
    "user_message",
    "assistant_text",
]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    Union[TextBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


def user_message(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def assistant_text(message: Message) -> str:
    """Concatenated text content of a message."""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))
