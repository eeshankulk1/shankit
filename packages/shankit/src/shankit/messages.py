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
    "coerce_message",
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


def coerce_message(item: Message | dict[str, Any]) -> Message:
    """Accept a :class:`Message` or a plain ``{"role", "content"}`` dict.

    A dict's ``content`` may be a string (wrapped in a text block) or a list
    of content blocks. This is the leniency layer for callers that keep chat
    history in plain-dict form.
    """
    if isinstance(item, Message):
        return item
    if isinstance(item, dict):
        content = item.get("content")
        if isinstance(content, str):
            return Message(role=item["role"], content=[TextBlock(text=content)])
        return Message.model_validate(item)
    raise TypeError(f"Cannot coerce {type(item).__name__} into a Message.")


def assistant_text(message: Message) -> str:
    """Concatenated text content of a message."""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))
