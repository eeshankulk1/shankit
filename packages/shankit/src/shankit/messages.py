"""Internal message representation.

Messages use the Anthropic content-block shape (design §3.1: one format at the
boundary). Model clients for other providers convert internally; the agent
loop and tool seam only ever see these types.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

__all__ = [
    "ContentBlock",
    "ImageBlock",
    "Message",
    "ProviderBlock",
    "ReasoningBlock",
    "TextBlock",
    "ToolResultBlock",
    "ToolResultContent",
    "ToolUseBlock",
    "assistant_text",
    "coerce_message",
    "content_text",
    "strip_reasoning",
    "user_message",
]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image in a tool result (e.g. a screenshot), base64-encoded."""

    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data: str


class ProviderBlock(BaseModel):
    """A provider-specific block in a tool result, carried opaquely.

    ``provider`` names the model client that understands it; that client
    sends ``data`` verbatim (e.g. Anthropic's ``browser_state``). Every other
    client sends ``text`` in its place, or nothing when ``text`` is empty -
    so a tool can return one result that reads well on any provider.
    """

    type: Literal["provider"] = "provider"
    provider: str
    data: dict[str, Any] = Field(default_factory=dict)
    text: str = ""


ToolResultContent = Annotated[
    Union[TextBlock, ImageBlock, ProviderBlock],
    Field(discriminator="type"),
]


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    #: The native toolset this call belongs to, as the provider named it
    #: (e.g. Anthropic's ``toolset_name: "browser"``); round-tripped on the
    #: call and its result.
    toolset: Optional[str] = None


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    #: Text, or a list of text / image / provider blocks.
    content: Union[str, list[ToolResultContent]]
    is_error: bool = False
    #: Echoes the call's ``toolset`` (providers require it back).
    toolset: Optional[str] = None


class ReasoningBlock(BaseModel):
    """A model's reasoning (e.g. Anthropic thinking), carried opaquely.

    Providers that return reasoning require it back unmodified on the next
    request of the same run for multi-step tool use to keep working (and to
    keep reasoning quality). ``provider`` names the model client that
    produced it; ``data`` is that provider's raw block, re-sent verbatim by
    the same provider's client and dropped by every other client. ``text``
    is any human-readable summary the provider returned (often empty).
    """

    type: Literal["reasoning"] = "reasoning"
    provider: str
    data: dict[str, Any] = Field(default_factory=dict)
    text: str = ""


ContentBlock = Annotated[
    Union[TextBlock, ToolUseBlock, ToolResultBlock, ReasoningBlock],
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


def content_text(content: Union[str, list[Any]]) -> str:
    """A tool result's content as plain text: text blocks as-is, a provider
    block's text fallback, and ``[image]`` for each image. For logs,
    trajectories, and providers that take only text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append("[image]")
        elif isinstance(block, ProviderBlock) and block.text:
            parts.append(block.text)
    return "\n".join(parts)


def assistant_text(message: Message) -> str:
    """Concatenated text content of a message."""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


def strip_reasoning(messages: list[Message]) -> list[Message]:
    """``messages`` without reasoning blocks (messages left empty are
    dropped). Use when replaying a transcript into a *different* run:
    providers bind reasoning to the exact conversation that produced it, so
    reasoning from an earlier run is at best ignored and at worst rejected
    once the system prompt or history around it differs."""
    out: list[Message] = []
    for message in messages:
        blocks = [b for b in message.content if not isinstance(b, ReasoningBlock)]
        if blocks:
            out.append(message.model_copy(update={"content": blocks}))
    return out
