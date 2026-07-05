from .base import (
    ForcedTool,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
)
from .registry import register_provider, resolve_model

__all__ = [
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "ModelTextDelta",
    "ModelResponseComplete",
    "ModelStreamEvent",
    "ForcedTool",
    "register_provider",
    "resolve_model",
]


def __getattr__(name: str):  # lazy: keep optional SDKs out of the import path
    if name == "AnthropicModel":
        from .anthropic import AnthropicModel

        return AnthropicModel
    if name == "OpenAIModel":
        from .openai import OpenAIModel

        return OpenAIModel
    raise AttributeError(name)
