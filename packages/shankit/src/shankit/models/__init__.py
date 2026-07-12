from .base import (
    ForcedTool,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
    default_stream_from_complete,
    map_sdk_error,
    model_error_for_status,
    wrap_sdk_errors,
)
from .registry import register_provider, resolve_model, shutdown

__all__ = [
    "ForcedTool",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseComplete",
    "ModelStreamEvent",
    "ModelTextDelta",
    "default_stream_from_complete",
    "map_sdk_error",
    "model_error_for_status",
    "register_provider",
    "resolve_model",
    "shutdown",
    "wrap_sdk_errors",
]


def __getattr__(name: str):  # lazy: keep optional SDKs out of the import path
    if name == "AnthropicModel":
        from .anthropic import AnthropicModel

        return AnthropicModel
    if name == "OpenAIModel":
        from .openai import OpenAIModel

        return OpenAIModel
    raise AttributeError(name)
