"""shankit — an AI agent framework that takes the integration seam as
seriously as the agent loop itself, without imposing how you authenticate.

Public surface (design §11):

- :class:`Agent` with two run modes over one loop: ``run()`` (structured)
  and ``stream()`` (typed event stream).
- The tool seam: :class:`ToolSource` and its shipped implementations —
  ``@tool`` local functions, :class:`MCPToolSource`, and
  :class:`CompositeToolSource`.
- The opaque per-run context: your plain data, threaded to tools and
  dynamic instructions; the framework never reads inside it.
- Provider-neutral model clients (Anthropic, OpenAI at launch).
- File-first definitions: :func:`load_agent` / :func:`load_agents`.
- Durability: :class:`Checkpointer` with in-memory and SQLite stores.
- Evaluation: :mod:`shankit.evals`.
- Experimental graph/network: :mod:`shankit.experimental.graph`.
"""

from .agent import Agent, RunResult, ToolCallRecord
from .durability import Checkpoint, Checkpointer, InMemoryCheckpointer, SqliteCheckpointer
from .events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    Source,
    SourceEvent,
    StepEvent,
    TextDeltaEvent,
    UsageEvent,
)
from .exceptions import (
    AgentFileError,
    MaxIterationsError,
    ModelError,
    OutputValidationError,
    PromptVariableError,
    ShankitError,
    ToolError,
    ToolNotFoundError,
)
from .files import Registry, default_registry, load_agent, load_agents, register
from .messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, coerce_message
from .models import ModelClient, ModelRequest, ModelResponse, register_provider, resolve_model
from .observe import StepDescriber, StepInfo, default_step_describer
from .sse import format_sse, sse_stream
from .tools import (
    CompositeToolSource,
    FunctionTool,
    FunctionToolSource,
    MCPToolSource,
    ToolDef,
    ToolResult,
    ToolSource,
    tool,
)
from .usage import Usage

__version__ = "0.1.0"

__all__ = [
    # agent
    "Agent",
    "RunResult",
    "ToolCallRecord",
    # tool seam
    "tool",
    "ToolDef",
    "ToolResult",
    "ToolSource",
    "FunctionTool",
    "FunctionToolSource",
    "CompositeToolSource",
    "MCPToolSource",
    # events
    "AgentEvent",
    "TextDeltaEvent",
    "StepEvent",
    "SourceEvent",
    "UsageEvent",
    "ErrorEvent",
    "DoneEvent",
    "Source",
    "Usage",
    # observability
    "StepInfo",
    "StepDescriber",
    "default_step_describer",
    # models
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "resolve_model",
    "register_provider",
    # messages (for ModelClient implementers)
    "Message",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "coerce_message",
    # files
    "load_agent",
    "load_agents",
    "Registry",
    "default_registry",
    "register",
    # durability
    "Checkpointer",
    "Checkpoint",
    "InMemoryCheckpointer",
    "SqliteCheckpointer",
    # sse
    "format_sse",
    "sse_stream",
    # errors
    "ShankitError",
    "ToolError",
    "ToolNotFoundError",
    "ModelError",
    "OutputValidationError",
    "MaxIterationsError",
    "AgentFileError",
    "PromptVariableError",
    "__version__",
]
