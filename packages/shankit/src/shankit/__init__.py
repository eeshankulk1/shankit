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

# Documented as reachable off the package (`shankit.evals`); import it so
# that holds without a separate import. It is dependency-light. Placed after
# the core imports because evals itself imports from shankit.agent.
from . import evals
from .agent import Agent, RunResult, ToolCallRecord
from .durability import (
    Checkpoint,
    Checkpointer,
    InMemoryCheckpointer,
    InterruptInfo,
    SqliteCheckpointer,
)
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
    "Agent",
    "AgentEvent",
    "AgentFileError",
    "Checkpoint",
    "Checkpointer",
    "CompositeToolSource",
    "DoneEvent",
    "ErrorEvent",
    "FunctionTool",
    "FunctionToolSource",
    "InMemoryCheckpointer",
    "InterruptInfo",
    "MCPToolSource",
    "MaxIterationsError",
    "Message",
    "ModelClient",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "OutputValidationError",
    "PromptVariableError",
    "Registry",
    "RunResult",
    "ShankitError",
    "Source",
    "SourceEvent",
    "SqliteCheckpointer",
    "StepDescriber",
    "StepEvent",
    "StepInfo",
    "TextBlock",
    "TextDeltaEvent",
    "ToolCallRecord",
    "ToolDef",
    "ToolError",
    "ToolNotFoundError",
    "ToolResult",
    "ToolResultBlock",
    "ToolSource",
    "ToolUseBlock",
    "Usage",
    "UsageEvent",
    "__version__",
    "coerce_message",
    "default_registry",
    "default_step_describer",
    "evals",
    "format_sse",
    "load_agent",
    "load_agents",
    "register",
    "register_provider",
    "resolve_model",
    "sse_stream",
    "tool",
]
