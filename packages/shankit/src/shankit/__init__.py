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
from ._calls import ToolCallContext, current_tool_call
from .agent import Agent, Prompt, RunResult, ToolCallRecord
from .control import Budget, RunControl
from .durability import (
    Checkpoint,
    Checkpointer,
    InMemoryCheckpointer,
    InterruptInfo,
    SqliteCheckpointer,
)
from .events import (
    AgentEvent,
    Artifact,
    ArtifactEvent,
    DoneEvent,
    ErrorEvent,
    Source,
    SourceEvent,
    StepEvent,
    TextDeltaEvent,
    UsageEvent,
    WorkerEvent,
)
from .exceptions import (
    AgentFileError,
    MaxIterationsError,
    ModelError,
    OutputValidationError,
    PromptVariableError,
    RunTimeoutError,
    ShankitError,
    ToolError,
    ToolNotFoundError,
)
from .files import Registry, default_registry, load_agent, load_agents, register
from .messages import (
    ImageBlock,
    Message,
    PromptBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    coerce_message,
    content_text,
    strip_reasoning,
)
from .models import ModelClient, ModelRequest, ModelResponse, register_provider, resolve_model
from .models.base import Reasoning
from .observe import StepDescriber, StepInfo, default_step_describer
from .sse import format_sse, sse_stream
from .tools import (
    AgentDelegate,
    CloneBatch,
    CloneHost,
    CloneOutcome,
    CloneSpec,
    CloneToolSource,
    CompositeToolSource,
    Delegate,
    DelegateToolSource,
    FunctionTool,
    FunctionToolSource,
    InlineCloneHost,
    MCPToolSource,
    ToolDef,
    ToolResult,
    Toolset,
    ToolSource,
    run_clone,
    tool,
)
from .usage import Usage
from .workspace import (
    ExecResult,
    LocalWorkspace,
    Workspace,
    WorkspaceTools,
    describe_workspace_step,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentDelegate",
    "AgentEvent",
    "AgentFileError",
    "Artifact",
    "ArtifactEvent",
    "Budget",
    "Checkpoint",
    "Checkpointer",
    "CloneBatch",
    "CloneHost",
    "CloneOutcome",
    "CloneSpec",
    "CloneToolSource",
    "CompositeToolSource",
    "Delegate",
    "DelegateToolSource",
    "DoneEvent",
    "ErrorEvent",
    "ExecResult",
    "FunctionTool",
    "FunctionToolSource",
    "ImageBlock",
    "InMemoryCheckpointer",
    "InlineCloneHost",
    "InterruptInfo",
    "LocalWorkspace",
    "MCPToolSource",
    "MaxIterationsError",
    "Message",
    "ModelClient",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "OutputValidationError",
    "Prompt",
    "PromptBlock",
    "PromptVariableError",
    "ProviderBlock",
    "Reasoning",
    "ReasoningBlock",
    "Registry",
    "RunControl",
    "RunResult",
    "RunTimeoutError",
    "ShankitError",
    "Source",
    "SourceEvent",
    "SqliteCheckpointer",
    "StepDescriber",
    "StepEvent",
    "StepInfo",
    "TextBlock",
    "TextDeltaEvent",
    "ToolCallContext",
    "ToolCallRecord",
    "ToolDef",
    "ToolError",
    "ToolNotFoundError",
    "ToolResult",
    "ToolResultBlock",
    "ToolSource",
    "ToolUseBlock",
    "Toolset",
    "Usage",
    "UsageEvent",
    "WorkerEvent",
    "Workspace",
    "WorkspaceTools",
    "__version__",
    "coerce_message",
    "content_text",
    "current_tool_call",
    "default_registry",
    "default_step_describer",
    "describe_workspace_step",
    "evals",
    "format_sse",
    "load_agent",
    "load_agents",
    "register",
    "register_provider",
    "resolve_model",
    "run_clone",
    "sse_stream",
    "strip_reasoning",
    "tool",
]
