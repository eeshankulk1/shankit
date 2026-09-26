"""The workspace seam and the tools that work over it (design §14.15)."""

from .base import ExecResult, Workspace, WorkspaceSpec, resolve_workspace
from .local import LocalWorkspace
from .tools import (
    WORKSPACE_TOOL_NAMES,
    WorkspaceTools,
    describe_workspace_step,
    format_exec_result,
)

__all__ = [
    "WORKSPACE_TOOL_NAMES",
    "ExecResult",
    "LocalWorkspace",
    "Workspace",
    "WorkspaceSpec",
    "WorkspaceTools",
    "describe_workspace_step",
    "format_exec_result",
    "resolve_workspace",
]
