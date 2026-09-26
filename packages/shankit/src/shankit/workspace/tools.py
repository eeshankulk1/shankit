"""``Bash``/``Read``/``Write``/``Edit``: the coding-agent tool set over a workspace.

The names and shapes are the ones coding agents converge on (Claude Code,
OpenCode, Codex), as plain JSON-schema tools, so any provider can call them
and models trained on agentic coding use them well. No Glob/Grep: ``rg`` and
``ls`` through ``Bash`` cover both.

Output shapes:

- ``Bash``: stdout, then stderr (marked), then a status line with the exit
  code and duration. A non-zero exit is information, not a tool failure
  (``grep`` with no match exits 1); only a timeout or a failure to run is
  ``is_error``. Oversized output is left to the loop, which spills it to a
  file when the agent has a workspace (``Agent(spill_threshold_chars=)``).
- ``Read``: ``cat -n``-style numbered lines, windowed by ``offset``/``limit``.
- ``Write``/``Edit``: a one-line confirmation.

Subclass hooks: :meth:`WorkspaceTools.after_tool` sees every call's result
(and the raw :class:`ExecResult` for ``Bash``) — the place to sync files
out of the workspace, record a work log, or attach artifacts.
"""

from __future__ import annotations

import inspect
import logging
import posixpath
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Optional, Union

from ..exceptions import ToolError, ToolNotFoundError
from ..observe import StepInfo
from ..tools.base import ToolDef, ToolResult, ToolSource
from .base import ExecResult, Workspace, WorkspaceSpec, resolve_workspace

__all__ = ["WORKSPACE_TOOL_NAMES", "WorkspaceTools", "describe_workspace_step"]

logger = logging.getLogger("shankit")

WORKSPACE_TOOL_NAMES = ("Bash", "Read", "Write", "Edit")

# Read windowing, matching the common coding-agent defaults.
_READ_DEFAULT_LIMIT = 2000
_READ_MAX_LINE_CHARS = 2000
_READ_MAX_CHARS = 100_000

EnvSpec = Union[
    Mapping[str, str],
    Callable[[Any], Union[Mapping[str, str], Awaitable[Mapping[str, str]]]],
]


class WorkspaceTools(ToolSource):
    """File and shell tools over a workspace.

    Args:
        workspace: A :class:`Workspace`, or a (sync/async) function of the
            per-run context returning one — the same spec ``Agent(workspace=)``
            takes, so both resolve the same workspace for a run.
        tools: Which of ``Bash``/``Read``/``Write``/``Edit`` to offer.
            ``Bash`` is dropped automatically for a workspace that can't
            run commands.
        default_timeout_s / max_timeout_s: The ``Bash`` timeout when the
            model doesn't pass one, and the ceiling on what it may ask for.
        env: Extra environment for every command — a mapping or a
            function of the per-run context (e.g. a per-run credential the
            command needs; passed per exec, never written to disk).
    """

    def __init__(
        self,
        workspace: WorkspaceSpec,
        *,
        tools: Sequence[str] = WORKSPACE_TOOL_NAMES,
        default_timeout_s: float = 60.0,
        max_timeout_s: float = 300.0,
        env: Optional[EnvSpec] = None,
    ) -> None:
        unknown = set(tools) - set(WORKSPACE_TOOL_NAMES)
        if unknown:
            raise ValueError(f"Unknown workspace tools {sorted(unknown)}")
        self.workspace = workspace
        self.tool_names = tuple(tools)
        self.default_timeout_s = default_timeout_s
        self.max_timeout_s = max_timeout_s
        self.env = env

    # ----------------------------------------------------------- the seam

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        ws = await resolve_workspace(self.workspace, context)
        if ws is None:
            return []
        defs = {
            "Bash": self._bash_def(ws),
            "Read": _READ_DEF,
            "Write": _WRITE_DEF,
            "Edit": _EDIT_DEF,
        }
        return [defs[name] for name in self.tool_names if name != "Bash" or ws.can_exec]

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        if name not in self.tool_names:
            raise ToolNotFoundError(name)
        ws = await resolve_workspace(self.workspace, context)
        if ws is None:
            raise ToolError("No workspace is available for this run.")
        exec_result: Optional[ExecResult] = None
        if name == "Bash":
            if not ws.can_exec:
                raise ToolNotFoundError(name)
            exec_result = await self.run_command(ws, arguments, context)
            result = ToolResult(
                content=format_exec_result(exec_result),
                is_error=exec_result.timed_out,
            )
        elif name == "Read":
            result = ToolResult(content=await _read(ws, arguments))
        elif name == "Write":
            result = ToolResult(content=await _write(ws, arguments))
        else:
            result = ToolResult(content=await _edit(ws, arguments))
        return await self.after_tool(name, arguments, result, context, ws, exec_result)

    # -------------------------------------------------------------- hooks

    async def after_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        context: Any,
        workspace: Workspace,
        exec_result: Optional[ExecResult],
    ) -> ToolResult:
        """Hook: post-process a completed call. Default: return ``result``."""
        return result

    async def run_command(
        self, workspace: Workspace, arguments: dict[str, Any], context: Any
    ) -> ExecResult:
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise ToolError("Bash requires a non-empty `command`.")
        timeout = _clamp_timeout(
            arguments.get("timeout_s"), self.default_timeout_s, self.max_timeout_s
        )
        env = await self._resolve_env(context)
        return await workspace.exec(command, timeout=timeout, env=env)

    async def _resolve_env(self, context: Any) -> Optional[Mapping[str, str]]:
        if self.env is None or isinstance(self.env, Mapping):
            return self.env
        resolved = self.env(context)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        return resolved

    def _bash_def(self, ws: Workspace) -> ToolDef:
        return ToolDef(
            name="Bash",
            description=(
                "Run a bash command in your workspace and get its output. Use it "
                "for scripts (python3), pipes, jq/rg/ls, and any CLI installed in "
                f"the workspace. Commands start in the home directory ({ws.home}); "
                "each call is a fresh shell, so `cd` doesn't persist between calls. "
                "Save big outputs to files and inspect them with head/rg/jq/python "
                "instead of printing everything. Default timeout "
                f"{int(self.default_timeout_s)}s, at most {int(self.max_timeout_s)}s."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                    "description": {
                        "type": "string",
                        "description": (
                            "What this command does, in a few plain words the user "
                            "will see as progress, e.g. 'Searching your inbox for "
                            "receipts'. No jargon, no file paths."
                        ),
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": f"Seconds before the command is killed (max {int(self.max_timeout_s)}).",
                    },
                },
                "required": ["command", "description"],
            },
        )


# ---------------------------------------------------------------- formatting


def format_exec_result(result: ExecResult) -> str:
    parts: list[str] = []
    out = result.stdout.rstrip("\n")
    err = result.stderr.rstrip("\n")
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    if not parts:
        parts.append("(no output)")
    if result.timed_out:
        parts.append(f"[timed out after {result.duration_ms / 1000:.0f}s; the command was killed]")
    else:
        parts.append(f"[exit code {result.exit_code} · {result.duration_ms} ms]")
    return "\n".join(parts)


def _clamp_timeout(raw: Any, default: float, ceiling: float) -> float:
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(1.0, min(value, ceiling))


# --------------------------------------------------------------- file tools


def _path_arg(ws: Workspace, arguments: dict[str, Any]) -> str:
    raw = str(arguments.get("path") or arguments.get("file_path") or "").strip()
    if not raw:
        raise ToolError("Provide a `path`.")
    return ws.resolve(raw)


async def _read_text(ws: Workspace, path: str) -> str:
    try:
        data = await ws.read_file(path)
    except FileNotFoundError:
        raise ToolError(f"No such file: {path}") from None
    except IsADirectoryError:
        raise ToolError(f"{path} is a directory; list it with Bash (`ls {path}`).") from None
    if b"\x00" in data[:8192]:
        raise ToolError(
            f"{path} is a binary file ({len(data)} bytes). Inspect it with Bash "
            "(e.g. `file`, or a python3 script) instead."
        )
    return data.decode("utf-8", errors="replace")


async def _read(ws: Workspace, arguments: dict[str, Any]) -> str:
    path = _path_arg(ws, arguments)
    text = await _read_text(ws, path)
    lines = text.splitlines()
    try:
        offset = max(1, int(arguments.get("offset") or 1))
        limit = max(1, int(arguments.get("limit") or _READ_DEFAULT_LIMIT))
    except (TypeError, ValueError):
        raise ToolError("`offset` and `limit` must be integers.") from None
    if not lines:
        return f"({path} is empty)"
    if offset > len(lines):
        raise ToolError(f"{path} has only {len(lines)} lines; offset {offset} is past the end.")
    window = lines[offset - 1 : offset - 1 + limit]
    out: list[str] = []
    size = 0
    last = offset - 1
    for i, line in enumerate(window, start=offset):
        if len(line) > _READ_MAX_LINE_CHARS:
            line = line[:_READ_MAX_LINE_CHARS] + "… [line truncated]"
        rendered = f"{i:>6}\t{line}"
        size += len(rendered) + 1
        if size > _READ_MAX_CHARS:
            break
        out.append(rendered)
        last = i
    if last < len(lines):
        out.append(
            f"[showing lines {offset}-{last} of {len(lines)}; pass offset={last + 1} to read on]"
        )
    return "\n".join(out)


async def _write(ws: Workspace, arguments: dict[str, Any]) -> str:
    path = _path_arg(ws, arguments)
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ToolError("Provide the file's full text as `content`.")
    await ws.write_file(path, content.encode("utf-8"))
    return f"Wrote {path} ({len(content.splitlines())} lines)."


async def _edit(ws: Workspace, arguments: dict[str, Any]) -> str:
    path = _path_arg(ws, arguments)
    old = arguments.get("old_string")
    new = arguments.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        raise ToolError("Provide a non-empty `old_string` and a `new_string`.")
    if old == new:
        raise ToolError("`old_string` and `new_string` are identical; nothing to change.")
    text = await _read_text(ws, path)
    count = text.count(old)
    if count == 0:
        raise ToolError(
            f"`old_string` was not found in {path}. Read the file and copy the text exactly."
        )
    replace_all = bool(arguments.get("replace_all"))
    if count > 1 and not replace_all:
        raise ToolError(
            f"`old_string` appears {count} times in {path}. Include more surrounding "
            "text to make it unique, or pass replace_all=true."
        )
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    await ws.write_file(path, updated.encode("utf-8"))
    replaced = count if replace_all else 1
    return f"Edited {path} ({replaced} replacement{'s' if replaced != 1 else ''})."


_PATH_PROP = {"type": "string", "description": "File path; `~` and relative paths are under home."}

_READ_DEF = ToolDef(
    name="Read",
    description=(
        "Read a text file from your workspace, with line numbers. Reads up to "
        f"{_READ_DEFAULT_LIMIT} lines from `offset` (1-based); page through big files "
        "with offset/limit, or search them with Bash (rg) first."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": _PATH_PROP,
            "offset": {"type": "integer", "description": "First line to read (1-based)."},
            "limit": {"type": "integer", "description": "How many lines to read."},
        },
        "required": ["path"],
    },
)

_WRITE_DEF = ToolDef(
    name="Write",
    description=(
        "Create or overwrite a file in your workspace with the given text "
        "(parent directories are created). For a small change to an existing "
        "file, use Edit instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": _PATH_PROP,
            "content": {"type": "string", "description": "The file's full new text."},
        },
        "required": ["path", "content"],
    },
)

_EDIT_DEF = ToolDef(
    name="Edit",
    description=(
        "Replace exact text in a file: `old_string` must match the file exactly "
        "(including whitespace) and be unique unless `replace_all` is true."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": _PATH_PROP,
            "old_string": {"type": "string", "description": "The exact text to replace."},
            "new_string": {"type": "string", "description": "The replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        "required": ["path", "old_string", "new_string"],
    },
)


# ------------------------------------------------------------- narration


def describe_workspace_step(
    tool_name: str, arguments: dict[str, Any], context: Any = None
) -> Optional[StepInfo]:
    """A step-describer for the workspace tools (``None`` for other tools, so
    it composes: ``describe_workspace_step(...) or my_describer(...)``).

    ``Bash`` narrates the model's own ``description``; file tools name the
    file."""
    if tool_name == "Bash":
        title = str(arguments.get("description") or "").strip()
        command = str(arguments.get("command") or "").strip()
        if not title:
            title = command.splitlines()[0][:60] if command else "Ran a command"
        return StepInfo(title=title, detail=command[:200] or None, phase="run")
    path = str(arguments.get("path") or arguments.get("file_path") or "")
    name = posixpath.basename(path.rstrip("/")) or path or "a file"
    if tool_name == "Read":
        return StepInfo(title=f"Read {name}", detail=path or None, phase="read")
    if tool_name == "Write":
        return StepInfo(title=f"Wrote {name}", detail=path or None, phase="write")
    if tool_name == "Edit":
        return StepInfo(title=f"Edited {name}", detail=path or None, phase="write")
    return None
