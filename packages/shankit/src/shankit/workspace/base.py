"""The workspace seam: a place an agent can keep files and run commands.

A :class:`Workspace` is the agent's computer — a filesystem, and optionally a
shell. It is a contract, not a vendor: a hosted sandbox (a microVM per user),
a container, or a local directory all implement the same three methods. The
loop and the shipped file/shell tools (:class:`~shankit.WorkspaceTools`) only
ever see this contract, so switching vendors never touches agent code.

``exec`` is optional: a workspace without it is legal and offers file tools
only (``can_exec`` is false). ``read_file``/``write_file`` are required.

Workspaces are usually per user, so an agent names one the same way a
connector names *whose* credentials to use — as a function of the opaque
per-run context::

    agent = Agent(..., workspace=lambda ctx: sandboxes.for_user(ctx.user_id))

Paths: absolute paths are used as-is; ``~`` and relative paths resolve
against :attr:`Workspace.home`. :attr:`Workspace.tmp` is scratch space the
loop may use (e.g. to spill oversized tool results to files).
"""

from __future__ import annotations

import abc
import inspect
import posixpath
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Optional, Union

from pydantic import BaseModel

__all__ = [
    "ExecResult",
    "Workspace",
    "WorkspaceSpec",
    "resolve_workspace",
]


class ExecResult(BaseModel):
    """The outcome of one command run in a workspace."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    timed_out: bool = False


class Workspace(abc.ABC):
    """Files (required) and a shell (optional) for an agent to work in.

    Implementations must make ``write_file`` create missing parent
    directories, and should raise :class:`FileNotFoundError` from
    ``read_file`` for a missing path (``IsADirectoryError`` for a
    directory) so the shipped tools can report it to the model plainly.
    """

    #: Home directory: where ``~`` and relative paths resolve, and the
    #: default working directory for commands.
    home: str = "/"
    #: Scratch directory. Contents may disappear when the workspace stops.
    tmp: str = "/tmp"

    @property
    def can_exec(self) -> bool:
        """Whether this workspace runs commands (implements ``exec``)."""
        return type(self).exec is not Workspace.exec

    async def exec(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ExecResult:
        """Run ``command`` in a shell. ``timeout`` is in seconds; a command
        that exceeds it is killed and reported with ``timed_out=True``
        rather than raised. ``env`` adds to (does not replace) the
        workspace's own environment for this one command."""
        raise NotImplementedError("This workspace cannot run commands.")

    @abc.abstractmethod
    async def read_file(self, path: str) -> bytes:
        """The bytes of the file at ``path``."""

    @abc.abstractmethod
    async def write_file(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path``, creating parent directories."""

    def resolve(self, path: str) -> str:
        """An absolute, normalized path: ``~`` and relative paths resolve
        against :attr:`home`."""
        path = path.strip()
        if path == "~" or path.startswith("~/"):
            path = self.home + path[1:]
        elif not path.startswith("/"):
            path = posixpath.join(self.home, path)
        return posixpath.normpath(path)


WorkspaceSpec = Union[
    Workspace,
    Callable[[Any], Union[Optional[Workspace], Awaitable[Optional[Workspace]]]],
]


async def resolve_workspace(spec: Optional[WorkspaceSpec], context: Any) -> Optional[Workspace]:
    """Resolve a workspace spec (an instance, or a sync/async function of the
    per-run context) for one run. ``None`` means no workspace."""
    if spec is None or isinstance(spec, Workspace):
        return spec
    resolved = spec(context)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    return resolved
