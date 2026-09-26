"""A workspace on the local machine: a directory plus a subprocess shell.

For development, tests, and single-user tools. **Not a security boundary**:
commands run as the current OS user with full access to the host (the
directory is only the default home and scratch location, not a jail). Host
an agent that reads untrusted content (email, web pages) for other people in
a real sandbox — a hosted microVM or container behind the same
:class:`~shankit.Workspace` contract.

Commands get a minimal environment (``PATH``, ``HOME``, ``TMPDIR``,
``LANG`` plus whatever the caller passes), never the parent process's
environment, so API keys in the host process don't leak into agent-run
commands by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

from .base import ExecResult, Workspace

__all__ = ["LocalWorkspace"]

_DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"


class LocalWorkspace(Workspace):
    """``root/home`` is home, ``root/tmp`` is scratch.

    Args:
        root: Directory to create the workspace in (created if missing).
        extra_path: Directories prepended to ``PATH`` for commands (e.g. a
            directory of CLIs the agent should be able to call).
        env: Environment added to every command.
        default_timeout: Seconds before a command is killed, when the
            caller doesn't pass one.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        extra_path: Sequence[Union[str, Path]] = (),
        env: Optional[Mapping[str, str]] = None,
        default_timeout: float = 60.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.home = str(self.root / "home")
        self.tmp = str(self.root / "tmp")
        Path(self.home).mkdir(parents=True, exist_ok=True)
        Path(self.tmp).mkdir(parents=True, exist_ok=True)
        self._extra_path = [str(p) for p in extra_path]
        self._env = dict(env or {})
        self._default_timeout = default_timeout

    def __repr__(self) -> str:
        return f"LocalWorkspace(root={str(self.root)!r})"

    def _base_env(self) -> dict[str, str]:
        path = ":".join([*self._extra_path, os.environ.get("PATH", _DEFAULT_PATH)])
        return {
            "PATH": path,
            "HOME": self.home,
            "TMPDIR": self.tmp,
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            **self._env,
        }

    async def exec(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ExecResult:
        limit = timeout if timeout is not None else self._default_timeout
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            command,
            cwd=self.resolve(cwd) if cwd else self.home,
            env={**self._base_env(), **(env or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Own process group, so a timeout kills the whole pipeline, not
            # just the shell.
            start_new_session=True,
        )
        # Drain stdout/stderr concurrently with waiting for exit, in our own
        # tasks rather than via `proc.communicate()`. A timeout needs to
        # kill the process and then keep reading; calling `communicate()`
        # a second time after a kill loses whatever the first (cancelled)
        # call had already pulled off the pipe into its own coroutine frame
        # but not yet returned — a command that prints something and then
        # hangs would report empty output. Reading in tasks that are never
        # cancelled avoids that: their buffered bytes survive the timeout.
        assert proc.stdout is not None  # requested PIPE above
        assert proc.stderr is not None
        stdout_task = asyncio.ensure_future(proc.stdout.read())
        stderr_task = asyncio.ensure_future(proc.stderr.read())
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=limit)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_group(proc.pid)
            await proc.wait()
        except BaseException:
            # Cancelled (run abandoned): don't leave the command running.
            _kill_group(proc.pid)
            stdout_task.cancel()
            stderr_task.cancel()
            raise
        stdout = await stdout_task
        stderr = await stderr_task
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
        )

    async def read_file(self, path: str) -> bytes:
        return await asyncio.to_thread(Path(self.resolve(path)).read_bytes)

    async def write_file(self, path: str, data: bytes) -> None:
        target = Path(self.resolve(path))

        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(write)


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
