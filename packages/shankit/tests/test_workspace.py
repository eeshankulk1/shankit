"""The workspace seam, the local workspace, and the Bash/Read/Write/Edit tools."""

from __future__ import annotations

import pytest
from shankit import (
    ExecResult,
    LocalWorkspace,
    ToolError,
    Workspace,
    WorkspaceTools,
    describe_workspace_step,
)


class MemoryWorkspace(Workspace):
    """A file-only workspace (no exec) backed by a dict."""

    home = "/home/user"
    tmp = "/tmp"

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def read_file(self, path: str) -> bytes:
        path = self.resolve(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, data: bytes) -> None:
        self.files[self.resolve(path)] = data


# ----------------------------------------------------------------- seam


def test_resolve_expands_home_and_relative_paths():
    ws = MemoryWorkspace()
    assert ws.resolve("~/notes.md") == "/home/user/notes.md"
    assert ws.resolve("~") == "/home/user"
    assert ws.resolve("files/a.csv") == "/home/user/files/a.csv"
    assert ws.resolve("/tmp/../tmp/x") == "/tmp/x"


def test_can_exec_reflects_whether_exec_is_implemented(tmp_path):
    assert MemoryWorkspace().can_exec is False
    assert LocalWorkspace(tmp_path).can_exec is True


# -------------------------------------------------------------- local


async def test_local_exec_captures_output_exit_code_and_duration(tmp_path):
    ws = LocalWorkspace(tmp_path)
    result = await ws.exec("echo out; echo err >&2; exit 3")
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.exit_code == 3
    assert result.timed_out is False
    assert result.duration_ms >= 0


async def test_local_exec_runs_in_home_with_a_minimal_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_API_KEY", "sk-should-not-leak")
    ws = LocalWorkspace(tmp_path, env={"FOO": "bar"})
    result = await ws.exec(
        'pwd; echo "$HOME"; echo "$FOO/$EXTRA"; echo "[$SECRET_API_KEY]"', env={"EXTRA": "x"}
    )
    lines = result.stdout.splitlines()
    assert lines[0].endswith("/home")
    assert lines[1] == ws.home
    assert lines[2] == "bar/x"
    assert lines[3] == "[]"  # the host process's environment doesn't leak in


async def test_local_exec_timeout_kills_the_command(tmp_path):
    ws = LocalWorkspace(tmp_path)
    result = await ws.exec("sleep 5; echo never", timeout=0.3)
    assert result.timed_out is True
    assert "never" not in result.stdout


async def test_local_files_round_trip_and_create_parents(tmp_path):
    ws = LocalWorkspace(tmp_path)
    await ws.write_file("~/deep/dir/a.txt", b"hello")
    assert await ws.read_file("~/deep/dir/a.txt") == b"hello"
    assert (await ws.exec("cat deep/dir/a.txt")).stdout == "hello"
    with pytest.raises(FileNotFoundError):
        await ws.read_file("~/missing.txt")


# -------------------------------------------------------------- tools


async def test_tool_list_drops_bash_for_file_only_workspaces(tmp_path):
    names = [t.name for t in await WorkspaceTools(MemoryWorkspace()).list_tools()]
    assert names == ["Read", "Write", "Edit"]
    names = [t.name for t in await WorkspaceTools(LocalWorkspace(tmp_path)).list_tools()]
    assert names == ["Bash", "Read", "Write", "Edit"]


async def test_tool_list_is_empty_without_a_workspace():
    tools = WorkspaceTools(lambda ctx: None)
    assert await tools.list_tools() == []


async def test_workspace_resolves_from_context(tmp_path):
    spaces = {"u1": LocalWorkspace(tmp_path / "u1"), "u2": LocalWorkspace(tmp_path / "u2")}
    tools = WorkspaceTools(lambda ctx: spaces[ctx["user"]])
    await tools.execute("Write", {"path": "~/who.txt", "content": "u1"}, {"user": "u1"})
    assert (tmp_path / "u1" / "home" / "who.txt").read_text() == "u1"
    assert not (tmp_path / "u2" / "home" / "who.txt").exists()


async def test_bash_formats_output_and_nonzero_exit_is_not_an_error(tmp_path):
    tools = WorkspaceTools(LocalWorkspace(tmp_path))
    result = await tools.execute("Bash", {"command": "echo hi; exit 1", "description": "x"})
    assert result.is_error is False
    assert result.content.startswith("hi\n")
    assert "[exit code 1 ·" in result.content


async def test_bash_timeout_is_an_error_and_clamped(tmp_path):
    tools = WorkspaceTools(LocalWorkspace(tmp_path), max_timeout_s=1)
    result = await tools.execute("Bash", {"command": "sleep 3", "timeout_s": 999})
    assert result.is_error is True
    assert "timed out" in result.content


async def test_bash_env_resolves_per_run(tmp_path):
    tools = WorkspaceTools(LocalWorkspace(tmp_path), env=lambda ctx: {"RUN_TOKEN": ctx["token"]})
    result = await tools.execute("Bash", {"command": "echo $RUN_TOKEN"}, {"token": "abc"})
    assert result.content.startswith("abc")


async def test_bash_rejects_empty_command(tmp_path):
    tools = WorkspaceTools(LocalWorkspace(tmp_path))
    with pytest.raises(ToolError):
        await tools.execute("Bash", {"command": "  "})


async def test_read_numbers_lines_and_windows():
    ws = MemoryWorkspace()
    ws.files["/home/user/f.txt"] = "\n".join(f"line {i}" for i in range(1, 11)).encode()
    tools = WorkspaceTools(ws)
    out = (await tools.execute("Read", {"path": "f.txt", "offset": 3, "limit": 2})).content
    assert out.splitlines()[0] == "     3\tline 3"
    assert out.splitlines()[1] == "     4\tline 4"
    assert "pass offset=5" in out


async def test_read_errors_are_model_visible():
    ws = MemoryWorkspace()
    ws.files["/home/user/bin"] = b"\x00\x01binary"
    tools = WorkspaceTools(ws)
    with pytest.raises(ToolError, match="No such file"):
        await tools.execute("Read", {"path": "nope.txt"})
    with pytest.raises(ToolError, match="binary"):
        await tools.execute("Read", {"path": "bin"})


async def test_write_then_edit():
    ws = MemoryWorkspace()
    tools = WorkspaceTools(ws)
    await tools.execute("Write", {"path": "~/memo.md", "content": "a\nb\na\n"})
    with pytest.raises(ToolError, match="appears 2 times"):
        await tools.execute("Edit", {"path": "~/memo.md", "old_string": "a", "new_string": "z"})
    with pytest.raises(ToolError, match="not found"):
        await tools.execute("Edit", {"path": "~/memo.md", "old_string": "q", "new_string": "z"})
    out = await tools.execute(
        "Edit", {"path": "~/memo.md", "old_string": "a", "new_string": "z", "replace_all": True}
    )
    assert "2 replacements" in out.content
    assert ws.files["/home/user/memo.md"] == b"z\nb\nz\n"


async def test_after_tool_hook_sees_exec_results(tmp_path):
    seen: list[tuple[str, ExecResult | None]] = []

    class Recording(WorkspaceTools):
        async def after_tool(self, name, arguments, result, context, workspace, exec_result):
            seen.append((name, exec_result))
            return result.model_copy(update={"content": result.content + "\n[synced]"})

    tools = Recording(LocalWorkspace(tmp_path))
    result = await tools.execute("Bash", {"command": "true"})
    assert result.content.endswith("[synced]")
    await tools.execute("Write", {"path": "x", "content": "y"})
    assert seen[0][0] == "Bash"
    assert seen[0][1] is not None
    assert seen[0][1].exit_code == 0
    assert seen[1] == ("Write", None)


def test_step_describer_narrates_workspace_tools():
    step = describe_workspace_step("Bash", {"command": "ls -la", "description": "Listing files"})
    assert step is not None
    assert step.title == "Listing files"
    assert step.detail == "ls -la"
    assert describe_workspace_step("Read", {"path": "~/memory/trips.md"}).title == "Read trips.md"
    assert describe_workspace_step("Edit", {"path": "/tmp/a.py"}).phase == "write"
    assert describe_workspace_step("search_email", {}) is None
