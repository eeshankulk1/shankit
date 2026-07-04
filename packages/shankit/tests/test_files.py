import pytest
from conftest import FakeModel, text_response, tool_call_response
from shankit import (
    Agent,
    AgentFileError,
    PromptVariableError,
    Registry,
    load_agent,
    load_agents,
)
from shankit.files.render import render_prompt


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


FULL_FILE = """---
name: email-triage
description: Triages the inbox.
model: anthropic:claude-sonnet-4-5
tools:
  - sample_defs:search_email
  - gmail
output_schema: sample_defs:Triage
---
You are an email triage assistant for {user_name}.
Today is {today}.
"""


def test_load_full_file(tmp_path):
    registry = Registry()
    sentinel_source = _StubSource()
    registry.register("gmail", sentinel_source)

    agent = load_agent(write(tmp_path, "triage.md", FULL_FILE), registry=registry,
                       variables={"today": "2026-07-04"})

    assert agent.name == "email-triage"
    assert agent.description == "Triages the inbox."
    assert agent.model == "anthropic:claude-sonnet-4-5"
    from sample_defs import Triage

    assert agent.output_type is Triage
    # dynamic instructions: context fills the rest of the blanks at run time
    rendered = agent.instructions({"user_name": "Ada"})
    assert rendered == "You are an email triage assistant for Ada.\nToday is 2026-07-04."


class _StubSource:
    async def list_tools(self, context=None):
        return []

    async def execute(self, name, arguments, context=None):
        raise NotImplementedError


def test_missing_context_key_raises(tmp_path):
    agent = load_agent(write(tmp_path, "a.md", FULL_FILE.replace("  - gmail\n", "")),
                       registry=Registry())
    with pytest.raises(PromptVariableError, match="user_name"):
        agent.instructions({})


def test_static_body_stays_string(tmp_path):
    content = "---\nmodel: anthropic:m\n---\nJust prose. Escaped {{braces}} stay literal.\n"
    agent = load_agent(write(tmp_path, "static.md", content))
    assert agent.instructions == "Just prose. Escaped {braces} stay literal."
    assert agent.name == "static"  # defaults to the file stem


def test_unknown_frontmatter_key_rejected(tmp_path):
    content = "---\nmodel: anthropic:m\nmode: chat\n---\nbody\n"
    with pytest.raises(AgentFileError, match="mode"):
        load_agent(write(tmp_path, "bad.md", content))


def test_missing_model_rejected_unless_default(tmp_path):
    content = "---\nname: x\n---\nbody\n"
    path = write(tmp_path, "nomodel.md", content)
    with pytest.raises(AgentFileError, match="model"):
        load_agent(path)
    agent = load_agent(path, default_model="openai:gpt-4.1")
    assert agent.model == "openai:gpt-4.1"


def test_list_reference_expands(tmp_path):
    content = "---\nmodel: anthropic:m\ntools:\n  - sample_defs:toolbox\n---\nbody\n"
    agent = load_agent(write(tmp_path, "box.md", content))
    assert agent.tool_source is not None


def test_agent_reference_becomes_tool(tmp_path):
    registry = Registry()
    sub = Agent(name="helper", model="fake", model_client=FakeModel([]))
    registry.register("helper", sub)
    content = "---\nmodel: anthropic:m\ntools:\n  - helper\n---\nbody\n"
    agent = load_agent(write(tmp_path, "parent.md", content), registry=registry)
    assert agent.tool_source is not None


def test_registry_helpful_error(tmp_path):
    content = "---\nmodel: anthropic:m\ntools:\n  - ghost\n---\nbody\n"
    with pytest.raises(AgentFileError, match="Nothing registered"):
        load_agent(write(tmp_path, "ghost.md", content), registry=Registry())


def test_bad_import_reference(tmp_path):
    content = "---\nmodel: anthropic:m\ntools:\n  - not_a_module:thing\n---\nbody\n"
    with pytest.raises(AgentFileError, match="Could not import"):
        load_agent(write(tmp_path, "imp.md", content))


def test_output_schema_must_be_type(tmp_path):
    content = "---\nmodel: anthropic:m\noutput_schema: sample_defs:toolbox\n---\nbody\n"
    with pytest.raises(AgentFileError, match="not a type"):
        load_agent(write(tmp_path, "sch.md", content))


def test_load_agents_directory(tmp_path):
    write(tmp_path, "one.md", "---\nmodel: anthropic:m\n---\nA\n")
    write(tmp_path, "two.md", "---\nmodel: anthropic:m\n---\nB\n")
    write(tmp_path, "notes.txt", "ignored")
    agents = load_agents(tmp_path)
    assert set(agents) == {"one", "two"}


def test_load_agents_duplicate_names(tmp_path):
    write(tmp_path, "a.md", "---\nname: same\nmodel: anthropic:m\n---\nA\n")
    write(tmp_path, "b.md", "---\nname: same\nmodel: anthropic:m\n---\nB\n")
    with pytest.raises(AgentFileError, match="Duplicate"):
        load_agents(tmp_path)


async def test_file_agent_runs_end_to_end(tmp_path):
    """A file produces the same agent the constructor would."""
    content = """---
name: runner
model: anthropic:placeholder
tools:
  - sample_defs:search_email
---
Help {user_name} with email.
"""
    agent = load_agent(write(tmp_path, "runner.md", content))
    # swap in a scripted client (the file's model string is bypassed)
    agent.model_client = FakeModel(
        [tool_call_response("search_email", {"query": "urgent"}), text_response("done")]
    )
    result = await agent.run("go", context={"user_name": "Ada"}, output_type=str)
    assert result.text == "done"
    assert result.trajectory[0].tool == "search_email"


def test_render_prompt_rules():
    assert render_prompt("Hi {name}", context={"name": "x"}) == "Hi x"
    with pytest.raises(PromptVariableError, match="simple names"):
        render_prompt("Hi {user.name}", context={})
    with pytest.raises(PromptVariableError, match="format specs"):
        render_prompt("Hi {n:>10}", context={"n": 1})
    # variables win over context
    assert render_prompt("{a}", context={"a": "ctx"}, variables={"a": "var"}) == "var"
    # pydantic and attribute contexts work
    from pydantic import BaseModel

    class Ctx(BaseModel):
        who: str

    assert render_prompt("{who}", context=Ctx(who="me")) == "me"
