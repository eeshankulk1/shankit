"""Loading agents from markdown files (design §5).

The file format mirrors Claude Code subagent files wherever concepts
overlap — YAML frontmatter for identity/description/model/tools, markdown
body as the system prompt — and diverges only to *add* one key:
``output_schema``, whose presence lets the agent be run for structured
output. A file produces the same :class:`~shankit.Agent` the Python
constructor would; the file is sugar.

Example (``triage.md``)::

    ---
    name: email-triage
    description: Triages the inbox and drafts replies.
    model: anthropic:claude-sonnet-4-5
    tools:
      - myapp.tools:search_email     # module:attribute import reference
      - gmail                        # explicitly registered live object
    output_schema: myapp.schemas:TriageResult
    ---
    You are an email triage assistant for {user_name}.
    ...

Body placeholders are fill-in-the-blank only, resolved at run time from the
per-run context (and load-time ``variables``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ..agent import Agent
from ..exceptions import AgentFileError
from ..observe import StepDescriber, default_step_describer
from .refs import resolve_ref
from .registry import Registry, default_registry
from .render import extract_placeholders, render_prompt

__all__ = ["load_agent", "load_agents"]

# Frontmatter keys mirror Claude Code subagents (name, description, model,
# tools); `output_schema` is our one addition (design §5).
_ALLOWED_KEYS = {"name", "description", "model", "tools", "output_schema"}


def load_agent(
    path: Union[str, Path],
    *,
    registry: Optional[Registry] = None,
    variables: Optional[Mapping[str, Any]] = None,
    default_model: Optional[str] = None,
    describe_step: Optional[StepDescriber] = default_step_describer,
) -> Agent:
    """Load one agent definition file into an :class:`~shankit.Agent`.

    Args:
        registry: Where short-name references resolve; defaults to the
            module-level default registry.
        variables: Load-time values for body placeholders (context values
            fill the rest at run time).
        default_model: Used when the file omits ``model``.
        describe_step: Step-describer to attach (observability is configured
            in code, not prose).
    """
    path = Path(path)
    registry = registry if registry is not None else default_registry
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentFileError(f"Could not read agent file {path}: {exc}") from exc

    frontmatter, body = _split_frontmatter(raw, path)

    unknown = set(frontmatter) - _ALLOWED_KEYS
    if unknown:
        raise AgentFileError(
            f"{path}: unknown frontmatter key(s) {sorted(unknown)}. "
            f"Allowed keys: {sorted(_ALLOWED_KEYS)}."
        )

    name = frontmatter.get("name") or path.stem
    model = frontmatter.get("model") or default_model
    if not model:
        raise AgentFileError(
            f"{path}: no `model` in frontmatter and no default_model provided. "
            "Set model: 'provider:model_id' in the file or pass default_model=."
        )

    tools: list[Any] = []
    raw_tools = frontmatter.get("tools") or []
    if not isinstance(raw_tools, list):
        raise AgentFileError(f"{path}: `tools` must be a list of references.")
    for ref in raw_tools:
        if not isinstance(ref, str):
            raise AgentFileError(f"{path}: tool reference {ref!r} must be a string.")
        resolved = resolve_ref(ref, registry)
        if isinstance(resolved, Agent):
            resolved = resolved.as_tool()
        if isinstance(resolved, (list, tuple)):
            tools.extend(resolved)
        else:
            tools.append(resolved)

    output_type: Optional[type] = None
    if "output_schema" in frontmatter:
        schema_ref = frontmatter["output_schema"]
        if not isinstance(schema_ref, str) or ":" not in schema_ref:
            raise AgentFileError(
                f"{path}: `output_schema` must be a 'module:attribute' import reference."
            )
        output_type = resolve_ref(schema_ref, registry)
        if not isinstance(output_type, type):
            raise AgentFileError(
                f"{path}: output_schema reference {schema_ref!r} resolved to "
                f"{output_type!r}, which is not a type."
            )

    body = body.strip()
    instructions: Any
    if extract_placeholders(body):
        load_time_variables = dict(variables) if variables else {}

        def instructions(context: Any, _body: str = body) -> str:
            return render_prompt(_body, context=context, variables=load_time_variables)

    else:
        # No placeholders: static prose (escaped braces resolved now).
        instructions = body.replace("{{", "{").replace("}}", "}")

    return Agent(
        name=str(name),
        description=frontmatter.get("description"),
        model=str(model),
        instructions=instructions,
        tools=tools,
        output_type=output_type,
        describe_step=describe_step,
    )


def load_agents(
    directory: Union[str, Path],
    *,
    registry: Optional[Registry] = None,
    variables: Optional[Mapping[str, Any]] = None,
    default_model: Optional[str] = None,
    describe_step: Optional[StepDescriber] = default_step_describer,
) -> dict[str, Agent]:
    """Load every ``*.md`` agent file in a directory (non-recursive).

    Returns agents keyed by name; duplicate names across files are an error.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise AgentFileError(f"{directory} is not a directory.")
    agents: dict[str, Agent] = {}
    for path in sorted(directory.glob("*.md")):
        agent = load_agent(
            path,
            registry=registry,
            variables=variables,
            default_model=default_model,
            describe_step=describe_step,
        )
        if agent.name in agents:
            raise AgentFileError(
                f"Duplicate agent name {agent.name!r} in {directory} (file {path.name})."
            )
        agents[agent.name] = agent
    return agents


def _split_frontmatter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    if not raw.lstrip().startswith("---"):
        raise AgentFileError(
            f"{path}: agent files start with a `---` YAML frontmatter block."
        )
    stripped = raw.lstrip()
    parts = stripped.split("\n---", 1)
    if len(parts) != 2:
        raise AgentFileError(f"{path}: unterminated frontmatter block.")
    yaml_text = parts[0].removeprefix("---")
    body = parts[1]
    if body.startswith("-"):  # guard against '----' style separators
        raise AgentFileError(f"{path}: malformed frontmatter separator.")
    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise AgentFileError(f"{path}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentFileError(f"{path}: frontmatter must be a YAML mapping.")
    return data, body
