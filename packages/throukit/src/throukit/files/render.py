"""Prompt body substitution (design §5).

Fill-in-the-blank only: ``{placeholder}`` is replaced from load-time
variables and the per-run context. Logic — conditionals, loops — is
deliberately unsupported; the moment a prompt needs an ``if`` is the signal
to define the agent in Python, where instructions can be a function.

Literal braces are escaped by doubling (``{{`` / ``}}``), same as
``str.format``.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any, Optional

from pydantic import BaseModel

from ..exceptions import PromptVariableError

__all__ = ["extract_placeholders", "render_prompt"]

_formatter = string.Formatter()


def extract_placeholders(template: str) -> set[str]:
    names: set[str] = set()
    for _, field_name, format_spec, conversion in _formatter.parse(template):
        if field_name is None:
            continue
        if not field_name or any(c in field_name for c in ".["):
            raise PromptVariableError(
                f"Unsupported placeholder {{{field_name}}}: only simple names are allowed "
                "(fill-in-the-blank, no attribute/index access)."
            )
        if format_spec or conversion:
            raise PromptVariableError(
                f"Unsupported placeholder {{{field_name}}}: format specs and conversions "
                "are not allowed in agent files."
            )
        names.add(field_name)
    return names


def _context_value(context: Any, key: str) -> tuple[bool, Any]:
    if context is None:
        return False, None
    if isinstance(context, Mapping):
        if key in context:
            return True, context[key]
        return False, None
    if isinstance(context, BaseModel):
        if key in type(context).model_fields:
            return True, getattr(context, key)
        return False, None
    if hasattr(context, key):
        return True, getattr(context, key)
    return False, None


def render_prompt(
    template: str,
    *,
    context: Any = None,
    variables: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render a prompt body. Variables win over context on name clashes."""
    values: dict[str, Any] = {}
    missing: list[str] = []
    for name in extract_placeholders(template):
        if variables is not None and name in variables:
            values[name] = variables[name]
            continue
        found, value = _context_value(context, name)
        if found:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise PromptVariableError(
            f"Prompt placeholder(s) {sorted(missing)} could not be filled from "
            "the run context or load-time variables."
        )
    return template.format_map(values)
