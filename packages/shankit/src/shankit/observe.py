"""User-facing observability: the step-describer contract (design §6).

A step-describer translates a raw tool call into a user-language step. It is
a plain callable ``(tool_name, arguments, context) -> StepInfo | None``:

- return a :class:`StepInfo` to narrate the call on the event stream,
- return ``None`` to hide the call from the user entirely.

The default describer produces a generic humanized title so narration works
out of the box; real products supply their own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from pydantic import BaseModel

__all__ = ["StepDescriber", "StepInfo", "default_step_describer"]


class StepInfo(BaseModel):
    """What a step-describer returns for one tool call."""

    title: str
    detail: Optional[str] = None
    phase: Optional[str] = None


StepDescriber = Callable[[str, dict[str, Any], Any], Optional[StepInfo]]


def default_step_describer(
    tool_name: str, arguments: dict[str, Any], context: Any = None
) -> StepInfo:
    """A generic fallback: ``search_email`` -> ``"Search email"``."""
    title = tool_name.replace("_", " ").replace("-", " ").strip()
    return StepInfo(title=title[:1].upper() + title[1:] if title else tool_name)
