"""Cases, and what running one gives back (design §7.2)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["Case", "CaseResult", "load_cases"]


@dataclass(frozen=True)
class Case:
    """One unit of work.

    Args:
        id: Unique within the suite; ``--cases`` globs match it.
        input: What the case hands the target.
        expected: What scorers compare the result against.
        context: The opaque per-run context an agent target runs with.
        tags: Scope bars (``Bar.parse(expr, group="a")``) and group aggregates.
        estimate_usd: Estimated spend per trial, in place of the suite's default.
        default: ``False`` cases run only when ``--cases`` names them.
    """

    id: str
    input: Any = None
    expected: Any = None
    context: Any = None
    tags: Mapping[str, str] = field(default_factory=dict)
    estimate_usd: Optional[float] = None
    default: bool = True


@dataclass
class CaseResult:
    """What ``Suite.run_case`` returns.

    ``metrics`` are the numbers and short labels bars and tables read;
    ``details`` is free-form evidence for the JSON report. ``cost_usd`` is
    only spend the meter cannot see (a subprocess, another vendor): metered
    calls are counted already, and the runner writes the case's total spend
    to ``metrics["cost_usd"]`` itself, with ``metrics["seconds"]`` beside it.
    The fields below ``cost_usd`` are the runner's.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    case_id: str = ""
    trial: int = 1
    tags: dict[str, str] = field(default_factory=dict)
    status: str = "ok"  # ok | error | budget
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """It ran, and its ``pass`` metric, if it has one, holds."""
        return self.status == "ok" and bool(self.metrics.get("pass", True))


def load_cases(path: Union[str, Path]) -> list[Case]:
    """Load cases from a JSONL file of ``Case``-shaped objects. A line with
    no ``id`` is named after its position."""
    cases = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            data = json.loads(line)
            data.setdefault("id", f"case_{i}")
            cases.append(Case(**data))
    return cases
