"""The evaluation runner (design §7.2)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

from ..agent import Agent, RunResult
from .dataset import Case, Dataset
from .scorers import Score, Scorer, call_scorer

__all__ = ["CaseResult", "EvalReport", "evaluate"]

Target = Union[Agent, Callable[[Case], Awaitable[RunResult]]]


@dataclass
class CaseResult:
    case: Case
    result: Optional[RunResult] = None
    error: Optional[str] = None
    scores: list[Score] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A case passes when it ran and no score explicitly failed."""
        if self.error is not None:
            return False
        return all(s.passed is not False for s in self.scores)


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def mean(self, score_name: str) -> float:
        values = [s.value for r in self.results for s in r.scores if s.name == score_name]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> str:
        lines = [f"{len(self.results)} case(s), pass rate {self.pass_rate:.0%}"]
        for r in self.results:
            status = "ERROR" if r.error else ("PASS" if r.passed else "FAIL")
            detail = r.error or ", ".join(f"{s.name}={s.value:.2f}" for s in r.scores)
            lines.append(f"  [{status}] {r.case.name}: {detail}")
        return "\n".join(lines)


async def evaluate(
    target: Target,
    dataset: Dataset,
    scorers: Sequence[Scorer] = (),
    *,
    output_type: Optional[type] = None,
    max_concurrency: int = 4,
) -> EvalReport:
    """Run every case against the target and score the results.

    ``target`` is an :class:`Agent` (run with each case's input and context)
    or any async callable ``(case) -> RunResult``. A case whose run raises is
    reported as an error, not a crash of the whole evaluation.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_case(case: Case) -> CaseResult:
        async with semaphore:
            try:
                if isinstance(target, Agent):
                    effective = output_type or target.output_type or str
                    result = await target.run(
                        case.input, context=case.context, output_type=effective
                    )
                else:
                    result = await target(case)
            except Exception as exc:
                return CaseResult(case=case, error=f"{type(exc).__name__}: {exc}")
            scores = [await call_scorer(scorer, case, result) for scorer in scorers]
            return CaseResult(case=case, result=result, scores=scores)

    results = await asyncio.gather(*(run_case(case) for case in dataset.cases))
    return EvalReport(results=list(results))
