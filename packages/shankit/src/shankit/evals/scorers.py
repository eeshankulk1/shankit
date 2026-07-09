"""Scorers (design §7.2).

A scorer is any callable ``(case, result) -> Score | float | bool`` (sync or
async). Floats become unlabelled scores; bools become pass/fail.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union

from pydantic import BaseModel

from .._serialize import dump_str
from ..agent import Agent, RunResult
from ..models import ModelClient

__all__ = ["Score", "Scorer", "exact_match", "llm_judge", "normalize_score", "output_contains"]


class Score(BaseModel):
    name: str
    value: float
    passed: Optional[bool] = None
    reason: Optional[str] = None


ScorerReturn = Union["Score", float, bool]
Scorer = Callable[..., Union[ScorerReturn, Awaitable[ScorerReturn]]]


def normalize_score(raw: ScorerReturn, default_name: str) -> Score:
    if isinstance(raw, Score):
        if not raw.name:
            raw.name = default_name
        return raw
    if isinstance(raw, bool):
        return Score(name=default_name, value=1.0 if raw else 0.0, passed=raw)
    return Score(name=default_name, value=float(raw))


async def call_scorer(scorer: Scorer, case: Any, result: RunResult) -> Score:
    name = getattr(scorer, "__name__", type(scorer).__name__)
    raw = scorer(case, result)
    if inspect.isawaitable(raw):
        raw = await raw
    return normalize_score(raw, name)


# ---------------------------------------------------------------- built-ins


def exact_match(case: Any, result: RunResult) -> Score:
    """Pass iff ``result.output`` equals ``case.expected`` (pydantic outputs
    compare by their dict dump)."""
    output = result.output
    if isinstance(output, BaseModel):
        output = output.model_dump()
    passed = output == case.expected
    return Score(name="exact_match", value=1.0 if passed else 0.0, passed=passed)


def output_contains(substring: Optional[str] = None) -> Scorer:
    """Pass iff the substring (or ``case.expected`` when omitted) appears in
    the result's deliverable.

    Scores ``result.output`` (falling back to ``result.text`` when there is
    no output) — the answer, not the full transcript, so an agent that
    merely *mentions* the expected string in an interim pass before
    answering something else does not pass.
    """

    def scorer(case: Any, result: RunResult) -> Score:
        needle = substring if substring is not None else str(case.expected)
        haystack = dump_str(result.output) if result.output is not None else result.text
        passed = needle in haystack
        return Score(name="output_contains", value=1.0 if passed else 0.0, passed=passed)

    scorer.__name__ = "output_contains"
    return scorer


class _JudgeVerdict(BaseModel):
    score: float
    reason: str


def llm_judge(
    rubric: str,
    *,
    model: str,
    threshold: float = 0.7,
    model_client: Optional[ModelClient] = None,
) -> Scorer:
    """A model-graded scorer: judges the result against a rubric, returning
    a 0..1 score. The judge is itself a structured agent run."""
    judge = Agent(
        name="judge",
        model=model,
        model_client=model_client,
        instructions=(
            "You are a strict evaluator. Score the candidate response against "
            "the rubric from 0.0 (fails entirely) to 1.0 (fully satisfies)."
        ),
        output_type=_JudgeVerdict,
        describe_step=None,
    )

    async def scorer(case: Any, result: RunResult) -> Score:
        # Judge the deliverable, not the transcript: interim narration
        # passes would otherwise shift scores on rubrics about format or
        # conciseness without the agent's answer changing.
        candidate = dump_str(result.output) if result.output is not None else result.text
        verdict = await judge.run(
            f"Rubric: {rubric}\n\nTask input: {case.input}\n\nCandidate response:\n{candidate}"
        )
        return Score(
            name="llm_judge",
            value=verdict.output.score,
            passed=verdict.output.score >= threshold,
            reason=verdict.output.reason,
        )

    scorer.__name__ = "llm_judge"
    return scorer
