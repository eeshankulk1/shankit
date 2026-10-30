"""Suites: what a run of cases is made of (design §7.2)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from ..agent import Agent, RunResult
from .bars import Bar
from .budget import Meter
from .cases import Case, CaseResult
from .scorers import Scorer, call_scorer

__all__ = ["AgentSuite", "EvalRun", "Suite", "SuiteError", "Target"]

Target = Union[Agent, Callable[[Case], Awaitable[RunResult]]]


class SuiteError(Exception):
    """A suite can't run here (missing data, wrong database, missing user
    row); the message says what to do."""


@dataclass
class EvalRun:
    """What a suite sees while it runs. Each case gets its own copy with
    ``trial`` set; ``meter`` is shared."""

    suite: str
    meter: Meter
    identity: Any = None  # who the run acts as; opaque to the framework
    # The report's name for ``identity``: never a secret.
    identity_label: str = ""
    # Strings scrubbed from reports, whatever a suite put in its details.
    redact: tuple[str, ...] = ()
    model: Optional[str] = None
    options: Mapping[str, str] = field(default_factory=dict)
    trials: int = 1
    trial: int = 1  # the trial this case is on (1-based)


class Suite:
    """Subclass per eval: class attributes describe it, `cases` and
    `run_case` do the work.

    ``run_case`` may be sync or async. An async one runs on the runner's event
    loop; a sync one runs in a worker thread, so it may block or call
    ``asyncio.run``. It returns a `CaseResult`, or a plain dict of metrics.
    The other hooks run with no event loop (`run_suite`).
    """

    name: str = ""
    description: str = ""
    # Suites that act as a user need one named on the command line
    # (`EvalCLI.bind_identity`). The rest may say why they don't.
    needs_identity: bool = False
    identity_note: str = ""
    default_model: Optional[str] = None
    default_trials: int = 1
    max_trials: Optional[int] = None
    # How many cases run at once. Above 1, the budget check counts the
    # estimates of the cases in flight.
    concurrency: int = 1
    case_estimate_usd: float = 0.0
    bars: tuple[Bar, ...] = ()
    # Metrics shown in the report's per-case table (cost and time always are).
    columns: tuple[str, ...] = ()
    # Options this suite reads -> help text.
    options: Mapping[str, str] = {}

    def cases(self, options: Mapping[str, str]) -> list[Case]:
        raise NotImplementedError

    def estimate(self, case: Case, run: EvalRun) -> float:
        """Estimated spend for one trial of ``case``: what the budget check
        compares against the remaining budget before starting it."""
        return self.case_estimate_usd if case.estimate_usd is None else case.estimate_usd

    @contextmanager
    def session(self, run: EvalRun) -> Iterator[None]:
        """Setup and teardown around all the cases (patches, servers,
        shared state on ``self``)."""
        yield

    def run_case(self, case: Case, run: EvalRun) -> Union[CaseResult, Awaitable[CaseResult]]:
        raise NotImplementedError

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        """Suite-level numbers (per-variant rates, consistency, precision).
        Keys are what bare-name bars read. Runs inside the session."""
        return {}

    def appendix(self) -> str:
        """Extra Markdown for the report (summary tables, cards). Runs
        inside the session, after `aggregate`."""
        return ""


class AgentSuite(Suite):
    """A suite over one target, graded by scorers.

    ``target`` is an `Agent` (run with each case's input and context) or any
    async callable ``(case) -> RunResult``. Each score becomes a metric of the
    same name: its verdict when it has one, else its value. A judged score
    between 0 and 1 also lands in ``<name>_score``. ``pass`` holds when no
    score failed, so ``rate(pass) >= 0.9`` reads the suite as a whole.

    The meter wraps a client handed straight to an agent target. A callable
    target's clients are its own: name them as ``"provider:model"`` so the
    provider registry meters them, or wrap them with ``Meter.wrap``.
    """

    def __init__(
        self,
        target: Target,
        cases: Iterable[Case],
        scorers: Sequence[Scorer] = (),
        *,
        name: str = "eval",
        description: str = "",
        bars: Sequence[Union[Bar, str]] = (),
        output_type: Optional[type] = None,
        concurrency: int = 4,
    ) -> None:
        self.target = target
        self._cases = list(cases)
        self.scorers = scorers
        self.name = name
        self.description = description
        self.bars = tuple(Bar.parse(bar) if isinstance(bar, str) else bar for bar in bars)
        self.output_type = output_type
        self.concurrency = concurrency

    def cases(self, options: Mapping[str, str]) -> list[Case]:
        return self._cases

    @contextmanager
    def session(self, run: EvalRun) -> Iterator[None]:
        agent = self.target
        if not isinstance(agent, Agent) or agent.model_client is None:
            yield
            return
        client = agent.model_client
        agent.model_client = run.meter.wrap(client)
        try:
            yield
        finally:
            agent.model_client = client

    async def run_case(self, case: Case, run: EvalRun) -> CaseResult:
        target = self.target
        if isinstance(target, Agent):
            result = await target.run(
                case.input,
                context=case.context,
                output_type=self.output_type or target.output_type or str,
            )
        else:
            result = await target(case)
        scores = [await call_scorer(scorer, case, result) for scorer in self.scorers]
        metrics: dict[str, Any] = {"pass": all(s.passed is not False for s in scores)}
        for score in scores:
            metrics[score.name] = score.value if score.passed is None else score.passed
            if score.passed is not None and score.value not in (0.0, 1.0):
                metrics[f"{score.name}_score"] = score.value
        return CaseResult(
            metrics=metrics,
            details={
                "output": result.output if result.output is not None else result.text,
                "tools": [record.tool for record in result.trajectory],
                "scores": [score.model_dump() for score in scores],
            },
        )
