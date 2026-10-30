"""The eval runner (design §7.2).

A run is a suite's cases times its trials, under a budget. The runner owns
what every eval would otherwise reimplement or skip:

- budget: spend is metered from real usage as calls complete
  (`throukit.evals.budget`). A case whose estimate would take spend past the
  cap is not started; once actual spend reaches the cap, further paid calls
  are refused; either way the run stops there and the report says so.
- bars: pass/fail thresholds over the cases (`throukit.evals.bars`).
- an exit code: a failed bar outranks an early stop, so a CI job shows red.

Cases start trial-major (every case once, then again), so a run the budget
stops early still covers every case evenly.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import statistics
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from fnmatch import fnmatch
from typing import Any, Optional, Union

from .bars import Bar, BarResult
from .budget import BudgetExceeded, Meter, Pricer, metered
from .cases import Case, CaseResult
from .scorers import Scorer
from .suite import AgentSuite, EvalRun, Suite, SuiteError, Target

__all__ = [
    "EXIT_BAR_FAILED",
    "EXIT_OK",
    "EXIT_STOPPED_EARLY",
    "EXIT_USAGE",
    "EvalReport",
    "dry_run",
    "evaluate",
    "plan",
    "run_suite",
    "select_cases",
]

# Exit codes (a failed bar outranks an early stop).
EXIT_OK = 0
EXIT_BAR_FAILED = 1
EXIT_USAGE = 2
EXIT_STOPPED_EARLY = 3

Metering = Callable[[Meter], AbstractContextManager[Any]]
OnResult = Callable[[int, int, CaseResult], None]


@dataclass
class EvalReport:
    """What a run did: its results, bars, aggregates and spend. Everything
    needed to compare two runs honestly."""

    suite: str
    description: str = ""
    identity: str = "none"
    redact: tuple[str, ...] = ()
    model: Optional[str] = None
    declared_models: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)
    # Settings the run depended on, for the report's header.
    modes: dict[str, str] = field(default_factory=dict)
    trials: int = 1
    case_ids: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    results: list[CaseResult] = field(default_factory=list)
    not_run: list[tuple[str, int]] = field(default_factory=list)
    stopped: Optional[str] = None
    aggregates: dict[str, Any] = field(default_factory=dict)
    bars: list[BarResult] = field(default_factory=list)
    spend: dict[str, Any] = field(default_factory=dict)
    appendix: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    @property
    def failed_bars(self) -> list[BarResult]:
        return [b for b in self.bars if b.status == "fail"]

    @property
    def exit_code(self) -> int:
        if self.failed_bars:
            return EXIT_BAR_FAILED
        if self.stopped:
            return EXIT_STOPPED_EARLY
        return EXIT_OK

    @property
    def status(self) -> str:
        return {EXIT_OK: "pass", EXIT_BAR_FAILED: "fail", EXIT_STOPPED_EARLY: "stopped early"}[
            self.exit_code
        ]

    @property
    def pass_rate(self) -> float:
        """The share of cases that passed (`CaseResult.passed`)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def mean(self, metric: str) -> float:
        """The mean of a metric across the cases that have it."""
        values = [
            float(r.metrics[metric])
            for r in self.results
            if isinstance(r.metrics.get(metric), (int, float))
        ]
        return statistics.fmean(values) if values else 0.0

    def summary(self) -> str:
        lines = [f"{len(self.results)} case(s), pass rate {self.pass_rate:.0%}"]
        for r in self.results:
            status = {"ok": "PASS" if r.passed else "FAIL", "error": "ERROR"}.get(
                r.status, r.status.upper()
            )
            detail = r.error or ", ".join(
                f"{k}={_fmt(v)}" for k, v in r.metrics.items() if k not in _RUNNER_METRICS
            )
            lines.append(f"  [{status}] {r.case_id}: {detail}")
        lines += [
            f"  {b.status.upper()} {b.bar}" + ("" if b.value is None else f" (value {b.value:.4g})")
            for b in self.bars
        ]
        if self.stopped:
            lines.append(f"stopped early: {self.stopped}")
        return "\n".join(lines)


_RUNNER_METRICS = ("pass", "cost_usd", "seconds")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _identity_label(suite: Suite, run: EvalRun) -> str:
    return (run.identity_label if suite.needs_identity else suite.identity_note) or "none"


def select_cases(
    suite: Suite, options: Mapping[str, str], patterns: Optional[list[str]]
) -> list[Case]:
    """The suite's default cases, or those matching ``patterns`` (globs over
    case ids, e.g. ``extractor/*``)."""
    every = suite.cases(options)
    ids = [c.id for c in every]
    if len(set(ids)) != len(ids):
        raise SuiteError(f"{suite.name}: duplicate case ids")
    if not patterns:
        return [c for c in every if c.default]
    unmatched = [p for p in patterns if not any(fnmatch(c.id, p) for c in every)]
    if unmatched:
        raise SuiteError(
            f"{suite.name}: no case matches {', '.join(unmatched)}. Cases: {', '.join(ids)}"
        )
    return [c for c in every if any(fnmatch(c.id, p) for p in patterns)]


def plan(cases: list[Case], trials: int) -> list[tuple[Case, int]]:
    return [(case, trial) for trial in range(1, trials + 1) for case in cases]


async def _call_case(
    suite: Suite, case: Case, run: EvalRun, pool: ThreadPoolExecutor
) -> Union[CaseResult, dict[str, Any]]:
    if inspect.iscoroutinefunction(suite.run_case):
        return await suite.run_case(case, run)
    context = contextvars.copy_context()
    out = await asyncio.wrap_future(pool.submit(context.run, suite.run_case, case, run))
    return await out if inspect.isawaitable(out) else out


async def _run_one(
    suite: Suite, case: Case, trial: int, run: EvalRun, pool: ThreadPoolExecutor
) -> CaseResult:
    meter = run.meter
    started = time.monotonic()
    with meter.case() as spend:
        try:
            out = await _call_case(suite, case, replace(run, trial=trial), pool)
            result = out if isinstance(out, CaseResult) else CaseResult(metrics=dict(out or {}))
            if result.cost_usd:
                meter.charge_usd(result.cost_usd, f"{suite.name}:reported")
        except BudgetExceeded as e:
            result = CaseResult(status="budget", error=str(e))
        except Exception as e:  # one failed case never ends the run
            result = CaseResult(status="error", error=f"{type(e).__name__}: {e}")
            result.details["traceback"] = traceback.format_exc(limit=8)
    if spend.refused is not None:
        # The cap refused a call during this case (whatever the refusal got
        # wrapped in): its numbers are incomplete, not a measurement.
        if result.error and "refused" not in result.error:
            result.details["error"] = result.error
        result.status = "budget"
        result.error = f"budget reached: refused {spend.refused}"
    result.case_id, result.trial, result.tags = case.id, trial, dict(case.tags)
    result.metrics["cost_usd"] = round(float(spend.usd), 6)
    result.metrics["seconds"] = round(time.monotonic() - started, 2)
    return result


async def _run_cases(
    suite: Suite,
    run: EvalRun,
    work: list[tuple[Case, int]],
    pool: ThreadPoolExecutor,
    report: EvalReport,
    on_result: Optional[OnResult],
) -> None:
    meter = run.meter
    budget = meter.budget_usd
    slots = asyncio.Semaphore(suite.concurrency)
    results: dict[int, CaseResult] = {}
    in_flight: set[asyncio.Task[None]] = set()
    held = Decimal("0")  # the estimates of the cases in flight

    async def one(index: int, case: Case, trial: int, estimate: Decimal) -> None:
        nonlocal held
        try:
            result = await _run_one(suite, case, trial, run, pool)
            results[index] = result
            if on_result is not None:
                on_result(len(results), len(work), result)
            over = budget is not None and meter.spent_usd > budget
            if (result.status == "budget" or over) and report.stopped is None:
                refused = f"; refused {meter.tripped}" if meter.tripped else ""
                report.stopped = (
                    f"hard stop after {case.id} (trial {trial}): actual spend "
                    f"${meter.spent_usd:.4f} reached the ${budget:.2f} budget{refused}"
                )
        finally:
            held -= estimate
            slots.release()

    try:
        for index, (case, trial) in enumerate(work):
            await slots.acquire()
            if report.stopped is not None:
                slots.release()
                break
            estimate = Decimal(str(suite.estimate(case, run)))
            if budget is not None and meter.spent_usd + held + estimate > budget:
                report.stopped = (
                    f"not started: {case.id} (trial {trial}) is estimated at ${estimate:.2f}, "
                    f"which would take spend from ${meter.spent_usd:.4f} past the "
                    f"${budget:.2f} budget"
                )
                slots.release()
                break
            held += estimate
            task = asyncio.create_task(one(index, case, trial, estimate))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
        if in_flight:
            await asyncio.wait(set(in_flight))
    except asyncio.CancelledError:
        report.stopped = "interrupted"
        for task in in_flight:
            task.cancel()
        await asyncio.gather(*in_flight, return_exceptions=True)
        raise
    finally:
        report.results = [results[i] for i in sorted(results)]
        report.not_run = [(c.id, t) for i, (c, t) in enumerate(work) if i not in results]


def _new_report(suite: Suite, run: EvalRun, cases: list[Case]) -> EvalReport:
    if suite.concurrency < 1:
        raise ValueError("Suite.concurrency must be at least 1")
    return EvalReport(
        suite=suite.name,
        description=suite.description,
        identity=_identity_label(suite, run),
        redact=run.redact,
        model=run.model,
        declared_models=[run.model] if run.model else [],
        options=dict(run.options),
        trials=run.trials,
        case_ids=[c.id for c in cases],
        columns=list(suite.columns),
        started_at=_now(),
    )


def _pool(suite: Suite) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=suite.concurrency, thread_name_prefix="eval-case")


def _summarize(suite: Suite, report: EvalReport) -> None:
    measured = _measured(report)
    report.aggregates = {
        "cases_run": len(report.results),
        "cases_errored": sum(1 for r in report.results if r.status == "error"),
    }
    try:
        report.aggregates.update(suite.aggregate(measured))
        report.appendix = suite.appendix()
    except Exception as e:  # the spend still gets reported
        report.appendix = f"(summary failed: {type(e).__name__}: {e})"


def _measured(report: EvalReport) -> list[CaseResult]:
    # Budget-cut cases are incomplete: they are listed, not measured.
    return [r for r in report.results if r.status != "budget"]


def _finish(suite: Suite, run: EvalRun, report: EvalReport, started: float) -> EvalReport:
    report.bars = [bar.check(_measured(report), report.aggregates) for bar in suite.bars]
    report.spend = run.meter.summary()
    report.declared_models = sorted(set(report.declared_models) | set(report.spend["models"]))
    report.finished_at = _now()
    report.duration_s = round(time.monotonic() - started, 1)
    return report


def run_suite(
    suite: Suite,
    run: EvalRun,
    cases: list[Case],
    *,
    metering: Metering = metered,
    on_result: Optional[OnResult] = None,
) -> EvalReport:
    """Run ``cases`` x ``run.trials`` under the budget; evaluate the bars.

    Blocks until the run ends; from async code, call it with
    ``asyncio.to_thread``. The suite's ``session``, ``aggregate`` and
    ``appendix`` run with no event loop, so they may block or call
    ``asyncio.run``.

    Whatever happens after money starts moving (a crash in a summary, Ctrl-C),
    the report comes back, so it still gets written. ``metering`` installs the
    meter: pass your own to cover paid calls the provider registry doesn't
    see. ``on_result`` hears each case as it ends: ``(done, total, result)``.
    """
    report = _new_report(suite, run, cases)
    started = time.monotonic()
    work = plan(cases, run.trials)
    # The pool closes first: a sync case still running after Ctrl-C ends
    # inside the meter and the session.
    with metering(run.meter), suite.session(run), _pool(suite) as pool:
        try:
            asyncio.run(_run_cases(suite, run, work, pool, report, on_result))
        except KeyboardInterrupt:
            report.stopped = "interrupted"
        _summarize(suite, report)
    return _finish(suite, run, report, started)


async def _run_on_this_loop(
    suite: Suite, run: EvalRun, cases: list[Case], metering: Metering
) -> EvalReport:
    report = _new_report(suite, run, cases)
    started = time.monotonic()
    work = plan(cases, run.trials)
    with metering(run.meter), suite.session(run), _pool(suite) as pool:
        await _run_cases(suite, run, work, pool, report, None)
        _summarize(suite, report)
    return _finish(suite, run, report, started)


def dry_run(suite: Suite, run: EvalRun, cases: list[Case]) -> str:
    """The plan and its estimated cost, without calling anything."""
    budget = run.meter.budget_usd
    total = Decimal("0")
    lines = [
        f"DRY RUN  {suite.name}: {suite.description}",
        f"identity: {_identity_label(suite, run)}"
        + (
            "  (not looked up in a dry run)"
            if run.identity is None and suite.needs_identity
            else ""
        ),
        f"model: {run.model or 'n/a'}  trials: {run.trials}  "
        f"budget: {'none' if budget is None else f'${budget:.2f}'}",
        "",
        f"{'case':<44} {'trial':>5} {'estimate':>9}  plan",
    ]
    stopped = False
    for case, trial in plan(cases, run.trials):
        estimate = Decimal(str(suite.estimate(case, run)))
        if not stopped and budget is not None and total + estimate > budget:
            stopped = True
        if not stopped:
            total += estimate
        lines.append(
            f"{case.id:<44} {trial:>5} {'$' + format(estimate, '.3f'):>9}  "
            + ("would not start (budget)" if stopped else "run")
        )
    lines += [
        "",
        f"estimated spend: ${total:.2f}" + (f" of ${budget:.2f}" if budget is not None else ""),
    ]
    if stopped:
        lines.append("the budget would stop this run early; raise --budget or narrow --cases")
    if suite.bars:
        lines += ["", "bars:"] + [f"  {bar.text}" for bar in suite.bars]
    return "\n".join(lines)


async def evaluate(
    target: Target,
    cases: Iterable[Case],
    scorers: Sequence[Scorer] = (),
    *,
    name: str = "eval",
    bars: Sequence[Union[Bar, str]] = (),
    trials: int = 1,
    budget_usd: Optional[float] = None,
    price: Optional[Pricer] = None,
    output_type: Optional[type] = None,
    max_concurrency: int = 4,
) -> EvalReport:
    """Run every case against ``target`` and score the results: an `AgentSuite`
    in one call, on the caller's event loop.

    ``target`` is an `Agent` (run with each case's input and context) or any
    async callable ``(case) -> RunResult``. A case whose run raises is
    reported as an error, not a crash of the whole evaluation. With no
    ``price``, spend is not counted and no provider is touched; a budget
    needs one.
    """
    suite = AgentSuite(
        target,
        cases,
        scorers,
        name=name,
        bars=bars,
        output_type=output_type,
        concurrency=max_concurrency,
    )
    meter = Meter(None if budget_usd is None else Decimal(str(budget_usd)), price=price)
    run = EvalRun(suite=name, meter=meter, trials=trials)
    metering = metered if price is not None else nullcontext
    return await _run_on_this_loop(suite, run, select_cases(suite, {}, None), metering)
