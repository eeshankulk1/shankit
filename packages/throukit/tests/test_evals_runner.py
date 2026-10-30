"""The eval runner, offline: a toy suite runs through the real runner, meter,
bars, session cap, reports and CLI with ``ScriptedModel`` answering every
provider spec, so the budget cap is exercised without a single paid call."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
from contextlib import contextmanager
from decimal import Decimal
from typing import ClassVar
from unittest import mock

import pytest
from throukit import Agent, RunResult, Usage, get_provider
from throukit.evals import (
    EXIT_BAR_FAILED,
    EXIT_OK,
    EXIT_STOPPED_EARLY,
    Bar,
    BudgetExceeded,
    CapError,
    Case,
    CaseResult,
    EvalCLI,
    EvalRun,
    Meter,
    Price,
    SessionCap,
    Suite,
    SuiteError,
    dry_run,
    evaluate,
    metered,
    run_suite,
    select_cases,
    write_report,
)
from throukit.testing import ScriptedModel, Turn, use_model

MODEL = "anthropic:toy-model"
TOY_ID = "toy-model"


def price(model: str, usage: Usage) -> Price:
    return Price(Decimal(usage.input_tokens) / 10_000_000, known=model == TOY_ID)


def spend(usd: float) -> Usage:
    """Usage that costs exactly ``usd`` at `price`."""
    return Usage(input_tokens=round(usd * 10_000_000), requests=1)


class Toy(Suite):
    """Each case makes ``calls`` model calls; the last answer decides success."""

    name = "toy"
    description = "a toy suite"
    case_estimate_usd = 0.01
    columns = ("success", "steps")
    bars = (
        Bar.parse("rate(success) >= 0.5"),
        Bar.parse("median(steps) <= 3"),
        Bar.parse("sum(unapproved) == 0"),
    )
    n = 3
    calls = 1

    def __init__(self) -> None:
        self.ran: list[str] = []

    def cases(self, options):
        return [Case(id=f"c{i}", tags={"group": "odd" if i % 2 else "even"}) for i in range(self.n)]

    async def run_case(self, case, run):
        self.ran.append(case.id)
        text = ""
        for _ in range(self.calls):
            agent = Agent(name="toy", model=MODEL, instructions="Answer yes or no.")
            text = (await agent.run(f"case {case.id}", output_type=str)).output
        return CaseResult(
            metrics={
                "success": text.startswith("yes"),
                "steps": len(text.split()),
                "unapproved": 0,
            },
            details={"text": text, "who": run.identity},
        )


class ToyCLI(EvalCLI):
    """A project's CLI: an account flag, and a secret the reports must not show."""

    identity_flags = "--account <name>"

    def add_identity_arguments(self, parser):
        parser.add_argument("--account")

    def identity_label(self, args):
        return f"account:{args.account}" if args.account else ""

    def bind_identity(self, run, args):
        if args.account == "missing":
            raise SuiteError("no such account")
        run.identity = "secret-id"
        run.redact = ("secret-id",)


class ToyAsUser(Toy):
    needs_identity = True


def toy_run(budget: float = 1.0, trials: int = 1, **fields) -> EvalRun:
    return EvalRun(
        suite="toy", meter=Meter(Decimal(str(budget)), price=price), trials=trials, **fields
    )


def go(suite: Suite, run: EvalRun):
    return run_suite(suite, run, select_cases(suite, run.options, None))


@pytest.fixture
def cap(tmp_path):
    session = SessionCap(tmp_path / "session", set_hint="cap <usd>")
    session.set(Decimal("100"))
    return session


def cli(cap=None, suite=ToyAsUser) -> ToyCLI:
    return ToyCLI({"toy": suite}, price=price, cap=cap)


RUN = ["run", "toy", "--account", "pro", "--budget", "1"]


# ── bars, spend and the report ──────────────────────────────────────────


def test_bars_pass_spend_is_metered_and_the_report_is_written(tmp_path):
    with use_model(
        ScriptedModel([Turn("yes it is", usage=spend(0.015)) for _ in range(3)])
    ) as model:
        suite = Toy()
        report = go(suite, toy_run(redact=("c1",)))
    model.assert_exhausted()

    assert [b.status for b in report.bars] == ["pass", "pass", "pass"]
    assert report.exit_code == EXIT_OK
    assert report.stopped is None
    assert [r.metrics["cost_usd"] for r in report.results] == [0.015, 0.015, 0.015]
    assert report.spend["spent_usd"] == pytest.approx(0.045)
    assert report.spend["models"][TOY_ID]["calls"] == 3
    assert report.declared_models == [TOY_ID]
    assert report.pass_rate == 1.0
    assert report.mean("steps") == 3.0

    md_path, json_path = write_report(report, tmp_path)
    data = json.loads(json_path.read_text())
    md = md_path.read_text()
    assert data["suite"] == "toy"
    assert data["status"] == "pass"
    assert data["exit_code"] == 0
    assert data["git"]["sha"]
    assert data["spent_usd"] == pytest.approx(0.045)
    assert {b["bar"]: b["status"] for b in data["bars"]} == {
        "rate(success) >= 0.5": "pass",
        "median(steps) <= 3": "pass",
        "sum(unapproved) == 0": "pass",
    }
    assert [c["case"] for c in data["cases"]] == ["c0", "<redacted>", "c2"]
    assert "## Bars" in md
    assert "c1" not in md


def test_bars_read_rates_medians_maxima_and_counts_scoped_by_tags():
    def r(success, steps, cost, commits, site="testsite"):
        return CaseResult(
            metrics={"success": success, "steps": steps, "cost_usd": cost, "commits": commits},
            tags={"site": site},
        )

    results = [
        r(True, 10, 0.20, 0),
        r(True, 30, 0.50, 0),
        r(False, 20, 0.90, 0),
        r(True, 5, 0.10, 0, "real"),
    ]
    checks = {
        "rate(success) >= 0.8": ("fail", 0.75),
        "median(steps) <= 25": ("pass", 15.0),
        "median(cost_usd) <= 0.60": ("pass", 0.35),
        "max(steps) <= 25": ("fail", 30.0),
        "sum(commits) == 0": ("pass", 0.0),
        "count(success) >= 3": ("pass", 3.0),
    }
    for expr, (status, value) in checks.items():
        got = Bar.parse(expr).check(results, {})
        assert (got.status, got.value) == (status, pytest.approx(value)), expr
    scoped = Bar.parse("rate(success) >= 0.6", site="testsite").check(results, {})
    assert (scoped.status, scoped.n, scoped.value) == ("pass", 3, pytest.approx(2 / 3))
    # An errored case (no metric) counts as a miss for a rate.
    errored = [*results, CaseResult(status="error")]
    assert Bar.parse("rate(success) >= 0.8").check(errored, {}).value == 0.6
    # A suite aggregate, and a bar with nothing to measure.
    assert Bar.parse("precision >= 0.9").check([], {"precision": 0.95}).status == "pass"
    assert Bar.parse("precision >= 0.9").check([], {}).status == "n/a"
    assert Bar.parse("rate(success) >= 0.8", site="nowhere").check(results, {}).status == "n/a"
    # A scorer's name works as a metric.
    assert Bar.parse("rate(used_tool:search) >= 1").metric == "used_tool:search"
    with pytest.raises(ValueError, match="bad bar"):
        Bar.parse("success is high")


# ── the budget cap ──────────────────────────────────────────────────────


def test_the_budget_stops_before_a_case_whose_estimate_would_cross_it():
    turns = [Turn("yes", usage=spend(0.30)), Turn("yes", usage=spend(0.30))]
    with use_model(ScriptedModel(turns)) as model:
        suite = Toy()
        suite.case_estimate_usd = 0.45
        report = go(suite, toy_run(budget=1.0))
    model.assert_exhausted()

    # 0 + .45 and .30 + .45 fit under $1; .60 + .45 does not.
    assert suite.ran == ["c0", "c1"]
    assert report.not_run == [("c2", 1)]
    assert report.stopped.startswith("not started: c2")
    assert report.exit_code == EXIT_STOPPED_EARLY
    assert report.spend["spent_usd"] == pytest.approx(0.60)


def test_the_budget_hard_stops_when_actual_spend_passes_it():
    # Estimates are far too low; the actual spend is what stops the run.
    turns = [Turn("yes", usage=spend(0.60)), Turn("yes", usage=spend(0.60))]
    with use_model(ScriptedModel(turns)) as model:
        suite = Toy()
        report = go(suite, toy_run(budget=1.0))
    model.assert_exhausted()

    assert suite.ran == ["c0", "c1"]
    assert report.stopped.startswith("hard stop after c1")
    assert report.not_run == [("c2", 1)]
    assert report.exit_code == EXIT_STOPPED_EARLY
    assert report.spend["spent_usd"] == pytest.approx(1.20)


def test_a_call_past_the_cap_is_refused_mid_case():
    # Two calls per case; the first spends past the cap, so the second is
    # never sent (the script has one turn: a second request would fail it).
    with use_model(ScriptedModel([Turn("yes", usage=spend(0.60))])) as model:
        suite = Toy()
        suite.calls = 2
        report = go(suite, toy_run(budget=0.50))
    model.assert_exhausted()

    assert len(model.requests) == 1
    [result] = report.results
    assert result.status == "budget"
    assert f"refused anthropic:{TOY_ID}" in result.error
    assert report.stopped.startswith("hard stop after c0")
    assert report.not_run == [("c1", 1), ("c2", 1)]
    assert report.exit_code == EXIT_STOPPED_EARLY
    # A budget-cut case is incomplete: listed, but not measured by the bars.
    assert {b.status for b in report.bars} == {"n/a"}


def test_the_meter_counts_vendors_and_flags_a_model_it_has_no_price_for():
    meter = Meter(Decimal("1"), price=price)
    meter.charge_usage("other-model", spend(0.10))
    meter.charge_usd(Decimal("0.05"), "search")
    assert meter.spent_usd == Decimal("0.15")
    assert meter.unpriced_models == {"other-model"}
    assert meter.summary()["vendors"] == {"search": 0.05}
    assert "other-model $0.1000 (1 calls), search $0.0500" in meter.describe()

    with pytest.raises(ValueError, match="price="):
        Meter(Decimal("1"))
    with pytest.raises(ValueError, match="finite"):
        Meter(Decimal("inf"), price=price)


async def test_metered_covers_any_registered_provider_and_puts_it_back():
    with use_model(ScriptedModel([Turn("hi", usage=spend(0.01))]), providers=("custom",)):
        inner = get_provider("custom")
        meter = Meter(Decimal("1"), price=price)
        with metered(meter):
            agent = Agent(name="a", model="custom:toy-model")
            await agent.run("hello", output_type=str)
            assert get_provider("custom") is not inner
        assert get_provider("custom") is inner
    assert meter.spent_usd == Decimal("0.01")


# ── how cases run ───────────────────────────────────────────────────────


class Blocking(Suite):
    """A sync suite: it may block and may call ``asyncio.run``."""

    name = "blocking"
    n = 2

    def __init__(self) -> None:
        self.threads: set[str] = set()

    def cases(self, options):
        return [Case(id=f"c{i}") for i in range(self.n)]

    def run_case(self, case, run):
        self.threads.add(threading.current_thread().name)
        agent = Agent(name="sync", model=MODEL)
        result = asyncio.run(agent.run("hi", output_type=str))
        return {"said": result.output, "trial": run.trial}


def test_a_sync_suite_runs_off_the_loop_and_a_dict_is_its_metrics():
    turns = [Turn("hi", usage=spend(0.01)) for _ in range(4)]
    with use_model(ScriptedModel(turns)):
        suite = Blocking()
        report = go(suite, toy_run(trials=2))
    assert [r.metrics["said"] for r in report.results] == ["hi"] * 4
    # Trial-major: every case once, then again.
    assert [(r.case_id, r.trial, r.metrics["trial"]) for r in report.results] == [
        ("c0", 1, 1),
        ("c1", 1, 1),
        ("c0", 2, 2),
        ("c1", 2, 2),
    ]
    assert [r.metrics["cost_usd"] for r in report.results] == [0.01] * 4
    assert threading.current_thread().name not in suite.threads


class Parallel(Suite):
    """Each case holds a slot for a moment, then spends ``step`` times its number."""

    name = "parallel"
    concurrency = 2
    case_estimate_usd = 0.10
    step = 0.01

    def __init__(self) -> None:
        self.peak = self.running = 0

    def cases(self, options):
        return [Case(id=f"c{i}", input=self.step * (i + 1)) for i in range(4)]

    async def run_case(self, case, run):
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(case.input / 10)
        run.meter.charge_usd(case.input, "vendor")
        self.running -= 1
        return {"ok": True}


def test_cases_run_side_by_side_each_with_its_own_cost():
    suite = Parallel()
    report = go(suite, toy_run(budget=5))
    assert suite.peak == 2
    assert [r.case_id for r in report.results] == ["c0", "c1", "c2", "c3"]
    assert [r.metrics["cost_usd"] for r in report.results] == [0.01, 0.02, 0.03, 0.04]


def test_the_budget_check_counts_the_cases_in_flight():
    suite = Parallel()
    suite.step = 0.10
    # After c0, $0.10 is spent and c1's $0.10 estimate is in flight: c2's
    # estimate on top of those crosses $0.25.
    report = go(suite, toy_run(budget=0.25))
    assert report.stopped.startswith("not started: c2")
    assert report.not_run == [("c2", 1), ("c3", 1)]


def test_a_cap_refusal_marks_every_case_it_cut_short_when_cases_overlap():
    class Racing(Parallel):
        async def run_case(self, case, run):
            await asyncio.sleep(0)
            run.meter.guard("racing")
            return {"ok": True}

    suite = Racing()
    suite.case_estimate_usd = 0.0
    report = go(suite, toy_run(budget=0))
    assert [r.status for r in report.results] == ["budget", "budget"]
    assert report.stopped.startswith("hard stop after c")


def test_ctrl_c_still_returns_the_report():
    class Stuck(Suite):
        name = "stuck"

        def cases(self, options):
            return [Case(id="done"), Case(id="stuck"), Case(id="never")]

        async def run_case(self, case, run):
            if case.id == "stuck":
                signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(30)
            return {"ok": True}

    report = go(Stuck(), toy_run())
    assert report.stopped == "interrupted"
    assert [r.case_id for r in report.results] == ["done"]
    assert report.not_run == [("stuck", 1), ("never", 1)]
    assert report.exit_code == EXIT_STOPPED_EARLY


def test_a_sync_case_cut_short_by_ctrl_c_ends_inside_the_session():
    events: list[str] = []

    class Slow(Suite):
        name = "slow"

        def cases(self, options):
            return [Case(id="slow")]

        @contextmanager
        def session(self, run):
            yield
            events.append("session closed")

        def run_case(self, case, run):
            threading.Event().wait(0.05)
            events.append("case ended")
            return {"ok": True}

    original = asyncio.wrap_future

    def interrupted(future):
        os.kill(os.getpid(), signal.SIGINT)
        return original(future)

    with mock.patch("asyncio.wrap_future", interrupted):
        report = go(Slow(), toy_run())
    assert report.stopped == "interrupted"
    assert events == ["case ended", "session closed"]


def test_the_hooks_run_with_no_event_loop():
    seen: list[str] = []

    class Hooked(Suite):
        name = "hooked"

        def cases(self, options):
            return [Case(id="a")]

        @contextmanager
        def session(self, run):
            seen.append(asyncio.run(asyncio.sleep(0, "session")))
            yield

        async def run_case(self, case, run):
            return {"ok": True}

        def aggregate(self, results):
            seen.append(asyncio.run(asyncio.sleep(0, "aggregate")))
            return {}

        def appendix(self):
            return asyncio.run(asyncio.sleep(0, "appendix"))

    report = go(Hooked(), toy_run())
    assert seen == ["session", "aggregate"]
    assert report.appendix == "appendix"


def test_a_failing_case_never_ends_the_run_and_a_failing_summary_still_reports():
    class Flaky(Suite):
        name = "flaky"

        def cases(self, options):
            return [Case(id="boom"), Case(id="fine")]

        async def run_case(self, case, run):
            if case.id == "boom":
                raise RuntimeError("model down")
            return {"ok": True}

        def aggregate(self, results):
            raise ValueError("bad summary")

    report = go(Flaky(), toy_run())
    assert [r.status for r in report.results] == ["error", "ok"]
    assert report.results[0].error == "RuntimeError: model down"
    assert report.aggregates["cases_errored"] == 1
    assert "summary failed" in report.appendix


async def test_evaluate_caps_a_client_handed_straight_to_the_agent():
    client = ScriptedModel([Turn("hi", usage=spend(0.60))])
    agent = Agent(name="a", model=TOY_ID, model_client=client)
    report = await evaluate(
        agent,
        [Case(id=f"c{i}", input="x") for i in range(3)],
        bars=["rate(pass) >= 1"],
        budget_usd=0.50,
        price=price,
        max_concurrency=1,
    )
    client.assert_exhausted()
    assert agent.model_client is client
    assert report.spend["models"][TOY_ID]["calls"] == 1
    assert report.not_run == [("c1", 1), ("c2", 1)]
    assert report.exit_code == EXIT_STOPPED_EARLY


async def test_evaluate_with_no_price_touches_no_provider():
    before = get_provider("anthropic")

    async def target(case: Case) -> RunResult:
        assert get_provider("anthropic") is before
        return RunResult(output="hi", text="hi", usage=Usage())

    report = await evaluate(target, [Case(id="c")], trials=3)
    assert len(report.results) == 3
    assert report.pass_rate == 1.0


async def test_duplicate_case_ids_and_unknown_patterns_are_refused():
    suite = Toy()
    assert [c.id for c in select_cases(suite, {}, ["c1", "c2*"])] == ["c1", "c2"]
    with pytest.raises(SuiteError, match="no case matches nope"):
        select_cases(suite, {}, ["nope"])

    async def target(case: Case) -> RunResult:
        return RunResult(output="", text="", usage=Usage())

    with pytest.raises(SuiteError, match="duplicate"):
        await evaluate(target, [Case(id="a"), Case(id="a")])


def test_a_case_marked_not_default_runs_only_when_named():
    class Mixed(Toy):
        def cases(self, options):
            return [Case(id="a"), Case(id="slow", default=False)]

    assert [c.id for c in select_cases(Mixed(), {}, None)] == ["a"]
    assert [c.id for c in select_cases(Mixed(), {}, ["slow"])] == ["slow"]


# ── the command line ────────────────────────────────────────────────────


def test_a_failing_bar_exits_nonzero_through_the_cli(cap, tmp_path, capsys):
    turns = [Turn("no, not at all today", usage=spend(0.001)) for _ in range(3)]
    with use_model(ScriptedModel(turns)) as model:
        code = cli(cap).main([*RUN, "--report-dir", str(tmp_path)])
    model.assert_exhausted()

    assert code == EXIT_BAR_FAILED
    out = capsys.readouterr().out
    assert "toy: FAIL" in out
    assert "FAIL  rate(success) >= 0.5" in out
    assert "[3/3] c2 #1 ok $0.0010" in out
    assert len(list(tmp_path.glob("*-toy.md"))) == 1
    assert len(list(tmp_path.glob("*-toy.json"))) == 1


def test_a_dry_run_calls_nothing(cap, tmp_path, capsys):
    reports = tmp_path / "reports"
    with use_model(ScriptedModel([])) as model:
        code = cli(cap).main(
            [
                "run",
                "toy",
                "--account",
                "pro",
                "--budget",
                "0.02",
                "--trials",
                "2",
                "--report-dir",
                str(reports),
                "--dry-run",
            ]
        )
    assert code == EXIT_OK
    assert model.requests == []
    assert not reports.exists()
    out = capsys.readouterr().out
    assert "DRY RUN  toy" in out
    assert "account:pro" in out
    assert "not looked up" in out
    assert "estimated spend: $0.02 of $0.02" in out
    assert "would not start (budget)" in out
    assert "session cap $100.00" in out

    suite = Toy()
    plan = dry_run(suite, toy_run(budget=5), select_cases(suite, {}, None))
    assert "estimated spend: $0.03" in plan


def test_the_cli_requires_an_identity_and_reports_a_failed_lookup(cap, capsys):
    with pytest.raises(SystemExit):
        cli(cap).main(["run", "toy", "--budget", "1"])
    assert "runs as a user: pass --account <name>" in capsys.readouterr().err

    assert cli(cap).main(["run", "toy", "--account", "missing", "--budget", "1"]) == 2
    assert "no such account" in capsys.readouterr().err


def test_the_identity_goes_on_the_run_and_stays_out_of_reports(cap, tmp_path):
    with use_model(ScriptedModel([Turn("yes") for _ in range(3)])):
        assert cli(cap).main([*RUN, "--report-dir", str(tmp_path)]) == EXIT_OK
    [json_path] = tmp_path.glob("*.json")
    data = json.loads(json_path.read_text())
    assert data["identity"] == "account:pro"
    assert data["cases"][0]["details"]["who"] == "<redacted>"
    assert "secret-id" not in json_path.read_text()


def test_a_suite_without_a_user_ignores_the_identity_flag(cap, tmp_path, capsys):
    with use_model(ScriptedModel([Turn("yes") for _ in range(3)])):
        code = cli(cap, suite=Toy).main([*RUN, "--report-dir", str(tmp_path)])
    assert code == EXIT_OK
    assert "ignoring the identity flag" in capsys.readouterr().out


def test_the_list_command_names_every_suite_and_a_suites_cases(capsys):
    cases = ToyCLI({"toy": "test_evals_runner:Toy"}, price=price)
    assert cases.main(["list"]) == 0
    assert "toy" in capsys.readouterr().out
    assert cases.main(["list", "toy"]) == 0
    out = capsys.readouterr().out
    assert "c1" in out
    assert "rate(success) >= 0.5" in out
    with pytest.raises(SystemExit):
        cases.main(["list", "nope"])


def test_options_and_trials_are_checked(cap, capsys):
    class Bounded(Toy):
        max_trials = 2
        options: ClassVar[dict[str, str]] = {"mode": "which mode"}

        def cases(self, options):
            assert options.get("mode") in (None, "fast")
            return super().cases(options)

    command = cli(cap, suite=Bounded)
    with pytest.raises(SystemExit):
        command.main(["run", "toy", "--budget", "1", "--opt", "nope=1", "--dry-run"])
    assert "has no option nope" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        command.main(["run", "toy", "--budget", "1", "--trials", "3", "--dry-run"])
    assert "runs 1..2 trials" in capsys.readouterr().err
    assert (
        command.main(["run", "toy", "--budget", "1", "--opt", "mode=fast", "--dry-run"]) == EXIT_OK
    )


def test_without_a_session_cap_the_budget_is_the_only_limit(tmp_path, capsys):
    with use_model(ScriptedModel([Turn("yes") for _ in range(3)])):
        code = cli().main([*RUN, "--report-dir", str(tmp_path)])
    assert code == EXIT_OK
    assert "session cap" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli().main(["cap", "5"])


# ── the session cap ─────────────────────────────────────────────────────


def test_a_run_needs_a_session_cap(tmp_path, capsys):
    session = SessionCap(tmp_path / "empty", set_hint="cap <usd>")
    command = cli(session)
    assert session.status() is None
    with use_model(ScriptedModel([])) as model:
        assert command.main([*RUN, "--report-dir", str(tmp_path / "r")]) == 2
        assert "no session cap" in capsys.readouterr().err
        assert model.requests == []
        assert not (tmp_path / "r").exists()

        assert command.main(["cap"]) == 0
        assert "no session cap set" in capsys.readouterr().out

        # A $0 run can spend nothing, so it asks for no cap and leaves no line
        # in the ledger.
        free = [*RUN[:-1], "0", "--report-dir", str(tmp_path / "r")]
        assert command.main(free) == EXIT_STOPPED_EARLY  # toy's cases cost: none starts
        assert "no session cap" not in capsys.readouterr().err
    assert model.requests == []
    assert not (session.directory / "ledger.jsonl").exists()


def test_runs_draw_the_session_cap_down_until_it_refuses_one(cap, tmp_path, capsys):
    command = cli(cap)
    reports = ["--report-dir", str(tmp_path)]
    assert command.main(["cap", "0.50"]) == 0
    assert "session cap $0.50: $0.00 spent, $0.50 left" in capsys.readouterr().out

    with use_model(ScriptedModel([Turn("yes", usage=spend(0.10)) for _ in range(5)])) as model:
        # The first run spends $0.30 of its $1 budget: all of it fits the session.
        assert command.main([*RUN, *reports]) == EXIT_OK
        assert "session cap $0.50: $0.30 spent, $0.20 left" in capsys.readouterr().out

        # The second asks for $1 and gets the $0.20 that is left: it stops there.
        assert command.main([*RUN, *reports]) == EXIT_STOPPED_EARLY
        out = capsys.readouterr().out
        assert "budget $0.20 (the session has less than the $1.00 asked for)" in out
        assert cap.status().spent_usd == Decimal("0.50")

        # The third is refused before anything is called.
        assert command.main([*RUN, *reports]) == 2
        assert "the session cap is used up" in capsys.readouterr().err
    model.assert_exhausted()

    # Raising the cap keeps what was spent; a new session starts the count over.
    assert cap.set(Decimal("2")).remaining_usd == Decimal("1.50")
    assert cap.set(Decimal("2"), new=True).remaining_usd == Decimal("2")


def test_runs_side_by_side_share_the_cap_and_a_dead_one_holds_only_what_it_spent(cap):
    cap.set(Decimal("5"))
    first = cap.start("toy", Decimal("3"))
    second = cap.start("toy", Decimal("3"))
    assert (first.usd, second.usd) == (Decimal("3"), Decimal("2"))
    with pytest.raises(CapError, match="used up"):
        cap.start("toy", Decimal("1"))

    # A finished run gives back what it didn't spend.
    first.spent(Decimal("0.40"))
    first.end(Decimal("0.40"))
    found = cap.status()
    assert (found.spent_usd, found.held_usd, found.running) == (Decimal("0.40"), Decimal("2"), 1)
    assert found.remaining_usd == Decimal("2.60")

    # A run that was killed never wrote its end: once its process is gone
    # it counts for what it had spent, not for its whole grant.
    second.spent(Decimal("0.25"))
    ledger = cap.directory / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace(f'"pid": {os.getpid()}', '"pid": 2147483646'))
    found = cap.status()
    assert (found.spent_usd, found.held_usd, found.running) == (Decimal("0.65"), Decimal("0"), 0)


def test_the_ledger_survives_torn_lines_and_totals_that_land_out_of_order(cap):
    cap.set(Decimal("5"))
    grant = cap.start("toy", Decimal("3"))
    grant.spent(Decimal("0.50"))
    grant.spent(Decimal("0.30"))  # a smaller total written late must not shrink the count
    with open(cap.directory / "ledger.jsonl", "a") as ledger:
        ledger.write(
            '{"event": "spend", "spent": "9"}\n{"run": "x", "spent": "nan?"}\n{"run"\n[]\n'
        )
    found = cap.status()
    assert (found.held_usd, found.running) == (Decimal("3"), 1)
    grant.end(Decimal("0.50"))
    assert cap.status().spent_usd == Decimal("0.50")


def test_the_cap_and_budget_must_be_finite(cap):
    for bad in ("nan", "inf", "-1", "five"):
        with pytest.raises(SystemExit):
            cli(cap).main(["cap", bad])
        with pytest.raises(SystemExit):
            cli(cap).main([*RUN[:-1], bad, "--dry-run"])
    with pytest.raises(ValueError, match="finite"):
        cap.set(Decimal("inf"))


def test_capped_gives_back_what_it_did_not_spend_however_it_ends(cap):
    cap.set(Decimal("2"))
    with cap.capped("script", 5, price=price) as meter:
        assert meter.budget_usd == Decimal("2")
        assert cap.status().held_usd == Decimal("2")
        meter.charge_usage(TOY_ID, spend(0.25))
    assert cap.status().spent_usd == Decimal("0.25")

    with pytest.raises(BudgetExceeded), cap.capped("script", 1, price=price) as meter:  # noqa: PT012
        meter.charge_usage(TOY_ID, spend(1))
        meter.guard("next call")
    assert cap.status().spent_usd == Decimal("1.25")

    # A $0 budget asks for nothing: no cap needed, and it refuses every call.
    empty = SessionCap(cap.directory.parent / "none")
    with empty.capped("free", 0, price=price) as meter, pytest.raises(BudgetExceeded):
        meter.guard("a paid call")
    assert empty.status() is None
    with pytest.raises(CapError, match="no session cap"), empty.capped("paid", 1, price=price):
        pass
