"""The eval command line: list the suites, or run one under a budget.

A project writes a few lines of its own and gets the whole CLI::

    # evals.py
    import sys
    from throukit.evals import EvalCLI, SessionCap

    cli = EvalCLI({"refunds": "myapp.evals:Refunds"}, price=my_price,
                  cap=SessionCap(".evals/session", set_hint="python evals.py cap <usd>"),
                  report_dir=".evals/reports")
    sys.exit(cli.main())

Then ``python evals.py list``, ``python evals.py cap 5``, and
``python evals.py run refunds --budget 2 --dry-run``. ``--budget`` caps one
run; with a `SessionCap`, ``cap <usd>`` also caps every run together, and a
run gets the smaller of its budget and what the session has left. A
``--budget 0`` run can spend nothing and needs no cap. There is no default
budget.

Exit codes: 0 every bar passed; 1 a bar failed; 2 usage or setup error; 3 the
budget (or Ctrl-C) stopped the run early.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union

from .budget import Meter, Pricer, metered
from .report import write_report
from .runner import EXIT_USAGE, EvalReport, Metering, dry_run, run_suite, select_cases
from .session import CapError, SessionCap
from .suite import EvalRun, Suite, SuiteError

__all__ = ["EvalCLI"]

SuiteSpec = Union[str, type[Suite]]


def _usd(text: str) -> Decimal:
    """An amount on the command line: finite, not negative."""
    try:
        amount = Decimal(text)
    except ArithmeticError:
        amount = Decimal("nan")
    if not amount.is_finite() or amount < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite amount >= 0")
    return amount


def _parse_opts(pairs: list[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            parser.error(f"--opt takes key=value, got {pair!r}")
        out[key.strip()] = value
    return out


def _print_case(columns: tuple[str, ...]):
    def show(done: int, total: int, result: Any) -> None:
        cols = " ".join(f"{c}={_fmt(result.metrics[c])}" for c in columns if c in result.metrics)
        status = result.status.upper() if result.status != "ok" else "ok"
        line = (
            f"[{done}/{total}] {result.case_id} #{result.trial} {status} "
            f"${result.metrics['cost_usd']:.4f} {result.metrics['seconds']:.1f}s {cols}"
        ).rstrip()
        print(line, flush=True)
        if result.error:
            print(f"    {result.error[:300]}", flush=True)

    return show


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class EvalCLI:
    """``list``, ``cap`` and ``run`` over a project's suites.

    Args:
        suites: Name -> a `Suite` class, or ``"module:Class"`` imported on
            demand so ``list`` and a one-suite run only load what they need.
        price: Prices model calls for the meter.
        metering: Installs the meter around a run; see `run_suite`.
        cap: The session cap. Without one there is no ``cap`` command, and
            ``--budget`` is the only limit.
        report_dir: Where reports land, unless ``--report-dir`` says otherwise.
        prog: The command's name in its help text.

    Subclass to name who a run acts as: `add_identity_arguments`,
    `identity_label` and `bind_identity`. ``run_type`` is the `EvalRun`
    subclass suites get.
    """

    run_type: type[EvalRun] = EvalRun
    # How the usage error names the identity flags.
    identity_flags = "--identity <value>"

    def __init__(
        self,
        suites: Mapping[str, SuiteSpec],
        *,
        price: Pricer,
        metering: Metering = metered,
        cap: Optional[SessionCap] = None,
        report_dir: Union[str, Path] = "evals/reports",
        prog: Optional[str] = None,
    ) -> None:
        self.suites = suites
        self.price = price
        self.metering = metering
        self.cap = cap
        self.report_dir = report_dir
        self.prog = prog

    # Hooks.

    def add_identity_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the flags that name who a run acts as."""
        parser.add_argument("--identity", help="who the run acts as")

    def identity_label(self, args: argparse.Namespace) -> str:
        """The report's name for the identity the flags ask for, before any
        lookup; empty when they ask for none."""
        return args.identity or ""

    def bind_identity(self, run: EvalRun, args: argparse.Namespace) -> None:
        """Look the identity up and set it on ``run`` (``identity``,
        ``identity_label``, ``redact``). Raise `SuiteError` when it can't be
        found. Not called on a dry run."""
        run.identity = args.identity

    def write_report(self, report: EvalReport, directory: Union[str, Path]) -> tuple[Path, Path]:
        return write_report(report, directory)

    # Commands.

    def load(self, name: str) -> Suite:
        """A fresh instance of the named suite (suites keep per-run state)."""
        try:
            target = self.suites[name]
        except KeyError:
            raise KeyError(f"unknown suite {name!r}; suites: {', '.join(self.suites)}") from None
        if isinstance(target, str):
            module, _, cls = target.partition(":")
            target = getattr(import_module(module), cls)
        return target()

    def _list(self, name: Optional[str]) -> int:
        if name is None:
            print(
                f"{'suite':<24} {'cases':>9} {'est/case':>9} {'trials':>6}  identity  description"
            )
            for suite_name in self.suites:
                suite = self.load(suite_name)
                every = suite.cases({})
                default = sum(1 for c in every if c.default)
                identity = "required" if suite.needs_identity else "none"
                estimate = f"${suite.case_estimate_usd:.2f}"
                print(
                    f"{suite_name:<24} {default:>4}/{len(every):<4} {estimate:>9} "
                    f"{suite.default_trials:>6}  {identity:<8}  {suite.description}"
                )
            print("\n`list <suite>` shows a suite's cases, bars and options.")
            return 0
        suite = self.load(name)
        print(f"{suite.name}: {suite.description}")
        print(f"identity: {'required' if suite.needs_identity else suite.identity_note}")
        print(
            f"default model: {suite.default_model or 'n/a'}  default trials: {suite.default_trials}"
        )
        print(f"\n{'case':<48} {'default':>7} {'estimate':>9}")
        for case in suite.cases({}):
            estimate = suite.case_estimate_usd if case.estimate_usd is None else case.estimate_usd
            print(
                f"{case.id:<48} {'yes' if case.default else 'no':>7} {'$' + format(estimate, '.3f'):>9}"
            )
        if suite.bars:
            print("\nbars:")
            for bar in suite.bars:
                print(f"  {bar.text}")
        if suite.options:
            print("\noptions (--opt key=value):")
            for key, help_text in suite.options.items():
                print(f"  {key}: {help_text}")
        return 0

    def _session_line(self) -> Optional[str]:
        if self.cap is None:
            return None
        found = self.cap.status()
        return found.describe() if found else f"no session cap set (`{self.cap.set_hint}`)"

    def _say_session(self) -> None:
        line = self._session_line()
        if line:
            print(line)

    def _cap(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
        assert self.cap is not None
        if args.usd is None:
            if args.new:
                parser.error("--new starts a session over: give its cap, e.g. `cap 5 --new`")
            self._say_session()
            return 0
        print(self.cap.set(args.usd, new=args.new).describe())
        return 0

    def _run(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
        try:
            suite = self.load(args.suite)
        except KeyError as e:
            parser.error(str(e.args[0]))
        options = _parse_opts(args.opt, parser)
        unknown = sorted(set(options) - set(suite.options))
        if unknown:
            parser.error(
                f"{suite.name} has no option {', '.join(unknown)} (see `list {suite.name}`)"
            )
        trials = args.trials or suite.default_trials
        if trials < 1 or (suite.max_trials and trials > suite.max_trials):
            parser.error(f"{suite.name} runs 1..{suite.max_trials or 'N'} trials")
        label = self.identity_label(args)
        if suite.needs_identity and not label:
            parser.error(f"{suite.name} runs as a user: pass {self.identity_flags}")
        if not suite.needs_identity and label:
            print(
                f"note: {suite.name} runs without a user ({suite.identity_note}); "
                "ignoring the identity flag"
            )

        patterns = [p.strip() for chunk in args.cases for p in chunk.split(",") if p.strip()]
        try:
            cases = select_cases(suite, options, patterns)
        except SuiteError as e:
            parser.error(str(e))
        if not cases:
            parser.error(f"{suite.name}: no cases selected")

        run = self.run_type(
            suite=suite.name,
            meter=Meter(args.budget, price=self.price),
            identity_label=label,
            model=args.model or suite.default_model,
            options=options,
            trials=trials,
        )
        if args.dry_run:
            print(dry_run(suite, run, cases))
            self._say_session()
            return 0
        if suite.needs_identity:
            try:
                self.bind_identity(run, args)
            except SuiteError as e:
                print(f"eval: {e}", file=sys.stderr)
                return EXIT_USAGE
        # The session's cap, last: nothing is held for a run that can't start.
        # A $0 run draws on no session: its meter refuses every paid call.
        grant = None
        if self.cap is not None and args.budget > 0:
            try:
                grant = self.cap.start(suite.name, args.budget)
            except CapError as e:
                print(f"eval: {e}", file=sys.stderr)
                return EXIT_USAGE
            run.meter = Meter(grant.usd, price=self.price, on_spend=grant.spent)

        # Everything after the grant is inside the try: whatever raises or
        # returns, the grant is given back.
        try:
            budget = run.meter.budget_usd
            assert budget is not None
            shorter = (
                f" (the session has less than the ${args.budget:.2f} asked for)"
                if budget < args.budget
                else ""
            )
            who = run.identity_label if suite.needs_identity else suite.identity_note
            print(
                f"eval {suite.name}: {len(cases)} case(s) x {trials} trial(s), "
                f"budget ${budget:.2f}{shorter}, identity {who}"
            )
            report = run_suite(
                suite, run, cases, metering=self.metering, on_result=_print_case(suite.columns)
            )
        except SuiteError as e:
            print(f"eval: {e}", file=sys.stderr)
            print(f"eval: {run.meter.describe()}", file=sys.stderr)
            return EXIT_USAGE
        finally:
            if grant is not None:
                grant.end(run.meter.spent_usd)
        md_path, json_path = self.write_report(report, args.report_dir or self.report_dir)

        print(f"\n{suite.name}: {report.status.upper()}  ({run.meter.describe()})")
        self._say_session()
        if report.stopped:
            print(f"stopped early: {report.stopped}")
        for bar in report.bars:
            value = "n/a" if bar.value is None else f"{bar.value:.4g}"
            print(f"  {bar.status.upper():<5} {bar.bar}  (value {value}, n={bar.n})")
        print(f"report: {md_path}\n        {json_path}")
        return report.exit_code

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=self.prog, description="List the eval suites, or run one under a budget."
        )
        sub = parser.add_subparsers(dest="command", required=True)

        p_list = sub.add_parser("list", help="list suites, or one suite's cases")
        p_list.add_argument("suite", nargs="?")

        if self.cap is not None:
            p_cap = sub.add_parser("cap", help="set, or show, what this session may spend on evals")
            p_cap.add_argument("usd", nargs="?", type=_usd, help="the cap in USD (omit to show it)")
            p_cap.add_argument(
                "--new", action="store_true", help="a new session: the spend count starts over"
            )

        p_run = sub.add_parser("run", help="run one suite under a budget")
        p_run.add_argument("suite")
        self.add_identity_arguments(p_run)
        p_run.add_argument("--budget", type=_usd, required=True, help="hard spend cap in USD")
        p_run.add_argument(
            "--trials", type=int, default=None, help="runs per case (default: the suite's)"
        )
        p_run.add_argument(
            "--cases", action="append", default=[], help="comma-separated globs over case ids"
        )
        p_run.add_argument("--model", default=None, help="model id for suites that take one")
        p_run.add_argument(
            "--opt", action="append", default=[], help="suite option key=value (repeatable)"
        )
        p_run.add_argument(
            "--dry-run", action="store_true", help="print the plan and estimate; call nothing"
        )
        p_run.add_argument("--report-dir", default=None, help=f"default: {self.report_dir}")
        return parser

    def main(self, argv: Optional[list[str]] = None) -> int:
        parser = self.parser()
        args = parser.parse_args(argv)
        if args.command == "list":
            try:
                return self._list(args.suite)
            except KeyError as e:
                parser.error(str(e.args[0]))
        if args.command == "cap":
            return self._cap(args, parser)
        return self._run(args, parser)
