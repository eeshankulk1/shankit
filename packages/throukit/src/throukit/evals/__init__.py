"""Evals: suites of cases, pass bars, a hard spend cap, reports and a CLI
(design §7.2)."""

from . import trajectory
from .bars import Bar, BarResult
from .budget import BudgetExceeded, CaseSpend, Meter, Price, Pricer, metered
from .cases import Case, CaseResult, load_cases
from .cli import EvalCLI
from .report import write_report
from .runner import (
    EXIT_BAR_FAILED,
    EXIT_OK,
    EXIT_STOPPED_EARLY,
    EXIT_USAGE,
    EvalReport,
    dry_run,
    evaluate,
    run_suite,
    select_cases,
)
from .scorers import Score, Scorer, exact_match, llm_judge, output_contains
from .session import CapError, SessionCap, SessionStatus
from .suite import AgentSuite, EvalRun, Suite, SuiteError

__all__ = [
    "EXIT_BAR_FAILED",
    "EXIT_OK",
    "EXIT_STOPPED_EARLY",
    "EXIT_USAGE",
    "AgentSuite",
    "Bar",
    "BarResult",
    "BudgetExceeded",
    "CapError",
    "Case",
    "CaseResult",
    "CaseSpend",
    "EvalCLI",
    "EvalReport",
    "EvalRun",
    "Meter",
    "Price",
    "Pricer",
    "Score",
    "Scorer",
    "SessionCap",
    "SessionStatus",
    "Suite",
    "SuiteError",
    "dry_run",
    "evaluate",
    "exact_match",
    "llm_judge",
    "load_cases",
    "metered",
    "output_contains",
    "run_suite",
    "select_cases",
    "trajectory",
    "write_report",
]
