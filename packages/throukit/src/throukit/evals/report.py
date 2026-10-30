"""Eval reports: one Markdown file to read and one JSON file to diff.

Both are named ``<UTC time>-<suite>``. A report carries what's needed to
compare two runs honestly: the suite, who it ran as, the git SHA, the models
actually called, per-case metrics, aggregates, every bar's verdict, and spend
against the budget (with the reason when the budget stopped the run early).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .runner import EvalReport

__all__ = ["git_state", "to_json", "to_markdown", "write_report"]


def git_state(repo: Optional[Path] = None) -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=5, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"sha": sha or "unknown", "dirty": bool(status) if status is not None else None}


def to_json(report: EvalReport, *, repo: Optional[Path] = None) -> dict[str, Any]:
    return {
        "suite": report.suite,
        "description": report.description,
        "status": report.status,
        "exit_code": report.exit_code,
        "identity": report.identity,
        "git": git_state(repo),
        "modes": report.modes,
        "model": report.model,
        "models_called": report.declared_models,
        "options": report.options,
        "trials": report.trials,
        "cases_selected": report.case_ids,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "duration_s": report.duration_s,
        "budget_usd": report.spend.get("budget_usd"),
        "spent_usd": report.spend.get("spent_usd"),
        "spend": report.spend,
        "stopped": report.stopped,
        "not_run": [{"case": c, "trial": t} for c, t in report.not_run],
        "bars": [asdict(b) for b in report.bars],
        "aggregates": report.aggregates,
        "cases": [
            {
                "case": r.case_id,
                "trial": r.trial,
                "status": r.status,
                "error": r.error,
                "tags": r.tags,
                "metrics": r.metrics,
                "details": r.details,
            }
            for r in report.results
        ],
        "appendix": report.appendix,
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(report: EvalReport, data: dict[str, Any]) -> str:
    spend = report.spend
    git = data["git"]
    budget = spend.get("budget_usd")
    spent = f"${spend.get('spent_usd', 0):.4f}"
    lines = [
        f"# Eval: {report.suite} - {report.status.upper()}",
        "",
        report.description,
        "",
        "| | |",
        "|---|---|",
        f"| identity | {report.identity} |",
        f"| git | `{git['sha'][:12]}`{' (dirty)' if git.get('dirty') else ''} |",
        f"| models called | {', '.join(report.declared_models) or 'none'} |",
        "| modes | " + ", ".join(f"{k}={v}" for k, v in report.modes.items()) + " |",
        f"| trials | {report.trials} |",
        f"| options | {', '.join(f'{k}={v}' for k, v in report.options.items()) or 'none'} |",
        f"| started | {report.started_at} ({report.duration_s}s) |",
        f"| spend | {spent}" + (f" of ${budget:.2f} budget |" if budget is not None else " |"),
        "",
    ]
    if report.stopped:
        lines += [f"**Stopped early:** {report.stopped}.", ""]
        if report.not_run:
            shown = ", ".join(f"{c} #{t}" for c, t in report.not_run[:20])
            more = f" (+{len(report.not_run) - 20} more)" if len(report.not_run) > 20 else ""
            lines += [f"Not run: {shown}{more}", ""]
    if spend.get("unpriced_models"):
        lines += ["No list price for: " + ", ".join(spend["unpriced_models"]), ""]
    if report.bars:
        lines += ["## Bars", "", "| bar | value | n | result |", "|---|---|---|---|"]
        for b in report.bars:
            lines.append(f"| `{b.bar}` | {_cell(b.value)} | {b.n} | {b.status.upper()} |")
        lines.append("")
    lines += [
        "## Spend",
        "",
        "| model / vendor | calls | in | out | cache read | cost |",
        "|---|---|---|---|---|---|",
    ]
    for mid, row in spend.get("models", {}).items():
        lines.append(
            f"| {mid} | {row['calls']} | {row['input_tokens']:,} | {row['output_tokens']:,} | "
            f"{row['cache_read_tokens']:,} | ${row['cost_usd']:.4f} |"
        )
    for name, cost in spend.get("vendors", {}).items():
        lines.append(f"| {name} | | | | | ${cost:.4f} |")
    lines.append("")
    if report.aggregates:
        lines += ["## Aggregates", "", "| name | value |", "|---|---|"]
        lines += [f"| {k} | {_cell(v)} |" for k, v in report.aggregates.items()]
        lines.append("")
    if report.results:
        cols = [*report.columns, "cost_usd", "seconds"]
        lines += [
            "## Cases",
            "",
            "| case | trial | status | " + " | ".join(cols) + " |",
            "|---|---|---|" + "---|" * len(cols),
        ]
        for r in report.results:
            lines.append(
                f"| {r.case_id} | {r.trial} | {r.status} | "
                + " | ".join(_cell(r.metrics.get(c)) for c in cols)
                + " |"
            )
        lines.append("")
        errors = [r for r in report.results if r.error]
        if errors:
            lines += ["## Errors", ""]
            lines += [f"- {r.case_id} #{r.trial}: {_cell((r.error or '')[:500])}" for r in errors]
            lines.append("")
    if report.appendix:
        lines += ["## Details", "", report.appendix.rstrip(), ""]
    return "\n".join(lines)


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:  # replacing "" would put the marker between every character
            text = text.replace(secret, "<redacted>")
    return text


def write_report(
    report: EvalReport, directory: Union[str, Path], *, repo: Optional[Path] = None
) -> tuple[Path, Path]:
    """Write ``<time>-<suite>.md`` and ``.json``; returns their paths. ``repo``
    is the checkout whose git SHA the report records (default: the working
    directory)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}-{report.suite.replace(':', '-').replace('/', '-')}"
    data = to_json(report, repo=repo)
    # Whatever a suite put in its details, a redacted string never lands in
    # a report.
    json_text = _redact(json.dumps(data, indent=1, default=str), report.redact)
    md_text = _redact(to_markdown(report, data), report.redact)
    md_path, json_path = directory / f"{stem}.md", directory / f"{stem}.json"
    md_path.write_text(md_text)
    json_path.write_text(json_text)
    return md_path, json_path
