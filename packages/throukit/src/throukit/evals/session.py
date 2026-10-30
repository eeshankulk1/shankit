"""The session cap: one amount for a piece of work, drawn down by every run
until it is gone (design §7.2).

A budget caps one run. Nothing caps a session: an agent tuning a prompt can
start thirty runs under sensible budgets and spend far more than anyone meant
to. So a run can also draw on a `SessionCap`, set once by whoever owns the
spend. A run is granted ``min(budget, what's left)`` and refused when nothing
is left.

A session is a folder holding the cap and a ledger; point two checkouts at one
folder and they draw on one cap. Runs in parallel share it safely: a run holds
its whole grant from the start (written under a file lock), records what it
has spent after every paid call, and gives back the rest when it ends. A run
that died holds only what it had spent.

POSIX only: it needs ``fcntl`` file locks and a way to probe a process id.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

from .budget import Meter, Number, Pricer

__all__ = ["CapError", "Grant", "SessionCap", "SessionStatus"]

_CAP, _LEDGER, _LOCK = "cap.json", "ledger.jsonl", "lock"


class CapError(Exception):
    """A paid run can't start: no session cap, or nothing left of it."""


@dataclass(frozen=True)
class SessionStatus:
    cap_usd: Decimal
    # What finished runs spent, and what dead ones had spent when they died.
    spent_usd: Decimal
    # What the runs going now may still spend: their grants.
    held_usd: Decimal
    running: int

    @property
    def remaining_usd(self) -> Decimal:
        return self.cap_usd - self.spent_usd - self.held_usd

    def describe(self) -> str:
        held = f", ${self.held_usd:.2f} held by {self.running} running" if self.running else ""
        return (
            f"session cap ${self.cap_usd:.2f}: ${self.spent_usd:.2f} spent{held}, "
            f"${max(self.remaining_usd, Decimal(0)):.2f} left"
        )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class _Run:
    """One run's ledger lines, folded."""

    spent: Decimal = Decimal(0)
    held: Decimal = Decimal(0)
    pid: int = 0  # 0 until its start line is read
    ended: bool = False

    @property
    def running(self) -> bool:
        return not self.ended and bool(self.pid) and _alive(self.pid)


class Grant:
    """What one run may spend, held against the session until it ends."""

    def __init__(self, folder: Path, run: str, usd: Decimal) -> None:
        self.folder, self.run, self.usd = folder, run, usd

    def spent(self, total: Decimal) -> None:
        """The run's spend so far (``Meter.on_spend``)."""
        _append(self.folder, {"run": self.run, "event": "spend", "spent": str(total)})

    def end(self, total: Decimal) -> None:
        _append(self.folder, {"run": self.run, "event": "end", "spent": str(total)})


def _append(folder: Path, entry: dict[str, Any]) -> None:
    # One short line per write, in append mode: whole lines even when
    # several runs write at once.
    with open(folder / _LEDGER, "a") as ledger:
        ledger.write(json.dumps(entry) + "\n")


class SessionCap:
    """A session's cap, in ``directory``.

    Args:
        directory: Holds the cap and the ledger.
        set_hint: The command that sets a cap, for the refusal messages.
    """

    def __init__(self, directory: Union[str, Path], *, set_hint: str = "SessionCap.set") -> None:
        if os.name != "posix":
            raise NotImplementedError("SessionCap needs POSIX file locks")
        self.directory = Path(directory)
        self.set_hint = set_hint

    @contextmanager
    def _locked(self) -> Iterator[None]:
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / _LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> Optional[SessionStatus]:
        try:
            raw = json.loads((self.directory / _CAP).read_text())
            cap = Decimal(str(raw["cap_usd"]))
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
            return None
        runs: dict[str, _Run] = defaultdict(_Run)
        try:
            lines = (self.directory / _LEDGER).read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            # A torn or foreign line is skipped, never fatal: the ledger is
            # appended to outside the lock.
            try:
                entry = json.loads(line)
                run = runs[entry["run"]]
                if "held" in entry:
                    run.held, run.pid = Decimal(entry["held"]), int(entry["pid"])
                if "spent" in entry:
                    # Totals can land out of order (the meter reports outside
                    # its lock) and only grow: the largest is the truth.
                    run.spent = max(run.spent, Decimal(entry["spent"]))
                run.ended = run.ended or entry.get("event") == "end"
            except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError):
                continue
        spent = held = Decimal(0)
        running = 0
        for run in runs.values():
            if run.running:
                held += max(run.held, run.spent)
                running += 1
            else:
                spent += run.spent
        return SessionStatus(cap, spent, held, running)

    def status(self) -> Optional[SessionStatus]:
        """The session's cap and spend, or ``None`` when no cap is set."""
        if not (self.directory / _CAP).exists():
            return None
        with self._locked():
            return self._read()

    def set(self, usd: Decimal, *, new: bool = False) -> SessionStatus:
        """Set the cap. Spend so far still counts, unless ``new`` starts the
        session over (the old ledger is kept beside it, renamed)."""
        if not usd.is_finite() or usd < 0:
            raise ValueError("the cap must be a finite amount >= 0")
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with self._locked():
            if new and (self.directory / _LEDGER).exists():
                (self.directory / _LEDGER).rename(self.directory / f"ledger-{now}.jsonl")
            cap = json.dumps({"cap_usd": str(usd), "set_at": now}) + "\n"
            (self.directory / _CAP).write_text(cap)
            found = self._read()
        assert found is not None
        return found

    def start(self, label: str, wanted_usd: Decimal) -> Grant:
        """Hold ``min(wanted_usd, what's left)`` for a run; `CapError` when
        no cap is set or none of it is left."""
        with self._locked():
            found = self._read()
            if found is None:
                raise CapError(
                    f"no session cap. Paid runs draw from one: set it first with "
                    f"`{self.set_hint}` (the owner's number; there is no default)."
                )
            if found.remaining_usd <= 0:
                raise CapError(
                    f"the session cap is used up ({found.describe()}). Raising it is the "
                    f"owner's call: ask, then `{self.set_hint}`."
                )
            grant = Grant(self.directory, uuid4().hex[:12], min(wanted_usd, found.remaining_usd))
            at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _append(
                self.directory,
                {
                    "run": grant.run,
                    "event": "start",
                    "suite": label,
                    "pid": os.getpid(),
                    "held": str(grant.usd),
                    "at": at,
                },
            )
        return grant

    @contextmanager
    def capped(self, label: str, budget_usd: Number, *, price: Pricer) -> Iterator[Meter]:
        """A meter for one paid run, drawn from the session: it may spend
        ``min(budget_usd, what's left)``, and what it didn't spend goes back
        when the block ends, however it ends. `CapError` before anything is
        spent when no cap is set or none of it is left. A $0 budget asks for
        nothing: its meter refuses every paid call."""
        budget = Decimal(str(budget_usd))
        if budget.is_nan():
            raise ValueError("the budget must be a number")
        if budget <= 0:
            yield Meter(budget, price=price)
            return
        grant = self.start(label, budget)
        meter = Meter(grant.usd, price=price, on_spend=grant.spent)
        try:
            yield meter
        finally:
            grant.end(meter.spent_usd)
