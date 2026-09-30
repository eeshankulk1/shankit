"""Run control: steering a running agent from outside its loop.

A :class:`RunControl` passed to :meth:`Agent.run` / :meth:`Agent.stream` /
:meth:`Agent.resume` lets code outside the run act on it while it works,
always at the loop's own boundaries (cooperative, never mid-call):

- **cancel** - the run ends at its next boundary, before another model
  pass or before the tools of a pass run, with what it has
  (``DoneEvent.stopped == "cancelled"``);
- **send** - a message delivered to the model at its next tool boundary
  (appended to that pass's tool results), or, if the run was about to
  finish, as one more user turn;
- **budget** - soft limits (passes, wall clock, cost) the run winds down
  at: one last pass with tools off to report what it has, then it ends
  with ``stopped == "budget"``;
- **refuse** - a per-run veto on tool calls, decided at call time. The
  tool list the model sees stays the same (so a prompt cache shared with
  another run of the same agent survives); a refused call comes back as an
  error result carrying the veto's text;
- **checkpoint** - called with the run's transcript at every boundary, so
  a crashed run can continue from it (:meth:`Agent.resume`);
- **worker / depth** - who this run is when it is one of several (a clone,
  see :class:`~shankit.CloneToolSource`); ``depth`` is what the spawn tool's
  depth guard reads;
- **parent** - the control of the run that spawned this one: cancelling it
  cancels this run too.

``cancel`` and ``send`` are safe to call from any thread.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

from .messages import Message
from .usage import Usage

__all__ = ["Budget", "RunControl"]

#: A per-run veto: ``(tool name, arguments) -> refusal text``, or ``None`` to
#: let the call run.
Refusal = Callable[[str, dict[str, Any]], Optional[str]]

#: Called with the run's transcript so far (every message the run added,
#: starting with its prompt). May be async.
CheckpointHook = Callable[[list[Message]], Union[None, Awaitable[None]]]

DEFAULT_BUDGET_NOTICE = (
    "[Budget reached: stop working now. Reply with your report from what you "
    "have, and say what's unfinished.]"
)


@dataclass
class Budget:
    """Soft limits a run winds down at, checked at each pass boundary.

    Reaching one doesn't raise (unlike ``Agent(max_iterations=, timeout_s=)``,
    the hard bounds): the model gets :attr:`notice` and one last pass with
    tools off, and the run ends with ``stopped == "budget"`` and whatever
    it wrote in that pass.

    Args:
        max_passes: Model passes, counting the wind-down pass. A resumed
            run counts the passes already in its transcript.
        timeout_s: Wall clock since this run (or resume) started.
        max_cost: Spend ceiling in your own unit, priced by ``cost`` from
            the run's cumulative usage (plus ``spent`` for a resumed run).
        cost: Prices a :class:`Usage`. Required with ``max_cost``.
        spent: Cost already spent before this run started (a resume).
        notice: What the model is told when the budget is reached.
    """

    max_passes: Optional[int] = None
    timeout_s: Optional[float] = None
    max_cost: Optional[float] = None
    cost: Optional[Callable[[Usage], float]] = None
    spent: float = 0.0
    notice: str = DEFAULT_BUDGET_NOTICE

    def __post_init__(self) -> None:
        if self.max_cost is not None and self.cost is None:
            raise ValueError("Budget(max_cost=...) needs cost= to price usage")
        if self.max_passes is not None and self.max_passes < 1:
            raise ValueError("Budget.max_passes must be at least 1")

    def reached(self, *, passes: int, elapsed_s: float, usage: Usage) -> Optional[str]:
        """Which limit the next pass would cross (``"passes"``, ``"time"``,
        ``"cost"``), or ``None``. ``passes`` is the passes already run."""
        if self.max_passes is not None and passes + 1 >= self.max_passes:
            return "passes"
        if self.timeout_s is not None and elapsed_s >= self.timeout_s:
            return "time"
        if (
            self.max_cost is not None
            and self.cost is not None
            and self.spent + self.cost(usage) >= self.max_cost
        ):
            return "cost"
        return None


class RunControl:
    """Cooperative control of one run from outside it (see the module doc).

    Args:
        worker: This run's id among several (a clone's). Whoever relays its
            steps attributes them with it (``StepEvent.worker``).
        depth: How many spawns deep this run is (0: a top-level run).
        budget: Soft limits the run winds down at.
        refuse: A call-time veto on tool calls.
        checkpoint: Called with the transcript at every boundary.
        parent: The control of the run that spawned this one; cancelling
            it cancels this run too (a clone stops with its parent).
    """

    def __init__(
        self,
        *,
        worker: Optional[str] = None,
        depth: int = 0,
        budget: Optional[Budget] = None,
        refuse: Optional[Refusal] = None,
        checkpoint: Optional[CheckpointHook] = None,
        parent: Optional[RunControl] = None,
    ) -> None:
        self.worker = worker
        self.depth = depth
        self.budget = budget
        self.refuse = refuse
        self.checkpoint = checkpoint
        self._parent = parent
        self._lock = threading.Lock()
        self._inbox: list[str] = []
        self._cancelled = False

    def __repr__(self) -> str:
        return f"RunControl(worker={self.worker!r}, depth={self.depth}, cancelled={self.cancelled})"

    def cancel(self) -> None:
        """End the run at its next boundary. Idempotent."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled or (self._parent is not None and self._parent.cancelled)

    def send(self, text: str) -> None:
        """Deliver ``text`` to the model at the run's next tool boundary."""
        if text:
            with self._lock:
                self._inbox.append(text)

    @property
    def has_messages(self) -> bool:
        with self._lock:
            return bool(self._inbox)

    def drain(self) -> list[str]:
        """Take every undelivered message (the loop calls this)."""
        with self._lock:
            taken, self._inbox = self._inbox, []
        return taken
