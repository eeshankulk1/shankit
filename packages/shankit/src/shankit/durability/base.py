"""The checkpointer contract (design §7.1).

Durability is a protocol you back with your own storage — no platform
gravity. A checkpoint has per-thread identity, and the state it persists is
the *same* state a network shares across steps (one state concept, not two).
The natural checkpoint boundary is a router decision, which is also the
interrupt/resume point for human-in-the-loop.

State must be JSON-serializable: checkpoints are round-tripped through JSON
by the shipped stores.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

__all__ = ["InterruptInfo", "Checkpoint", "Checkpointer"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InterruptInfo(BaseModel):
    """A pending human-in-the-loop pause recorded in a checkpoint."""

    reason: str
    key: str = "approval"
    payload: Any = None


class Checkpoint(BaseModel):
    """A durable snapshot of a run at a router boundary.

    ``in_flight`` is the write-ahead marker for crash recovery: it names the
    step that was executing when the checkpoint was written. Between steps it
    is ``None``. If a process dies and leaves a ``running`` checkpoint with
    ``in_flight`` set, that step may have partially executed its side
    effects — recovery treats it as ambiguous (see ``Network.recover``).
    """

    state: dict[str, Any] = Field(default_factory=dict)
    status: Literal["running", "interrupted", "done"] = "running"
    interrupt: Optional[InterruptInfo] = None
    steps_run: list[str] = Field(default_factory=list)
    in_flight: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utcnow)


class Checkpointer(abc.ABC):
    """Implement this against your own storage to make runs durable."""

    @abc.abstractmethod
    async def save(self, thread_id: str, checkpoint: Checkpoint) -> None:
        """Persist the latest checkpoint for a thread (upsert semantics)."""

    @abc.abstractmethod
    async def load(self, thread_id: str) -> Optional[Checkpoint]:
        """The latest checkpoint for a thread, or ``None``."""

    @abc.abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Forget a thread. Deleting a missing thread is a no-op."""
