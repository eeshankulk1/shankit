"""Shipped checkpointer implementations: in-memory and SQLite (design §7.1)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Optional, Union

from .base import Checkpoint, Checkpointer

__all__ = ["InMemoryCheckpointer", "SqliteCheckpointer"]


class InMemoryCheckpointer(Checkpointer):
    """Process-local checkpoints — for tests and non-durable defaults."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}

    async def save(self, thread_id: str, checkpoint: Checkpoint) -> None:
        # Serialize/deserialize to enforce the JSON-serializable contract and
        # to snapshot (mutating live state must not mutate the checkpoint).
        self._store[thread_id] = Checkpoint.model_validate_json(checkpoint.model_dump_json())

    async def load(self, thread_id: str) -> Optional[Checkpoint]:
        found = self._store.get(thread_id)
        return found.model_copy(deep=True) if found else None

    async def delete(self, thread_id: str) -> None:
        self._store.pop(thread_id, None)


class SqliteCheckpointer(Checkpointer):
    """Durable checkpoints in a SQLite file — zero-dependency durability.

    Suitable for single-node deployments; back the same contract with your
    own database for anything bigger.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = str(path)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shankit_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    async def save(self, thread_id: str, checkpoint: Checkpoint) -> None:
        payload = checkpoint.model_dump_json()

        def _save() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO shankit_checkpoints (thread_id, data, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        data = excluded.data, updated_at = excluded.updated_at
                    """,
                    (thread_id, payload, checkpoint.updated_at.isoformat()),
                )

        await asyncio.to_thread(_save)

    async def load(self, thread_id: str) -> Optional[Checkpoint]:
        def _load() -> Optional[str]:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT data FROM shankit_checkpoints WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
            return row[0] if row else None

        data = await asyncio.to_thread(_load)
        return Checkpoint.model_validate_json(data) if data else None

    async def delete(self, thread_id: str) -> None:
        def _delete() -> None:
            with self._connect() as conn:
                conn.execute("DELETE FROM shankit_checkpoints WHERE thread_id = ?", (thread_id,))

        await asyncio.to_thread(_delete)
