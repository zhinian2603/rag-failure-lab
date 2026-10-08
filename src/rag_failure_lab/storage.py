"""SQLite run store with deterministic IDs and idempotent writes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS runs_created ON evaluation_runs(created_at DESC)")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save(self, run: dict[str, Any]) -> str:
        with self._connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO evaluation_runs
                   (id, dataset, created_at, mode, top_k, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run["id"],
                    run["dataset"],
                    run["created_at"],
                    run["config"]["mode"],
                    run["config"]["top_k"],
                    json.dumps(run, separators=(",", ":")),
                ),
            )
        return run["id"]

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT payload FROM evaluation_runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, limit: int = 25) -> list[dict[str, Any]]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, dataset, created_at, mode, top_k FROM evaluation_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
