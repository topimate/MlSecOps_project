"""Persist prediction events in a local SQLite database."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def save_prediction_event(
    request_id: str,
    event: dict[str, object],
    database_path: Path,
) -> None:
    """Store one prediction event in a committed transaction."""
    payload = json.dumps(
        {**event, "request_id": request_id},
        ensure_ascii=False,
        allow_nan=False,
    )

    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path, timeout=10)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_events (
                    request_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO prediction_events (request_id, payload)
                VALUES (?, ?)
                """,
                (request_id, payload),
            )
