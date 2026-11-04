"""Export durable prediction-event snapshots to MLflow without deleting events."""

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, cast
from uuid import uuid4

from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.exceptions import MlflowException


EXPERIMENT_NAME = "predictive-maintenance-predictions"
ARTIFACT_PATH = "predictions/predictions.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_export_time(value: str) -> datetime:
    """Parse an ISO timestamp with an explicit timezone and normalize to UTC."""
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("Export timestamps must include a timezone, for example 2026-11-01T00:00:00Z.")
    return timestamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class ExportSelection:
    """Select new events or an explicitly requested historical snapshot."""

    include_exported: bool = False
    model_uri: str | None = None
    since: datetime | None = None
    until: datetime | None = None

    def __post_init__(self) -> None:
        for timestamp in (self.since, self.until):
            if timestamp is not None and timestamp.utcoffset() is None:
                raise ValueError("Export timestamps must include a timezone.")
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("The since timestamp must be earlier than until.")

    def as_dict(self) -> dict[str, object]:
        """Return a stable, serializable selection description."""
        return {
            "include_exported": self.include_exported,
            "model_uri": self.model_uri,
            "since": self.since.astimezone(timezone.utc).isoformat() if self.since else None,
            "until": self.until.astimezone(timezone.utc).isoformat() if self.until else None,
        }

    def matches(self, event: dict[str, object]) -> bool:
        """Apply exact model matching and an inclusive/exclusive time interval."""
        if self.model_uri is not None and event.get("model_uri") != self.model_uri:
            return False
        if self.since is None and self.until is None:
            return True
        value = event.get("timestamp")
        if not isinstance(value, str):
            return False
        timestamp = parse_export_time(value)
        return (self.since is None or timestamp >= self.since) and (
            self.until is None or timestamp < self.until
        )


@dataclass(frozen=True)
class ExportBatch:
    """Represent a persisted, immutable snapshot and its remote run association."""

    batch_id: str
    payload: dict[str, object]
    run_id: str | None = None

    @property
    def events(self) -> list[dict[str, object]]:
        """Read events from the snapshot created by this module."""
        return cast(list[dict[str, object]], self.payload["events"])


@contextmanager
def _export_lock(database_path: Path) -> Iterator[None]:
    """Serialize exporters without holding a lock on the prediction database."""
    lock_path = database_path.with_name(database_path.name + ".export-lock")
    with closing(sqlite3.connect(lock_path, timeout=1)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        yield


def _initialize_exports(connection: sqlite3.Connection) -> None:
    """Add export bookkeeping without modifying existing event records."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS prediction_events (
            request_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_export_batches (
            batch_id TEXT PRIMARY KEY,
            destination TEXT NOT NULL,
            selection TEXT NOT NULL,
            mode TEXT NOT NULL,
            payload TEXT NOT NULL,
            run_id TEXT,
            status TEXT NOT NULL CHECK (status IN ('pending', 'completed'))
        );
        CREATE TABLE IF NOT EXISTS prediction_export_items (
            batch_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            PRIMARY KEY (batch_id, request_id)
        );
        """
    )


def _matching_events(
    connection: sqlite3.Connection,
    destination: str,
    selection: ExportSelection,
) -> list[dict[str, object]]:
    query = "SELECT e.payload FROM prediction_events e"
    parameters: tuple[str, ...] = ()
    if not selection.include_exported:
        query += """
            WHERE NOT EXISTS (
                SELECT 1 FROM prediction_export_items i
                JOIN prediction_export_batches b ON b.batch_id = i.batch_id
                WHERE i.request_id = e.request_id AND b.destination = ?
                  AND b.mode = 'incremental' AND b.status = 'completed'
            )
        """
        parameters = (destination,)
    rows = connection.execute(query + " ORDER BY e.rowid", parameters).fetchall()
    events = [cast(dict[str, object], json.loads(row[0])) for row in rows]
    return [event for event in events if selection.matches(event)]


def _prepare_batch(
    connection: sqlite3.Connection,
    destination: str,
    selection: ExportSelection,
) -> ExportBatch | None:
    description = json.dumps(selection.as_dict(), sort_keys=True)
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        pending = connection.execute(
            "SELECT batch_id, selection, payload, run_id FROM prediction_export_batches "
            "WHERE destination = ? AND status = 'pending' ORDER BY rowid LIMIT 1",
            (destination,),
        ).fetchone()
        if pending is not None:
            if pending[1] != description:
                raise ValueError("An unfinished export exists. Retry its original selection before changing filters.")
            return ExportBatch(pending[0], json.loads(pending[2]), pending[3])

        events = _matching_events(connection, destination, selection)
        if not events:
            return None
        batch_id = str(uuid4())
        payload: dict[str, object] = {
            "schema_version": 1,
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": selection.as_dict(),
            "event_count": len(events),
            "events": events,
        }
        mode = "historical" if selection.include_exported else "incremental"
        connection.execute(
            "INSERT INTO prediction_export_batches "
            "(batch_id, destination, selection, mode, payload, status) VALUES (?, ?, ?, ?, ?, 'pending')",
            (batch_id, destination, description, mode, json.dumps(payload, allow_nan=False)),
        )
        connection.executemany(
            "INSERT INTO prediction_export_items (batch_id, request_id) VALUES (?, ?)",
            [(batch_id, event["request_id"]) for event in events],
        )
        return ExportBatch(batch_id, payload)


def _experiment_id(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(EXPERIMENT_NAME)
    except MlflowException:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise
        return experiment.experiment_id


def _remote_run(client: MlflowClient, experiment_id: str, batch: ExportBatch) -> Run:
    if batch.run_id is not None:
        return client.get_run(batch.run_id)
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.batch_id = '{batch.batch_id}'",
        max_results=2,
    )
    if len(runs) > 1:
        raise RuntimeError("Multiple MLflow runs match this batch. Resolve the duplicate runs before retrying.")
    if runs:
        return runs[0]
    models = sorted({str(event["model_uri"]) for event in batch.events if event.get("model_uri")})
    selection = cast(dict[str, object], batch.payload["selection"])
    return client.create_run(
        experiment_id,
        run_name=f"predictions-{batch.batch_id}",
        tags={
            "operation": "predict",
            "batch_id": batch.batch_id,
            "event_count": str(len(batch.events)),
            "model_uri": models[0] if len(models) == 1 else "multiple-or-unconfigured",
            "export_mode": "historical" if selection["include_exported"] else "incremental",
        },
    )


def _complete_batch(connection: sqlite3.Connection, batch_id: str) -> None:
    with connection:
        connection.execute(
            "UPDATE prediction_export_batches SET status = 'completed' WHERE batch_id = ?",
            (batch_id,),
        )


def export_prediction_events(
    database_path: Path,
    tracking_uri: str | None = None,
    selection: ExportSelection | None = None,
) -> dict[str, object] | None:
    """Upload or resume one snapshot; leave all original events available."""
    database_path = database_path.resolve()
    if not database_path.is_file():
        return None
    destination = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    with _export_lock(database_path), closing(sqlite3.connect(database_path, timeout=10)) as connection:
        _initialize_exports(connection)
        batch = _prepare_batch(connection, destination, selection or ExportSelection())
        if batch is None:
            return None
        client = MlflowClient(tracking_uri=destination)
        run = _remote_run(client, _experiment_id(client), batch)
        run_id = run.info.run_id
        with connection:
            connection.execute(
                "UPDATE prediction_export_batches SET run_id = ? WHERE batch_id = ?",
                (run_id, batch.batch_id),
            )
        if run.info.status != "FINISHED":
            client.log_dict(run_id, batch.payload, ARTIFACT_PATH)
            client.set_terminated(run_id, status="FINISHED")
        _complete_batch(connection, batch.batch_id)
        return {
            "status": "exported",
            "batch_id": batch.batch_id,
            "run_id": run_id,
            "event_count": len(batch.events),
            "artifact_uri": f"runs:/{run_id}/{ARTIFACT_PATH}",
        }
