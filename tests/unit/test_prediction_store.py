"""Tests for durable prediction-event storage."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from predictive_maintenance.monitoring.prediction_store import save_prediction_event


def read_events(path: Path) -> list[dict[str, object]]:
    """Read persisted records after the writer connection has closed."""
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT payload FROM prediction_events ORDER BY rowid").fetchall()
    return [json.loads(row[0]) for row in rows]


def test_save_prediction_event_creates_storage_and_preserves_payload(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "predictions.db"
    event: dict[str, object] = {
        "request_id": "ignored-payload-id",
        "timestamp": "2026-11-04T12:00:00+00:00",
        "inputs": {"type": "L", "torque_nm": 40.0},
        "prediction": 0,
        "probability": 0.1,
        "model_uri": "models:/m-test-model",
        "status": "success",
    }

    save_prediction_event("request-1", event, database)

    assert database.is_file()
    assert read_events(database) == [{**event, "request_id": "request-1"}]
    assert event["request_id"] == "ignored-payload-id"


def test_save_prediction_event_appends_without_overwriting(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"

    save_prediction_event("first", {"prediction": 0}, database)
    save_prediction_event("second", {"prediction": 1}, database)

    assert read_events(database) == [
        {"request_id": "first", "prediction": 0},
        {"request_id": "second", "prediction": 1},
    ]


def test_save_prediction_event_rejects_duplicate_id_without_changing_original(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"
    save_prediction_event("same-id", {"prediction": 0}, database)

    with pytest.raises(sqlite3.IntegrityError):
        save_prediction_event("same-id", {"prediction": 1}, database)

    assert read_events(database) == [{"request_id": "same-id", "prediction": 0}]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_save_prediction_event_rejects_non_finite_values(tmp_path: Path, value: float) -> None:
    database = tmp_path / "predictions.db"

    with pytest.raises(ValueError):
        save_prediction_event("request-1", {"probability": value}, database)

    assert not database.exists()


def test_save_prediction_event_rejects_unserializable_values(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"

    with pytest.raises(TypeError):
        save_prediction_event("request-1", {"invalid": object()}, database)

    assert not database.exists()
