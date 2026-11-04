"""Tests for incremental, historical, and recoverable prediction exports."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from predictive_maintenance.monitoring import prediction_export as exports
from predictive_maintenance.monitoring.prediction_store import save_prediction_event


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.db"
    for number, model in ((1, "a"), (2, "b"), (3, "a")):
        save_prediction_event(
            f"request-{number}",
            {
                "timestamp": f"2026-11-0{number}T00:00:00+00:00",
                "model_uri": f"models:/m-{model}",
                "status": "success" if number != 2 else "error",
            },
            path,
        )
    return path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(spec=MlflowClient)
    mock.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="123")
    mock.search_runs.return_value = []
    mock.create_run.side_effect = [
        SimpleNamespace(info=SimpleNamespace(run_id=f"run-{number}", status="RUNNING"))
        for number in range(1, 10)
    ]
    mock.get_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-1", status="RUNNING"))
    monkeypatch.setattr(exports, "MlflowClient", MagicMock(return_value=mock))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    return mock


def stored_batches(database: Path) -> list[tuple[str, str, str | None]]:
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(
            "SELECT batch_id, status, run_id FROM prediction_export_batches ORDER BY rowid"
        ).fetchall()


def test_incremental_export_skips_completed_events_and_preserves_originals(
    database: Path, client: MagicMock,
) -> None:
    first = exports.export_prediction_events(database)
    assert first is not None
    assert first["event_count"] == 3
    assert first["artifact_uri"] == "runs:/run-1/predictions/predictions.json"
    assert stored_batches(database) == [(first["batch_id"], "completed", "run-1")]
    assert client.log_dict.call_args is not None
    payload = client.log_dict.call_args.args[1]
    assert [event["request_id"] for event in payload["events"]] == ["request-1", "request-2", "request-3"]
    assert exports.export_prediction_events(database) is None
    assert client.create_run.call_count == 1

    save_prediction_event("request-4", {"status": "success"}, database)
    second = exports.export_prediction_events(database)
    assert second is not None
    assert second["event_count"] == 1
    assert client.log_dict.call_args.args[1]["events"] == [{"request_id": "request-4", "status": "success"}]
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0] == 4


def test_historical_export_can_repeat_old_data_and_does_not_consume_incremental_queue(
    database: Path, client: MagicMock,
) -> None:
    selection = exports.ExportSelection(
        include_exported=True,
        model_uri="models:/m-a",
        since=exports.parse_export_time("2026-11-02T00:00:00Z"),
        until=exports.parse_export_time("2026-11-04T00:00:00Z"),
    )
    first = exports.export_prediction_events(database, selection=selection)
    second = exports.export_prediction_events(database, selection=selection)
    assert first is not None and second is not None
    assert first["batch_id"] != second["batch_id"]
    assert first["event_count"] == second["event_count"] == 1
    assert client.log_dict.call_args is not None
    assert client.log_dict.call_args.args[1]["events"][0]["request_id"] == "request-3"
    incremental = exports.export_prediction_events(database)
    assert incremental is not None
    assert incremental["event_count"] == 3


def test_failed_upload_resumes_same_run_and_snapshot(database: Path, client: MagicMock) -> None:
    client.log_dict.side_effect = [RuntimeError("Upload interrupted"), None, None]
    with pytest.raises(RuntimeError, match="Upload interrupted"):
        exports.export_prediction_events(database)
    original = stored_batches(database)[0]
    assert original[1:] == ("pending", "run-1")
    save_prediction_event("later", {"status": "success"}, database)

    retried = exports.export_prediction_events(database)
    assert retried is not None
    assert retried["batch_id"] == original[0]
    assert retried["run_id"] == "run-1"
    assert retried["event_count"] == 3
    assert client.create_run.call_count == 1
    assert client.log_dict.call_args_list[0].args == client.log_dict.call_args_list[1].args
    next_batch = exports.export_prediction_events(database)
    assert next_batch is not None
    assert next_batch["event_count"] == 1


def test_lost_create_response_is_recovered_by_batch_tag(database: Path, client: MagicMock) -> None:
    client.create_run.side_effect = RuntimeError("Create response lost")
    with pytest.raises(RuntimeError, match="Create response lost"):
        exports.export_prediction_events(database)
    batch_id = stored_batches(database)[0][0]
    client.search_runs.return_value = [
        SimpleNamespace(info=SimpleNamespace(run_id="remote-created-run", status="RUNNING"))
    ]

    result = exports.export_prediction_events(database)

    assert result is not None
    assert result["run_id"] == "remote-created-run"
    assert result["batch_id"] == batch_id
    assert client.create_run.call_count == 1


def test_finished_remote_run_is_not_uploaded_again_after_local_commit_failure(
    database: Path, client: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(exports, "_complete_batch", MagicMock(side_effect=sqlite3.OperationalError("Disk error")))
        with pytest.raises(sqlite3.OperationalError, match="Disk error"):
            exports.export_prediction_events(database)
    assert stored_batches(database)[0][1] == "pending"
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-1", status="FINISHED"))

    result = exports.export_prediction_events(database)

    assert result is not None
    assert stored_batches(database)[0][1] == "completed"
    client.log_dict.assert_called_once()


def test_pending_selection_cannot_be_silently_changed(database: Path, client: MagicMock) -> None:
    client.log_dict.side_effect = RuntimeError("Upload interrupted")
    with pytest.raises(RuntimeError):
        exports.export_prediction_events(database)
    with pytest.raises(ValueError, match="Retry its original selection"):
        exports.export_prediction_events(database, selection=exports.ExportSelection(include_exported=True))
    assert len(stored_batches(database)) == 1


def test_destinations_have_independent_incremental_bookkeeping(database: Path, client: MagicMock) -> None:
    first = exports.export_prediction_events(database, tracking_uri="http://first.test")
    second = exports.export_prediction_events(database, tracking_uri="http://second.test")
    assert first is not None and second is not None
    assert first["event_count"] == second["event_count"] == 3
    assert client.create_run.call_count == 2


def test_missing_or_empty_database_does_not_contact_mlflow(tmp_path: Path, client: MagicMock) -> None:
    path = tmp_path / "empty.db"
    assert exports.export_prediction_events(path) is None
    assert not path.exists()
    with closing(sqlite3.connect(path)):
        pass
    assert exports.export_prediction_events(path) is None
    client.create_run.assert_not_called()


def test_experiment_creation_and_concurrent_creation_recovery(database: Path, client: MagicMock) -> None:
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "new-experiment"
    assert exports.export_prediction_events(database) is not None
    client.create_experiment.assert_called_once_with(exports.EXPERIMENT_NAME)

    client.get_experiment_by_name.side_effect = [None, SimpleNamespace(experiment_id="existing-experiment")]
    client.create_experiment.side_effect = MlflowException("Experiment already exists")
    assert exports.export_prediction_events(database, tracking_uri="http://other.test") is not None


def test_experiment_creation_failure_remains_retryable(database: Path, client: MagicMock) -> None:
    client.get_experiment_by_name.return_value = None
    client.create_experiment.side_effect = MlflowException("Unavailable")
    with pytest.raises(MlflowException, match="Unavailable"):
        exports.export_prediction_events(database)
    assert stored_batches(database)[0][1] == "pending"


def test_duplicate_remote_batch_matches_are_rejected(database: Path, client: MagicMock) -> None:
    client.search_runs.return_value = [object(), object()]
    with pytest.raises(RuntimeError, match="Multiple MLflow runs"):
        exports.export_prediction_events(database)
    client.create_run.assert_not_called()


def test_export_lock_does_not_block_new_prediction_events(database: Path, client: MagicMock) -> None:
    with exports._export_lock(database):
        save_prediction_event("during-export", {"status": "success"}, database)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            exports.export_prediction_events(database)
    assert exports.export_prediction_events(database) is not None


def test_date_filter_boundaries_and_missing_timestamps() -> None:
    selection = exports.ExportSelection(
        since=exports.parse_export_time("2026-11-02T00:00:00Z"),
        until=exports.parse_export_time("2026-11-03T00:00:00Z"),
    )
    assert selection.matches({"timestamp": "2026-11-02T00:00:00+00:00"})
    assert not selection.matches({"timestamp": "2026-11-03T00:00:00+00:00"})
    assert not selection.matches({"timestamp": "2026-11-01T00:00:00+00:00"})
    assert not selection.matches({})
    assert exports.ExportSelection(until=selection.until).matches({"timestamp": "2026-11-01T00:00:00Z"})
    assert exports.ExportSelection(since=selection.since).matches({"timestamp": "2026-11-04T00:00:00Z"})


def test_selection_rejects_naive_or_reversed_timestamps() -> None:
    naive = datetime(2026, 11, 1)
    aware = naive.replace(tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="timezone"):
        exports.ExportSelection(since=naive)
    with pytest.raises(ValueError, match="earlier"):
        exports.ExportSelection(since=aware, until=aware)
    with pytest.raises(ValueError, match="timezone"):
        exports.parse_export_time("2026-11-01T00:00:00")
    with pytest.raises(ValueError):
        exports.parse_export_time("not-a-date")
    assert exports.parse_export_time("2026-11-01T02:00:00+02:00") == aware


def test_invalid_saved_timestamp_fails_without_consuming_events(database: Path, client: MagicMock) -> None:
    save_prediction_event("invalid-date", {"timestamp": "not-a-date"}, database)
    with pytest.raises(ValueError):
        exports.export_prediction_events(
            database,
            selection=exports.ExportSelection(since=exports.parse_export_time("2026-11-01T00:00:00Z")),
        )
    client.log_dict.assert_not_called()
    assert stored_batches(database) == []
