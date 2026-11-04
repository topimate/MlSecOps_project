"""Tests for the prediction-export command-line interface."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import export_predictions as command


def test_command_passes_historical_filters_and_prints_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    exporter = MagicMock(return_value={"status": "exported", "artifact_uri": "runs:/run-1/predictions.json"})
    monkeypatch.setattr(command, "export_prediction_events", exporter)
    database = tmp_path / "events.db"

    status = command.main([
        "--database", str(database), "--tracking-uri", "http://mlflow.test", "--all",
        "--model-uri", "models:/m-a", "--since", "2026-11-01T00:00:00Z", "--until", "2026-11-03T00:00:00Z",
    ])

    assert status == 0
    assert json.loads(capsys.readouterr().out)["status"] == "exported"
    exporter.assert_called_once()
    assert exporter.call_args is not None
    assert exporter.call_args.args[0] == database
    selection = exporter.call_args.args[2]
    assert selection.include_exported
    assert selection.model_uri == "models:/m-a"


def test_command_handles_empty_selection_and_environment_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    exporter = MagicMock(return_value=None)
    monkeypatch.setattr(command, "export_prediction_events", exporter)
    database = tmp_path / "from-env.db"
    monkeypatch.setenv("PREDICTION_LOG_DB", str(database))

    assert command.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "no_matching_events"}
    assert exporter.call_args is not None
    assert exporter.call_args.args[0] == database


def test_command_reports_storage_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(command, "export_prediction_events", MagicMock(side_effect=sqlite3.OperationalError("Locked")))
    assert command.main([]) == 1
    assert "Export failed: Locked" in capsys.readouterr().err


def test_command_rejects_invalid_date_argument() -> None:
    with pytest.raises(SystemExit) as error:
        command.main(["--since", "not-a-date"])
    assert error.value.code == 2


def test_command_reports_reversed_time_interval(capsys: pytest.CaptureFixture[str]) -> None:
    assert command.main(["--since", "2026-11-03T00:00:00Z", "--until", "2026-11-01T00:00:00Z"]) == 1
    assert "earlier than until" in capsys.readouterr().err
