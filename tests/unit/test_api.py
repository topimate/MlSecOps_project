import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from predictive_maintenance.api import routes
from predictive_maintenance.api.main import app
from predictive_maintenance.api.schemas import PredictionResponse

client = TestClient(app)


@pytest.fixture(autouse=True)
def prediction_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use isolated storage for every API test."""
    path = tmp_path / "predictions.db"
    monkeypatch.setenv("PREDICTION_LOG_DB", str(path))
    monkeypatch.setenv("MODEL_URI", "models:/m-test-model")
    return path


def read_prediction_events(path: Path) -> list[dict[str, object]]:
    """Read committed events through a new connection."""
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT payload FROM prediction_events ORDER BY rowid").fetchall()
    return [json.loads(row[0]) for row in rows]


@pytest.fixture
def prediction_data() -> dict[str, str | float]:
    return {
        "type": "L",
        "air_temperature_k": 300.8,
        "process_temperature_k": 310.3,
        "rotational_speed_rpm": 1538.0,
        "torque_nm": 36.1,
        "tool_wear_min": 198.0,
    }


@pytest.fixture(autouse=True)
def mock_training(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace training during API tests to avoid external side effects."""
    training = MagicMock(
        return_value={
            "run_id": "test-run-123",
            "model_type": "gradient_boosting",
            "model_uri": "models:/test-model",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.80,
                "recall": 0.75,
                "f1": 0.77,
                "roc_auc": 0.90,
                "average_precision": 0.85,
            },
        }
    )
    monkeypatch.setattr(routes, "train_pipeline", training)
    return training


@pytest.fixture(autouse=True)
def mock_prediction(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace prediction during API tests without loading a real model."""
    prediction = MagicMock(
        return_value=PredictionResponse(
            prediction=0,
            probability=0.1,
            model_version="m-test-model",
            mode="real",
        )
    )
    monkeypatch.setattr(routes, "predict_pipeline", prediction)
    return prediction


# TEST HEALTH ENDPOINT


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# TEST PREDICT ENDPOINT


def test_predict_returns_prediction_result(
    prediction_data: dict[str, str | float],
    mock_prediction: MagicMock,
    prediction_db: Path,
) -> None:
    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "probability": 0.1,
        "model_version": "m-test-model",
        "mode": "real",
    }

    mock_prediction.assert_called_once()
    assert mock_prediction.call_args is not None
    request = mock_prediction.call_args.args[0]
    assert request.model_dump() == prediction_data

    request_id = response.headers["X-Request-ID"]
    assert str(UUID(request_id)) == request_id
    events = read_prediction_events(prediction_db)
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == request_id
    assert event["inputs"] == prediction_data
    assert event["model_uri"] == "models:/m-test-model"
    assert event["model_version"] == response.json()["model_version"]
    assert event["prediction"] == response.json()["prediction"]
    assert event["probability"] == response.json()["probability"]
    assert event["decision_threshold"] == 0.5
    assert event["status"] == "success"
    assert event["http_status"] == 200
    assert event["mode"] == "real"
    timestamp = event["timestamp"]
    assert isinstance(timestamp, str)
    assert datetime.fromisoformat(timestamp).utcoffset() is not None


def test_predict_rejects_missing_field(
    prediction_data: dict[str, str | float],
) -> None:
    del prediction_data["tool_wear_min"]

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "tool_wear_min"]
        and error["type"] == "missing"
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("type", "X"),
        ("air_temperature_k", 0),
        ("process_temperature_k", -1),
        ("rotational_speed_rpm", -1),
        ("torque_nm", -5),
        ("tool_wear_min", -1),
        ("torque_nm", "abc"),
    ],
)
def test_predict_rejects_invalid_value(
    prediction_data: dict[str, str | float],
    field: str,
    invalid_value: str | float,
    mock_prediction: MagicMock,
    prediction_db: Path,
) -> None:
    prediction_data[field] = invalid_value

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", field]
        for error in response.json()["detail"]
    )
    mock_prediction.assert_not_called()
    assert not prediction_db.exists()


def test_predict_rejects_extra_field(
    prediction_data: dict[str, str | float],
) -> None:
    prediction_data["unexpected"] = "value"

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "unexpected"]
        and error["type"] == "extra_forbidden"
        for error in response.json()["detail"]
    )


# TEST TRAIN ENDPOINT
@pytest.mark.parametrize("model_type", ["gradient_boosting", "xgboost"])
def test_train_returns_training_result(
    model_type: str,
    mock_training: MagicMock,
) -> None:
    mock_training.return_value["model_type"] = model_type

    response = client.post(
        "/train",
        json={
            "dataset_id": "default",
            "model_type": model_type,
            "random_state": 42,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "training_run_id": "test-run-123",
        "model_type": model_type,
        "model_uri": "models:/test-model",
        "metrics": mock_training.return_value["metrics"],
        "mode": "real",
    }
    mock_training.assert_called_once_with(model_type=model_type)


def test_train_accepts_empty_object(mock_training: MagicMock) -> None:
    response = client.post("/train", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["model_type"] == "gradient_boosting"
    mock_training.assert_called_once_with(model_type="gradient_boosting")


def test_train_rejects_missing_body() -> None:
    response = client.post("/train")

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body"]
        and error["type"] == "missing"
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("field", "invalid_value", "error_type"),
    [
        ("dataset_id", "unknown", "literal_error"),
        ("random_state", -1, "literal_error"),
        ("random_state", 7, "literal_error"),
        ("model_type", "unknown", "literal_error"),
        ("random_state", "abc", "literal_error"),
    ],
)
def test_train_rejects_invalid_value(
    field: str,
    invalid_value: str | int,
    error_type: str,
) -> None:
    data: dict[str, str | int] = {
        "dataset_id": "default",
        "random_state": 42,
    }
    data[field] = invalid_value

    response = client.post("/train", json=data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", field]
        and error["type"] == error_type
        for error in response.json()["detail"]
    )


def test_train_rejects_extra_field() -> None:
    response = client.post(
        "/train",
        json={
            "dataset_id": "default",
            "random_state": 42,
            "unexpected": True,
        },
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "unexpected"]
        and error["type"] == "extra_forbidden"
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_detail"),
    [
        (
            FileNotFoundError,
            503,
            "The training dataset is unavailable.",
        ),
        (
            MlflowException,
            503,
            "The MLflow operation could not be completed.",
        ),
        (
            ValueError,
            500,
            "Training failed because of invalid data or configuration.",
        ),
    ],
)
def test_train_handles_service_errors(
    mock_training: MagicMock,
    error_type: type[Exception],
    expected_status: int,
    expected_detail: str,
) -> None:
    mock_training.side_effect = error_type("Internal diagnostic details.")

    response = client.post("/train", json={})

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    mock_training.assert_called_once_with(model_type="gradient_boosting")


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_detail"),
    [
        (
            RuntimeError,
            503,
            "The prediction model is unavailable.",
        ),
        (
            OSError,
            503,
            "The prediction model is unavailable.",
        ),
        (
            MlflowException,
            503,
            "The prediction model is unavailable.",
        ),
        (
            ValueError,
            500,
            "Prediction failed because of invalid model data.",
        ),
    ],
)
def test_predict_handles_service_errors(
    prediction_data: dict[str, str | float],
    mock_prediction: MagicMock,
    prediction_db: Path,
    error_type: type[Exception],
    expected_status: int,
    expected_detail: str,
) -> None:
    mock_prediction.side_effect = error_type("Internal diagnostic details.")

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    mock_prediction.assert_called_once()

    events = read_prediction_events(prediction_db)
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["inputs"] == prediction_data
    assert event["status"] == "error"
    assert event["http_status"] == expected_status
    assert event["error"] == expected_detail
    assert "Internal diagnostic details" not in json.dumps(event)


@pytest.mark.parametrize("error_type", [OSError, sqlite3.OperationalError, ValueError])
def test_predict_reports_event_storage_failure(
    prediction_data: dict[str, str | float],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    storage = MagicMock(side_effect=error_type("Internal storage details."))
    monkeypatch.setattr(routes, "save_prediction_event", storage)

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 503
    assert response.json() == {"detail": "The prediction event could not be stored."}
    assert response.headers["X-Request-ID"]
    storage.assert_called_once()


def test_predict_persists_multiple_requests(
    prediction_data: dict[str, str | float],
    prediction_db: Path,
) -> None:
    first = client.post("/predict", json=prediction_data)
    second = client.post("/predict", json=prediction_data)

    assert first.status_code == second.status_code == 200
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    events = read_prediction_events(prediction_db)
    assert {event["request_id"] for event in events} == {
        first.headers["X-Request-ID"],
        second.headers["X-Request-ID"],
    }


def test_predict_uses_default_event_storage(
    prediction_data: dict[str, str | float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREDICTION_LOG_DB")
    monkeypatch.setattr(routes, "PROJECT_ROOT", tmp_path)

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 200
    events = read_prediction_events(tmp_path / "artifacts" / "predictions.db")
    assert events[0]["request_id"] == response.headers["X-Request-ID"]
