"""Tests for API training and prediction services."""

from pathlib import Path
from unittest.mock import MagicMock
from collections.abc import Iterator
from sklearn.pipeline import Pipeline

import pandas as pd
import pytest

from predictive_maintenance.api import services
from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS
from predictive_maintenance.data.splitting import DataSplits
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.api.schemas import PredictionRequest


@pytest.fixture
def prediction_request() -> PredictionRequest:
    """Provide a valid machine-data record."""
    return PredictionRequest(
        type="L",
        air_temperature_k=300.8,
        process_temperature_k=310.3,
        rotational_speed_rpm=1538,
        torque_nm=36.1,
        tool_wear_min=198,
    )


@pytest.fixture
def prediction_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock]:
    """Provide a controlled model without downloading artifacts."""
    model = MagicMock(spec=Pipeline)
    model.predict_proba.return_value = [[0.9, 0.1]]
    loader = MagicMock(return_value=model)

    monkeypatch.setenv("MODEL_URI", "models:/m-test-model")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.setattr(services, "load_prediction_model", loader)

    return model, loader


@pytest.fixture(autouse=True)
def clear_model_cache() -> Iterator[None]:
    """Keep cached models isolated between tests."""
    clear_cache = services.load_prediction_model.cache_clear
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def training_csv(tmp_path: Path) -> Path:
    """Create an isolated initial dataset without using repository data."""
    data = pd.DataFrame(
        {
            "UDI": range(1, 5001),
            "Product ID": [f"L{index:05d}" for index in range(5000)],
            "Type": ["L"] * 5000,
            "Air temperature [K]": [300.0] * 5000,
            "Process temperature [K]": [310.0] * 5000,
            "Rotational speed [rpm]": [1500] * 5000,
            "Torque [Nm]": [40.0] * 5000,
            "Tool wear [min]": [100] * 5000,
            "Target": [0] * 4800 + [1] * 200,
            "Failure Type": ["No Failure"] * 4800 + ["Power Failure"] * 200,
        }
    )

    path = tmp_path / "training.csv"
    data.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("model_type", ["gradient_boosting", "xgboost"])
def test_train_pipeline_prepares_data_and_calls_training(
    training_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
) -> None:
    expected_result: dict[str, object] = {
        "run_id": "service-test-run",
        "model_type": model_type,
        "model_uri": "models:/service-test-model",
        "metrics": {"f1": 0.8},
    }
    training = MagicMock(return_value=expected_result)

    monkeypatch.setattr(services, "DATASET_PATH", training_csv)
    monkeypatch.setattr(services, "run_training_experiment", training)

    result = services.train_pipeline(model_type=model_type)

    assert result == expected_result
    training.assert_called_once()
    assert training.call_args is not None
    assert training.call_args.kwargs["model_type"] == model_type

    splits: DataSplits = training.call_args.kwargs["splits"]

    assert list(splits.X_train.columns) == list(FEATURE_COLUMNS)
    assert len(splits.X_train) == 3000
    assert len(splits.X_validation) == 750
    assert len(splits.X_future) == 750
    assert len(splits.X_test) == 500

    all_indexes = set().union(
        splits.X_train.index,
        splits.X_validation.index,
        splits.X_future.index,
        splits.X_test.index,
    )
    assert len(all_indexes) == 5000

    for partition in ("train", "validation", "future", "test"):
        features = getattr(splits, f"X_{partition}")
        target = getattr(splits, f"y_{partition}")
        assert features.index.equals(target.index)


def test_train_pipeline_rejects_missing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = MagicMock()

    monkeypatch.setattr(services, "DATASET_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(services, "run_training_experiment", training)

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        services.train_pipeline()

    training.assert_not_called()


def test_load_prediction_model_reuses_cached_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Pipeline(steps=[])
    loader = MagicMock(return_value=model)
    configure_tracking = MagicMock()

    monkeypatch.setattr(services.mlflow_sklearn, "load_model", loader)
    monkeypatch.setattr(
        services.mlflow,
        "set_tracking_uri",
        configure_tracking,
    )

    first = services.load_prediction_model(
        "models:/m-test-model",
        "http://127.0.0.1:5000",
    )
    second = services.load_prediction_model(
        "models:/m-test-model",
        "http://127.0.0.1:5000",
    )

    assert first is model
    assert second is model
    loader.assert_called_once_with("models:/m-test-model")
    configure_tracking.assert_called_once_with("http://127.0.0.1:5000")


def test_load_prediction_model_rejects_non_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MagicMock(return_value={"unexpected": "artifact"})

    monkeypatch.setattr(services.mlflow_sklearn, "load_model", loader)
    monkeypatch.setattr(
        services.mlflow,
        "set_tracking_uri",
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="The loaded model must be an sklearn Pipeline.",
    ):
        services.load_prediction_model(
            "models:/m-invalid-model",
            "http://127.0.0.1:5000",
        )

    loader.assert_called_once_with("models:/m-invalid-model")


@pytest.mark.parametrize(
    ("probability", "expected_prediction"),
    [
        (0.49, 0),
        (0.50, 1),
        (0.90, 1),
    ],
)
def test_predict_pipeline_applies_threshold(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    probability: float,
    expected_prediction: int,
) -> None:
    model, loader = prediction_setup
    model.predict_proba.return_value = [[1 - probability, probability]]

    result = services.predict_pipeline(prediction_request)

    assert result.prediction == expected_prediction
    assert result.probability == pytest.approx(probability)
    assert result.model_version == "m-test-model"
    assert result.mode == "real"

    loader.assert_called_once_with(
        "models:/m-test-model",
        "http://mlflow.test:5000",
    )
    model.predict_proba.assert_called_once()
    assert model.predict_proba.call_args is not None

    features = model.predict_proba.call_args.args[0]
    assert isinstance(features, pd.DataFrame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.to_dict(orient="records") == [
        prediction_request.model_dump()
    ]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([[1.0]], "Expected probabilities"),
        ([[0.5, 0.5], [0.5, 0.5]], "Expected probabilities"),
        ([[float("nan"), 0.1]], "invalid probabilities"),
        ([[float("inf"), 0.1]], "invalid probabilities"),
        ([[-0.1, 1.1]], "invalid probabilities"),
        ([[1.1, -0.1]], "invalid probabilities"),
    ],
)
def test_predict_pipeline_rejects_invalid_probabilities(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    probabilities: list[list[float]],
    message: str,
) -> None:
    model, _ = prediction_setup
    model.predict_proba.return_value = probabilities

    with pytest.raises(ValueError, match=message):
        services.predict_pipeline(prediction_request)


def test_predict_pipeline_rejects_missing_model_uri(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loader = prediction_setup
    monkeypatch.delenv("MODEL_URI")

    with pytest.raises(RuntimeError, match="MODEL_URI is not configured"):
        services.predict_pipeline(prediction_request)

    loader.assert_not_called()


def test_predict_pipeline_uses_default_tracking_uri(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loader = prediction_setup
    monkeypatch.delenv("MLFLOW_TRACKING_URI")

    services.predict_pipeline(prediction_request)

    loader.assert_called_once_with(
        "models:/m-test-model",
        "http://127.0.0.1:5000",
    )
