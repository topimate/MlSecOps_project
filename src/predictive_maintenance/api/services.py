"""Connect API operations to the existing machine learning pipeline."""

import os
from typing import Literal

from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.models.mlflow_tracking import run_training_experiment
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.api.schemas import PredictionRequest, PredictionResponse
from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_5000.csv"


def train_pipeline(
    model_type: ModelType = "gradient_boosting",
) -> dict[str, object]:
    """Train and log a model using the initial dataset and fixed splits."""
    raw_data = load_raw_data(DATASET_PATH)

    processor = DataProcessor()
    normalized_data = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized_data)

    splits = create_initial_data_splits(features, target)

    return run_training_experiment(
        splits=splits,
        model_type=model_type,
    )


@lru_cache(maxsize=1)
def load_prediction_model(
    model_uri: str,
    tracking_uri: str,
) -> Pipeline:
    """Load and cache a trusted preprocessing and classifier pipeline."""
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow_sklearn.load_model(model_uri)

    if not isinstance(model, Pipeline):
        raise ValueError("The loaded model must be an sklearn Pipeline.")

    return model


def predict_pipeline(data: PredictionRequest) -> PredictionResponse:
    """Predict machine failure using the configured fitted pipeline."""
    model_uri = os.environ.get("MODEL_URI")

    if not model_uri:
        raise RuntimeError("MODEL_URI is not configured.")

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    model = load_prediction_model(model_uri, tracking_uri)

    features = pd.DataFrame(
        [data.model_dump()],
        columns=pd.Index(FEATURE_COLUMNS),
    )
    probabilities = np.asarray(model.predict_proba(features), dtype=float)

    if probabilities.shape != (1, 2):
        raise ValueError("Expected probabilities for one binary prediction.")

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("The model returned invalid probabilities.")

    probability = float(probabilities[0, 1])
    prediction: Literal[0, 1] = 1 if probability >= 0.5 else 0

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=model_uri.removeprefix("models:/"),
        mode="real",
    )
