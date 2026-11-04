import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response
from mlflow.exceptions import MlflowException

from predictive_maintenance.api.services import PROJECT_ROOT, predict_pipeline, train_pipeline
from predictive_maintenance.monitoring.prediction_store import save_prediction_event
from predictive_maintenance.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _persist_prediction_event(request_id: str, event: dict[str, object]) -> None:
    """Require successful event persistence before returning a prediction."""
    database_path = Path(
        os.environ.get("PREDICTION_LOG_DB", str(PROJECT_ROOT / "artifacts" / "predictions.db"))
    )
    try:
        save_prediction_event(request_id, event, database_path)
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The prediction event could not be stored.",
            headers={"X-Request-ID": request_id},
        ) from exc


@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest, response: Response) -> PredictionResponse:
    request_id = str(uuid4())
    event: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": data.model_dump(),
        "model_uri": os.environ.get("MODEL_URI"),
        "decision_threshold": 0.5,
    }
    try:
        result = predict_pipeline(data)
    except (MlflowException, RuntimeError, OSError) as exc:
        detail = "The prediction model is unavailable."
        event.update(status="error", http_status=503, error=detail)
        _persist_prediction_event(request_id, event)
        raise HTTPException(
            status_code=503,
            detail=detail,
            headers={"X-Request-ID": request_id},
        ) from exc
    except ValueError as exc:
        detail = "Prediction failed because of invalid model data."
        event.update(status="error", http_status=500, error=detail)
        _persist_prediction_event(request_id, event)
        raise HTTPException(
            status_code=500,
            detail=detail,
            headers={"X-Request-ID": request_id},
        ) from exc

    event.update(
        status="success",
        http_status=200,
        prediction=result.prediction,
        probability=result.probability,
        model_version=result.model_version,
        mode=result.mode,
    )
    _persist_prediction_event(request_id, event)
    response.headers["X-Request-ID"] = request_id
    return result


@router.post("/train", response_model=TrainingResponse)
def train(data: TrainingRequest) -> TrainingResponse:
    try:
        result = train_pipeline(model_type=data.model_type)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The training dataset is unavailable.",
        ) from exc
    except MlflowException as exc:
        raise HTTPException(
            status_code=503,
            detail="The MLflow operation could not be completed.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Training failed because of invalid data or configuration.",
        ) from exc

    return TrainingResponse.model_validate(
        {
            "status": "completed",
            "training_run_id": result["run_id"],
            "model_type": result["model_type"],
            "model_uri": result["model_uri"],
            "metrics": result["metrics"],
            "mode": "real",
        }
    )
