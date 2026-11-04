from predictive_maintenance.api.schemas import (
    MockTrainingResponse,
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
)


def mock_predict(data: PredictionRequest) -> PredictionResponse:
    """return a fixed prediction and probability for testing purposes."""
    return PredictionResponse(
        prediction=0,
        probability=0.1,
        model_version="mock-v1",
        mode="mock",
    )


def mock_train(data: TrainingRequest) -> MockTrainingResponse:
    """Return a simulated training result without creating a model."""
    return MockTrainingResponse(
        status="mock_completed",
    )
