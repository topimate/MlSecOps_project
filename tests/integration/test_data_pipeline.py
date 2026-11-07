
"""Integration tests for the predictive maintenance data pipeline."""

from typing import cast

from pathlib import Path

import pandas as pd
import pytest

from src.predictive_maintenance.data.imbalance import calculate_sample_weights
from src.predictive_maintenance.data.loader import load_raw_data
from src.predictive_maintenance.data.preprocessing import DataProcessor
from src.predictive_maintenance.data.splitting import create_initial_data_splits

from src.predictive_maintenance.data.persistence import (
    load_preprocessing_pipeline,
    save_preprocessing_pipeline,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "predictive_maintenance_5000.csv"
)


@pytest.fixture(scope="module")
def prepared_data():
    """Load and split the real dataset for integration testing."""
    df = load_raw_data(DATASET_PATH)

    processor = DataProcessor()
    renamed = processor.rename_columns(df)
    X, y = processor.split_features_target(renamed)

    splits = create_initial_data_splits(X, y)

    return processor, splits


def test_end_to_end_preprocessing(prepared_data) -> None:
    """Verify the complete data preparation and transformation flow."""
    processor, splits = prepared_data

    pipeline = processor.create_preprocessing_pipeline()

    X_train = pipeline.fit_transform(splits.X_train)
    X_validation = pipeline.transform(splits.X_validation)
    X_future = pipeline.transform(splits.X_future)
    X_test = pipeline.transform(splits.X_test)

    expected_sizes = {
        "train": (X_train, 3000),
        "validation": (X_validation, 750),
        "future": (X_future, 750),
        "test": (X_test, 500),
    }

    for name, (features, expected_rows) in expected_sizes.items():
        assert isinstance(features, pd.DataFrame)
        assert features.shape == (expected_rows, 9), name
        assert not features.isna().to_numpy().any(), name
        assert features.columns.equals(X_train.columns), name

    assert "temperature_difference_k" in X_train.columns
    assert "target" not in X_train.columns
    assert "failure_type" not in X_train.columns

    original_indexes = set().union(
        splits.X_train.index,
        splits.X_validation.index,
        splits.X_future.index,
        splits.X_test.index,
    )

    assert len(original_indexes) == 5000

    for name in ("train", "validation", "future", "test"):
        features = getattr(splits, f"X_{name}")
        target = getattr(splits, f"y_{name}")
        assert features.index.equals(target.index)


def test_training_sample_weights(prepared_data) -> None:
    """Verify training weights align with the training partition."""
    _, splits = prepared_data

    weights = calculate_sample_weights(splits.y_train)

    assert len(weights) == 3000
    assert weights.index.equals(splits.X_train.index)
    assert (weights > 0).all()

    normal_weight = cast(pd.Series, weights.loc[splits.y_train == 0]).iloc[0]
    failure_weight = cast(pd.Series, weights.loc[splits.y_train == 1]).iloc[0]

    assert failure_weight > normal_weight

    other_indexes = set().union(
        splits.X_validation.index,
        splits.X_future.index,
        splits.X_test.index,
    )

    assert set(weights.index).isdisjoint(other_indexes)


def test_pipeline_persistence_and_inference(
    prepared_data,
    tmp_path: Path,
) -> None:
    """Verify fitted pipeline persistence and inference consistency."""
    processor, splits = prepared_data

    pipeline = processor.create_preprocessing_pipeline()
    pipeline.fit(splits.X_train)

    # Simulate new observations without known target labels.
    new_observations = splits.X_future.iloc[:10].copy()
    inference_features = processor.select_features(new_observations)

    original_result = pipeline.transform(inference_features)

    artifact_path = tmp_path / "preprocessing.joblib"

    save_preprocessing_pipeline(pipeline, artifact_path)
    assert artifact_path.is_file()

    restored_pipeline = load_preprocessing_pipeline(artifact_path)
    restored_result = restored_pipeline.transform(inference_features)

    pd.testing.assert_frame_equal(original_result, restored_result)

    assert restored_result.shape == (10, 9)
    assert not restored_result.isna().any().any()
    assert "target" not in restored_result.columns
