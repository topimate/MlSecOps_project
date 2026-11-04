
"""Unit tests for feature selection and preprocessing."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.predictive_maintenance.data.preprocessing import (
    DataProcessor,
    engineer_temperature_difference,
)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create valid model features for preprocessing tests."""
    return pd.DataFrame(
        {
            "type": ["L", "M", "H", "L", "M", "H"],
            "air_temperature_k": [298.0, 299.0, 300.0, 301.0, 302.0, 303.0],
            "process_temperature_k": [308.0, 310.0, 311.0, 312.0, 313.0, 314.0],
            "rotational_speed_rpm": [1400, 1500, 1600, 1700, 1800, 1900],
            "torque_nm": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            "tool_wear_min": [10, 20, 30, 40, 50, 60],
        }
    )


def test_rename_columns() -> None:
    """Original column names should become snake_case."""
    processor = DataProcessor()

    raw = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature [K]": [300.0],
            "Target": [0],
        }
    )

    result = processor.rename_columns(raw)

    assert "type" in result.columns
    assert "air_temperature_k" in result.columns
    assert "target" in result.columns
    assert "Air temperature [K]" not in result.columns


def test_rename_empty_dataframe() -> None:
    """Empty input should be rejected."""
    processor = DataProcessor()

    with pytest.raises(ValueError, match="empty"):
        processor.rename_columns(pd.DataFrame())


def test_select_features_excludes_leakage_columns(
    sample_features: pd.DataFrame,
) -> None:
    """Identifiers, target and failure type must be excluded."""
    processor = DataProcessor()
    df = sample_features.copy()

    df["udi"] = range(len(df))
    df["product_id"] = ["X"] * len(df)
    df["failure_type"] = ["No Failure"] * len(df)
    df["target"] = [0, 0, 1, 0, 0, 1]

    result = processor.select_features(df)

    assert result.shape == (6, 6)
    assert "udi" not in result.columns
    assert "product_id" not in result.columns
    assert "failure_type" not in result.columns
    assert "target" not in result.columns


def test_missing_feature(sample_features: pd.DataFrame) -> None:
    """Missing model features should raise an error."""
    processor = DataProcessor()
    invalid = sample_features.drop(columns=["torque_nm"])

    with pytest.raises(ValueError, match="torque_nm"):
        processor.select_features(invalid)


def test_split_features_target(sample_features: pd.DataFrame) -> None:
    """Features and binary targets should be separated."""
    processor = DataProcessor()
    df = sample_features.copy()
    df["target"] = [0, 1, 0, 0, 1, 0]

    X, y = processor.split_features_target(df)

    assert X.shape == (6, 6)
    assert y.shape == (6,)
    assert "target" not in X.columns
    assert y.tolist() == [0, 1, 0, 0, 1, 0]


@pytest.mark.parametrize("invalid_target", [2, -1, None])
def test_invalid_target(
    sample_features: pd.DataFrame,
    invalid_target: int | None,
) -> None:
    """Only 0 and 1 are accepted as target values."""
    processor = DataProcessor()
    df = sample_features.copy()
    df["target"] = [0, 1, 0, 0, 1, 0]
    df.loc[0, "target"] = invalid_target

    with pytest.raises(ValueError, match="Target"):
        processor.split_features_target(df)


def test_temperature_difference(sample_features: pd.DataFrame) -> None:
    """Temperature difference should equal process minus air temperature."""
    processor = DataProcessor()

    result = processor.add_engineered_features(sample_features)

    expected = cast(
        pd.Series,
        sample_features["process_temperature_k"]
        - sample_features["air_temperature_k"],
    )

    pd.testing.assert_series_equal(
        cast(pd.Series, result["temperature_difference_k"]),
        expected.rename("temperature_difference_k"),
    )


def test_feature_engineering_preserves_input(
    sample_features: pd.DataFrame,
) -> None:
    """Feature engineering must not modify the original DataFrame."""
    original = sample_features.copy(deep=True)

    engineer_temperature_difference(sample_features)

    pd.testing.assert_frame_equal(sample_features, original)


def test_feature_engineering_missing_column(
    sample_features: pd.DataFrame,
) -> None:
    """Missing temperature inputs should be rejected."""
    invalid = sample_features.drop(columns=["process_temperature_k"])

    with pytest.raises(ValueError, match="process_temperature_k"):
        engineer_temperature_difference(invalid)


def test_preprocessing_output_shape(
    sample_features: pd.DataFrame,
) -> None:
    """The pipeline should produce six numerical and three encoded features."""
    processor = DataProcessor()
    pipeline = processor.create_preprocessing_pipeline()

    result = pipeline.fit_transform(sample_features)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (6, 9)
    assert result.columns.is_unique
    assert result.isna().sum().sum() == 0


def test_missing_value_imputation(
    sample_features: pd.DataFrame,
) -> None:
    """Numeric and categorical missing values should be imputed."""
    processor = DataProcessor()
    invalid = sample_features.copy()

    invalid.loc[0, "torque_nm"] = np.nan
    invalid.loc[1, "type"] = np.nan

    pipeline = processor.create_preprocessing_pipeline()
    result = pipeline.fit_transform(invalid)

    assert result.isna().sum().sum() == 0
    assert result.shape == (6, 9)


def test_unknown_category(
    sample_features: pd.DataFrame,
) -> None:
    """Unknown categories should not change the output feature schema."""
    processor = DataProcessor()
    pipeline = processor.create_preprocessing_pipeline()

    pipeline.fit(sample_features)

    new_data = sample_features.iloc[[0]].copy()
    new_data["type"] = "X"

    result = pipeline.transform(new_data)

    assert result.shape == (1, 9)

    type_columns = [
        column for column in result.columns
        if column.startswith("type_")
    ]

    assert len(type_columns) == 3
    assert (result[type_columns].iloc[0] == 0).all()


def test_numeric_scaling(sample_features: pd.DataFrame) -> None:
    """Scaled numeric columns should have mean approximately zero."""
    processor = DataProcessor()
    pipeline = processor.create_preprocessing_pipeline()

    result = pipeline.fit_transform(sample_features)

    numeric_columns = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temperature_difference_k",
    ]

    for column in numeric_columns:
        assert np.isclose(result[column].mean(), 0.0, atol=1e-10)


def test_train_only_statistics(
    sample_features: pd.DataFrame,
) -> None:
    """Validation data must not change learned scaling parameters."""
    processor = DataProcessor()
    pipeline = processor.create_preprocessing_pipeline()

    pipeline.fit(sample_features)

    scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )

    mean_before = scaler.mean_.copy()

    validation = sample_features.iloc[[0]].copy()
    validation["torque_nm"] = 999999.0

    pipeline.transform(validation)

    np.testing.assert_array_equal(mean_before, scaler.mean_)


# Additional coverage for less common validation and imputation paths.

def test_feature_engineering_rejects_empty_dataframe(
    sample_features: pd.DataFrame,
) -> None:
    """An empty frame with required columns must be rejected."""
    with pytest.raises(ValueError, match="empty"):
        engineer_temperature_difference(sample_features.iloc[:0])


def test_renaming_rejects_duplicate_output_names(
    sample_features: pd.DataFrame,
) -> None:
    """A raw and already-renamed field cannot share a final name."""
    df = sample_features.assign(Type="L")
    with pytest.raises(ValueError, match="Duplicate column names"):
        DataProcessor().rename_columns(df)


def test_select_features_rejects_empty_dataframe(
    sample_features: pd.DataFrame,
) -> None:
    """Selecting from empty input should not silently succeed."""
    with pytest.raises(ValueError, match="empty"):
        DataProcessor().select_features(sample_features.iloc[:0])


def test_split_rejects_missing_target(sample_features: pd.DataFrame) -> None:
    """Model training requires a target column."""
    with pytest.raises(ValueError, match="Missing target column"):
        DataProcessor().split_features_target(sample_features)


def test_imputer_handles_missing_values(sample_features: pd.DataFrame) -> None:
    """The standalone imputer learns numeric medians and categorical modes."""
    processor = DataProcessor()
    engineered = processor.add_engineered_features(sample_features)
    engineered.loc[0, "torque_nm"] = float("nan")
    engineered.loc[1, "type"] = np.nan
    imputer = processor.create_imputer()
    transformed = imputer.fit_transform(engineered)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == (6, 7)
    assert not transformed.isna().to_numpy().any()
