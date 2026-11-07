"""Verify real Evidently reports using reproducible synthetic data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.monitoring.drift_report import DriftReporter
from predictive_maintenance.monitoring.simulation import simulate_feature_drift


@pytest.fixture
def reference_features() -> pd.DataFrame:
    """Create varied model inputs with deterministic distributions."""
    rng = np.random.default_rng(42)
    size = 120

    return pd.DataFrame({
        "type": np.resize(["L", "M", "H"], size),
        "air_temperature_k": rng.normal(300, 2, size),
        "process_temperature_k": rng.normal(310, 2, size),
        "rotational_speed_rpm": rng.normal(1500, 100, size),
        "torque_nm": rng.normal(40, 5, size),
        "tool_wear_min": rng.uniform(0, 250, size),
    })


@pytest.mark.parametrize(
    "with_drift, expected_count",
    [(False, 0), (True, 4)],
)
def test_real_monitoring_report(
    tmp_path: Path,
    reference_features: pd.DataFrame,
    with_drift: bool,
    expected_count: int,
) -> None:
    """Detect controlled shifts and export consistent report files."""
    original = reference_features.copy(deep=True)

    current = (
        simulate_feature_drift(reference_features)
        if with_drift
        else reference_features.copy(deep=True)
    )
    current_before_report = current.copy(deep=True)

    html_path, json_path = DriftReporter().save(
        reference=reference_features,
        current=current,
        output_dir=tmp_path / "monitoring",
    )

    assert html_path.is_file()
    assert html_path.stat().st_size > 0

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["metrics"]

    summary_path = json_path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["column_count"] == 6
    assert summary["drifted_column_count"] == expected_count
    assert summary["drifted_column_share"] == pytest.approx(
        expected_count / 6
    )
    assert summary["threshold_reached"] is with_drift

    pd.testing.assert_frame_equal(reference_features, original)
    pd.testing.assert_frame_equal(current, current_before_report)

    for column in ("type", "tool_wear_min"):
        pd.testing.assert_series_equal(current[column], original[column])
