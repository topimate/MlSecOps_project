"""Test drift decisions and invalid report handling."""

import json
from pathlib import Path
from typing import Any

import pytest

from predictive_maintenance.monitoring.summary import (
    save_summary,
    summarize_report,
)


def make_payload(count: int = 4) -> dict[str, Any]:
    """Build a minimal report using the observed Evidently format."""
    return {
        "metrics": [{
            "config": {
                "type": "evidently:metric_v2:DriftedColumnsCount",
                "columns": ["a", "b", "c", "d", "e", "f"],
                "drift_share": 0.5,
            },
            "value": {
                "count": count,
                "share": count / 6,
            },
        }],
        "tests": [],
    }


def write_report(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Write an isolated test report."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "count, expected",
    [(0, False), (2, False), (3, True), (4, True), (6, True)],
)
def test_drift_decision(
    tmp_path: Path,
    count: int,
    expected: bool,
) -> None:
    """Apply the decision threshold, including exactly 50 percent."""
    report_path = write_report(tmp_path, make_payload(count))

    summary = summarize_report(report_path)

    assert summary["column_count"] == 6
    assert summary["drifted_column_count"] == count
    assert summary["drifted_column_share"] == pytest.approx(count / 6)
    assert summary["drift_share_threshold"] == 0.5
    assert summary["threshold_reached"] is expected


@pytest.mark.parametrize("metric_count", [0, 2])
def test_requires_one_drift_metric(
    tmp_path: Path,
    metric_count: int,
) -> None:
    """Reject missing or ambiguous aggregate drift results."""
    payload = make_payload()
    payload["metrics"] = payload["metrics"] * metric_count
    report_path = write_report(tmp_path, payload)

    with pytest.raises(ValueError, match="exactly one"):
        summarize_report(report_path)


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("config", "columns", []),
        ("config", "columns", None),
        ("config", "columns", ["a", "a"]),
        ("config", "drift_share", 0),
        ("config", "drift_share", 1.1),
        ("config", "drift_share", float("inf")),
        ("value", "count", -1),
        ("value", "count", 7),
        ("value", "count", 1.5),
        ("value", "count", float("nan")),
        ("value", "share", -0.1),
        ("value", "share", 1.1),
        ("value", "share", float("nan")),
        ("value", "share", 0.1),
    ],
)
def test_rejects_invalid_summary(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    """Reject invalid or inconsistent values instead of reporting success."""
    payload = make_payload()
    payload["metrics"][0][section][field] = value
    report_path = write_report(tmp_path, payload)

    with pytest.raises(ValueError):
        summarize_report(report_path)


def test_save_summary(tmp_path: Path) -> None:
    """Persist a readable summary without changing the source report."""
    report_path = write_report(tmp_path, make_payload())
    original = report_path.read_bytes()

    summary_path = save_summary(report_path)

    assert summary_path == tmp_path / "summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["drifted_column_count"] == 4
    assert saved["threshold_reached"] is True
    assert report_path.read_bytes() == original
