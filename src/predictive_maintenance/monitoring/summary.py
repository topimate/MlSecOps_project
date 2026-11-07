"""Extract a compact drift summary from an Evidently report."""

import json
from math import isclose, isfinite
from pathlib import Path


def summarize_report(report_path: Path) -> dict[str, int | float | bool]:
    """Read and validate the aggregate drift metric."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    matches = [
        metric for metric in payload["metrics"]
        if metric["config"]["type"]
        == "evidently:metric_v2:DriftedColumnsCount"
    ]

    if len(matches) != 1:
        raise ValueError("Expected exactly one aggregate drift metric.")

    metric = matches[0]
    columns = metric["config"]["columns"]

    if not isinstance(columns, list) or not columns:
        raise ValueError("The monitored column list is missing or empty.")

    if len(set(columns)) != len(columns):
        raise ValueError("The monitored column list contains duplicates.")

    count = float(metric["value"]["count"])
    share = float(metric["value"]["share"])
    threshold = float(metric["config"]["drift_share"])

    if not all(isfinite(value) for value in (count, share, threshold)):
        raise ValueError("Drift summary values must be finite.")

    if not count.is_integer() or not 0 <= count <= len(columns):
        raise ValueError("Invalid drifted column count.")

    if not 0 <= share <= 1 or not 0 < threshold <= 1:
        raise ValueError("Invalid drift share or decision threshold.")

    if not isclose(share, count / len(columns), abs_tol=1e-9):
        raise ValueError("Drift count and share are inconsistent.")

    return {
        "column_count": len(columns),
        "drifted_column_count": int(count),
        "drifted_column_share": share,
        "drift_share_threshold": threshold,
        "threshold_reached": share >= threshold,
    }


def save_summary(report_path: Path) -> Path:
    """Save the validated summary next to the Evidently report."""
    summary = summarize_report(report_path)
    summary_path = report_path.with_name("summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary_path
