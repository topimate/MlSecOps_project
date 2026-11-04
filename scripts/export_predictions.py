"""Export new or selected historical prediction events to MLflow."""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from mlflow.exceptions import MlflowException

from predictive_maintenance.monitoring.prediction_export import (
    PROJECT_ROOT,
    ExportSelection,
    export_prediction_events,
    parse_export_time,
)


def export_time_argument(value: str) -> datetime:
    """Turn invalid dates into a helpful command-line argument error."""
    try:
        return parse_export_time(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    """Run one export and print its run/artifact identifiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.environ.get("PREDICTION_LOG_DB", str(PROJECT_ROOT / "artifacts" / "predictions.db"))),
    )
    parser.add_argument("--tracking-uri", help="Defaults to MLFLOW_TRACKING_URI or http://127.0.0.1:5000.")
    parser.add_argument("--all", dest="include_exported", action="store_true", help="Create a historical snapshot.")
    parser.add_argument("--model-uri", help="Select events for this exact model URI.")
    parser.add_argument("--since", type=export_time_argument, help="Inclusive ISO timestamp with timezone.")
    parser.add_argument("--until", type=export_time_argument, help="Exclusive ISO timestamp with timezone.")
    arguments = parser.parse_args(argv)
    try:
        selection = ExportSelection(
            include_exported=arguments.include_exported,
            model_uri=arguments.model_uri,
            since=arguments.since,
            until=arguments.until,
        )
        result = export_prediction_events(arguments.database, arguments.tracking_uri, selection)
    except (OSError, sqlite3.Error, MlflowException, ValueError, RuntimeError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result or {"status": "no_matching_events"}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    raise SystemExit(main())
