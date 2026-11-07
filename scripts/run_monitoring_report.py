"""Generate the baseline monitoring report from fixed data splits."""

from pathlib import Path

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.monitoring.drift_report import DriftReporter
from predictive_maintenance.monitoring.simulation import simulate_feature_drift


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Compare training features with the reserved future sample."""
    dataset_path = (
        PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_5000.csv"
    )
    raw_data = load_raw_data(dataset_path)

    processor = DataProcessor()
    normalized_data = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized_data)
    splits = create_initial_data_splits(features, target)

    html_path, json_path = DriftReporter().save(
        reference=splits.X_train,
        current=splits.X_future,
        output_dir=PROJECT_ROOT / "reports" / "baseline",
    )

    print(f"Reference rows: {len(splits.X_train)}")
    print(f"Current rows: {len(splits.X_future)}")
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")

    shifted_current = simulate_feature_drift(splits.X_future)

    shifted_html, shifted_json = DriftReporter().save(
        reference=splits.X_train,
        current=shifted_current,
        output_dir=PROJECT_ROOT / "reports" / "simulated_drift",
    )

    print(f"Simulated drift HTML: {shifted_html}")
    print(f"Simulated drift JSON: {shifted_json}")


if __name__ == "__main__":
    main()
