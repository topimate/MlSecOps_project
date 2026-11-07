"""Generate Evidently data quality and drift reports."""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from predictive_maintenance.data.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    DataProcessor,
)

from predictive_maintenance.monitoring.summary import save_summary


class DriftReporter:
    """Compare reference and current model input features."""

    def save(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Calculate and save a report as HTML and JSON."""
        processor = DataProcessor()
        reference_features = processor.select_features(reference)
        current_features = processor.select_features(current)

        definition = DataDefinition(
            numerical_columns=[
                column for column in FEATURE_COLUMNS
                if column not in CATEGORICAL_FEATURES
            ],
            categorical_columns=list(CATEGORICAL_FEATURES),
        )

        reference_dataset = Dataset.from_pandas(
            reference_features,
            data_definition=definition,
        )
        current_dataset = Dataset.from_pandas(
            current_features,
            data_definition=definition,
        )

        report = Report([
            DataSummaryPreset(),
            DataDriftPreset(
                columns=list(FEATURE_COLUMNS),
                num_method="wasserstein",
                num_threshold=0.1,
                cat_method="jensenshannon",
                cat_threshold=0.1,
                drift_share=0.5,
            ),
        ])

        snapshot = report.run(
            current_data=current_dataset,
            reference_data=reference_dataset,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "report.html"
        json_path = output_dir / "report.json"

        snapshot.save_html(str(html_path))
        snapshot.save_json(str(json_path))
        save_summary(json_path)

        return html_path, json_path
