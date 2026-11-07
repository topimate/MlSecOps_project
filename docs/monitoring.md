# Evidently monitoring

## Purpose

Compare reference and current model input distributions, inspect data quality,
and export reports for later dashboard and retraining integration.

The component currently runs on file-based data. It does not consume live API
logs or automatically retrain or promote models.

## Environment

Install the project requirements in the activated virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

Monitoring uses Evidently 0.7.23 and pandas 2.3.3. With Evidently 0.7.23,
pandas 3.1.0 caused an import error involving `infer_dtype_from_object`.

## Input data

Restore the versioned dataset using the documented DVC setup:

```powershell
dvc pull -r dagshub
```

Expected file:

```text
data/raw/predictive_maintenance_5000.csv
```

The runner reuses the existing reproducible, stratified split with seed 42:

- Reference: 3,000 training rows.
- Current: 750 reserved future rows.
- Validation and final-test partitions are not used for monitoring.

The current sample simulates future observations; it is not a chronological
production dataset.

Only the six model input features are monitored. Identifiers, target labels,
failure types, and the derived temperature difference are excluded.

## Running the demonstration

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_monitoring_report.py
```

Two scenarios are generated:

| Scenario | Current data | Observed drift |
| --- | --- | --- |
| Baseline | Original future sample | 0 of 6 features |
| Simulated drift | Modified copy of the future sample | 4 of 6 features |

The simulated shifts are:

- Air temperature: +10 K.
- Process temperature: +10 K.
- Rotational speed: multiplied by 1.4.
- Torque: +20 Nm.

Product type and tool wear remain unchanged. The original data is not modified.

## Detection rules

- Numerical features: normalized Wasserstein distance, threshold 0.1.
- Categorical feature: Jensen–Shannon distance, threshold 0.1.
- Summary decision: `threshold_reached` is true when at least 50% of monitored
  features are flagged as drifted.

These are initial demonstration thresholds. They have not been calibrated
against production observations. Data drift alone does not establish model
performance degradation or justify automatic model replacement.

## Outputs

Each scenario directory contains:

- `report.html`: interactive data quality and drift report.
- `report.json`: Evidently metric results.
- `summary.json`: validated aggregate drift summary.

Directories:

```text
reports/baseline/
reports/simulated_drift/
```

Generated reports are Git-ignored and overwritten when the demonstration runs
again.

Summary fields:

| Field | Meaning |
| --- | --- |
| `column_count` | Number of monitored features |
| `drifted_column_count` | Number flagged as drifted |
| `drifted_column_share` | Fraction flagged as drifted |
| `drift_share_threshold` | Aggregate decision threshold |
| `threshold_reached` | Result of the aggregate decision |

The summary reader rejects missing or ambiguous aggregate metrics and invalid
or inconsistent counts, shares, and thresholds.

## Components

- `monitoring/drift_report.py`: report calculation and export.
- `monitoring/simulation.py`: deterministic demonstration shifts.
- `monitoring/summary.py`: aggregate result validation and export.
- `scripts/run_monitoring_report.py`: dataset loading and demonstration runner.

Module paths above are relative to `src/predictive_maintenance/`.

## Validation

Local validation on Windows with Python 3.12.10:

- Full suite: 144 tests passed.
- Source coverage: 99.47%.
- Combined source and test coverage: 99.76%.
- Monitoring source modules: 100% line coverage.
- Flake8 and Pyright: no errors.

Monitoring tests cover decision boundaries, invalid summaries, real Evidently
exports, controlled drift detection, and preservation of input data.

Dependency deprecation warnings remain. CI and Linux/Python 3.11 validation
must be checked separately.
