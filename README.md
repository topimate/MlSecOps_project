# MlSecOps_project

Predictive Maintenance / MLSecOps project by Mate, Anna, and Beni.

## Project structure

```text
MlSecOps_project/
├── app/                         # Streamlit dashboard placeholder
├── data/                        # Dataset files, not Python modules
│   ├── raw/                     # Original CSV, managed separately from Git
│   ├── raw.dvc                  # DVC metadata for the raw dataset
│   └── processed/               # Processed dataset storage
├── docs/                        # API contract, workflow, and component handoffs
├── infra/                       # Infrastructure configuration placeholder
├── scripts/                     # Baseline training and model verification
├── src/
│   └── predictive_maintenance/
│       ├── api/                 # FastAPI routes, schemas, and API services
│       ├── data/                # Loading, preprocessing, and splitting code
│       ├── models/              # Training, evaluation, and MLflow tracking
│       └── monitoring/          # Monitoring package placeholder
├── tests/
│   ├── unit/                    # Component and isolated API/service tests
│   └── integration/             # Dataset and MLflow integration tests
├── requirements.txt
├── pytest.ini
└── README.md
```

The root-level `data/` directory holds datasets. References such as
`data/loader.py` in component handoffs are relative to
`src/predictive_maintenance/`, not the repository root.

## Data versioning with DVC

Git tracks source code, configuration, documentation, tests, and DVC metadata.
Dataset files must not be committed directly to Git.

Raw data is expected at:

```text
data/raw/predictive_maintenance_5000.csv
```

Use the original 5,000-row CSV, preserving its contents, row order, and original
column names. The training code creates the initial 3,000 training, 750
validation, 750 future, and 500 final-test partitions in memory with seed `42`.
Do not prepare separate split CSVs for the current initial-training workflow.

`data/raw.dvc` records the raw data version. The committed `.dvc/config` is
currently empty: it does not define a shared remote. Agree on remote access
with the team before relying on `dvc pull`. For current local development,
place the team's unchanged original CSV at the path above. It is Git-ignored.

## Running the API locally

Run all commands from the repository root. The CI target is Python 3.11.
The commands below use the Python environment selected by `python`.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start MLflow in the first terminal

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --workers 1
```

Keep this terminal running. The MLflow UI is available at
http://127.0.0.1:5000. Run metadata is stored in `mlflow.db` and artifacts in
`mlartifacts/`. These generated files are Git-ignored. Keep the same database
and artifacts between restarts to retain access to logged models.

### Start the API in a second terminal

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m uvicorn predictive_maintenance.api.main:app --app-dir src --reload
```

The `--app-dir src` option makes the application package importable.
The `--reload` option reloads the development server when source files change.
The repository root is also required because training logs `src/` with the model.

- API base URL: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI specification: http://127.0.0.1:8000/openapi.json

The server can start before `MODEL_URI` is configured. `/health` only checks
API responsiveness; `/predict` requires a configured model.

### Train and choose a prediction model

In Swagger UI, call `POST /train` with:

```json
{
  "dataset_id": "default",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

`xgboost` is also supported. Training is synchronous and returns a real
`training_run_id`, `model_uri`, and six validation metrics. Check the run and
logged model in the `predictive-maintenance-baselines` MLflow experiment.

Copy the returned `model_uri`. Stop the API with `Ctrl+C`, then configure it
in that same API terminal and restart. Replace the placeholder below with
the complete URI returned by your training run:

```powershell
$env:MODEL_URI = "models:/<model-id-from-train-response>"
python -m uvicorn predictive_maintenance.api.main:app --app-dir src --reload
```

The placeholder is not a usable model URI. Alternatively, use an existing
trusted logged model accessible through the configured MLflow backend.

Environment variables set this way apply to the current terminal and its
child processes. Set them again when starting from a new terminal.

### Make a real prediction

In Swagger UI, call `POST /predict` with:

```json
{
  "type": "L",
  "air_temperature_k": 300.8,
  "process_temperature_k": 310.3,
  "rotational_speed_rpm": 1538,
  "torque_nm": 36.1,
  "tool_wear_min": 198
}
```

The fitted pipeline performs preprocessing and calculates the failure
probability. The response uses `mode: "real"`; prediction `1` means the
probability is at least `0.5`, otherwise the prediction is `0`.

`model_version` currently contains the logged-model identifier, such as
`m-...`; it is not a Model Registry version number.

The process caches the most recently loaded model/MLflow-URI pair.
Training does not automatically change the prediction model. To switch
models, update `MODEL_URI` and restart the API. Only configure trusted models.

## Persistent prediction events

Validated `/predict` requests now produce a committed SQLite event before a
successful response is returned. The default storage path is
`artifacts/predictions.db`, which is Git-ignored. The directory and table are
created automatically on first use. Override the location in the API terminal
before starting the server when needed:

```powershell
$env:PREDICTION_LOG_DB = "C:/path/to/predictions.db"
```

The `X-Request-ID` response header identifies the stored event. Events include
a UTC timestamp, validated input features, model URI, threshold, status, and
the prediction/probability/model identifier for successful requests. Handled
prediction failures are stored with their public error message. Invalid `422`
requests are rejected before this logging step. A storage failure returns
`503` with `The prediction event could not be stored.` rather than reporting
an unaudited prediction as successful.

The JSON prediction response fields remain unchanged. Reusing the same database
file preserves committed events across API restarts. Docker persistence will
require mounting this path on persistent storage in Task 4.2. No pruning or
retention policy is implemented yet. Prediction artifacts are uploaded by the
separate export command below; a prediction request does not upload a batch.

## Export prediction snapshots to MLflow

Keep the MLflow server running. In a separate PowerShell terminal at the
repository root, configure the same tracking backend used by the API:

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python scripts/export_predictions.py
```

The default source is `artifacts/predictions.db`, or `PREDICTION_LOG_DB` when
configured. `--database` and `--tracking-uri` explicitly override those choices.
If no database or matching events exist, the command prints
`{"status": "no_matching_events"}` without creating an MLflow run.

By default, one snapshot contains all currently matching events not yet
successfully uploaded by an incremental export to that tracking URI. Both
successful predictions and logged prediction failures are included. New
events arriving after snapshot creation wait for the next batch.

The result prints `batch_id`, `run_id`, `event_count`, and `artifact_uri`.
In MLflow UI, open the `predictive-maintenance-predictions` experiment, the
indicated run, and `predictions/predictions.json` under Artifacts. Runs are
tagged with `operation=predict`, `batch_id`, event count, model URI (or a
mixed-model marker), and export mode. The JSON contains batch metadata,
selection filters, and the complete original event records.

### Historical snapshots and downloads

All original events stay in SQLite. To explicitly upload a new snapshot of
previously uploaded data, use `--all`. Each successful `--all` invocation
creates a new snapshot, even for the same selection:

```powershell
python scripts/export_predictions.py --all
python scripts/export_predictions.py --all --since "2026-11-01T00:00:00Z" --until "2026-12-01T00:00:00Z"
python scripts/export_predictions.py --all --model-uri "models:/<actual-model-id>"
```

Model matching uses the complete, exact model URI; replace the placeholder
with a real URI. Date filters use event timestamps, include `--since`, exclude
`--until`, and require an explicit timezone (`Z` means UTC). These filters can
also narrow an incremental export when `--all` is omitted. Events without a
timestamp are excluded when date filters are used. Invalid saved timestamps
produce an error rather than silently changing the selection.

Historical exports do not consume or reset the incremental queue. Downloading
any earlier artifact from MLflow UI is independent of these commands and
does not change export bookkeeping. Nothing is deleted by export or download.

### Interrupted exports and concurrent use

SQLite records a stable batch ID, immutable event snapshot, destination, and
remote run ID. After an upload failure, rerun the same command and selection.
The exporter resumes the pending batch and the same remote artifact path.
If remote run creation succeeded but its response was lost, the stored batch
tag is used to locate the run. If the remote run finished before local
completion was recorded, the retry completes local bookkeeping without
uploading the artifact again. Conflicting duplicate remote batch tags are
reported for manual resolution instead of choosing a run silently.

Complete an unfinished batch with its original filters before changing the
selection for the same destination. Use a consistent tracking URI string;
different URI strings have separate incremental bookkeeping. An explicit
historical snapshot is the supported way to upload earlier events again.

Export commands for the same local database are serialized with a separate
SQLite lock file (`predictions.db.export-lock` by default). This lock does not
block prediction-event writes during network uploads. A second exporter may
report a lock error; retry after the first command finishes. API requests can
continue writing new events. Keep the database and its export tables; deleting
local bookkeeping loses the record of which batches were completed.

No upload scheduler is installed. Export is explicitly started by the command.
Only JSON is generated; CSV is not required for the current Task 4.1 artifact.

## Available endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Check API responsiveness; returns `{"status": "ok"}`. |
| POST | `/train` | Train and evaluate a model, then log it to MLflow. |
| POST | `/predict` | Return a real prediction using the configured fitted pipeline. |

Successful requests return HTTP `200`. Invalid request bodies return `422`.
Handled operational failures return `503` or `500`; see the
[API contract](docs/api_contract.md) for exact fields and messages.

## Running tests and quality checks

The focused API and service tests do not require the real CSV, a configured
`MODEL_URI`, or a running MLflow server. They use controlled service/model
results and temporary synthetic data.

```powershell
python -m pytest tests/unit/test_api.py tests/unit/test_api_services.py tests/unit/test_prediction_store.py -v
```

Run the complete test suite after making the original CSV available:

```powershell
python -m pytest -v
```

MLflow integration tests use isolated temporary storage. Dataset integration
tests require the real CSV; not all of them skip automatically when it is absent.

Check code style and types:

```powershell
python -m flake8 src tests infra app scripts
python -m pyright
```

The shared Pyright configuration references `.venv`. If that directory is
absent, Pyright may print an environment message. Resolve any actual errors
and make sure it uses the intended Python interpreter.

## Current scope and remaining work

Real training, MLflow model logging, model loading/cache, real prediction,
input validation, and basic HTTP error handling are implemented. Local
Swagger calls have demonstrated successful training, artifact loading,
and a real prediction. Automated tests verify the API and service behavior
with isolated data and controlled external calls; a passing unit test does
not by itself establish end-to-end MLflow availability.

Persistent prediction request/result storage and its tests are now implemented.
Prediction JSON artifact export, historical selection, retry bookkeeping,
and automated artifact download verification are implemented. Final Task 4.1
review should include the local command against the shared development backend,
the latest repository-wide quality checks, and a reviewed PR.
Model Registry activation, automated retraining, monitoring, and Docker
deployment belong to later integration/infrastructure work.

See the [API contract](docs/api_contract.md) for request/response details and
the [Git workflow](docs/branching.md) for feature branches, review, and merges.
