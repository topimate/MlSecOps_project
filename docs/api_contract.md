# API Contract

This document describes real model training with MLflow tracking and real prediction using a configured fitted pipeline.

Base URL for local development: `http://127.0.0.1:8000`

Interactive documentation: `/docs`  
OpenAPI specification: `/openapi.json`

POST endpoints accept JSON request bodies.

## GET /health

Checks whether the API is responding. It does not check dataset availability, model availability, or external services.

### Successful response — 200 OK

```json
{
  "status": "ok"
}
```

## POST /predict

Accepts one machine-data record and returns a prediction calculated by the configured fitted preprocessing and classifier pipeline.

### Request fields

All fields are required. Null values and additional fields are rejected.

| Field | Type | Constraints | Meaning |
| --- | --- | --- | --- |
| `type` | string | `"L"`, `"M"` or `"H"` | Product type |
| `air_temperature_k` | number | Finite, greater than 0 | Air temperature in kelvin |
| `process_temperature_k` | number | Finite, greater than 0 | Process temperature in kelvin |
| `rotational_speed_rpm` | number | Finite, at least 0 | Rotational speed in revolutions per minute |
| `torque_nm` | number | Finite, at least 0 | Torque in newton-metres |
| `tool_wear_min` | number | Finite, at least 0 | Tool usage time in minutes |

The numeric fields use Pydantic's default type conversion rather than strict type validation. For example, a numeric string such as `"300.8"` can be converted to a number. Clients should send measurements as JSON numbers.

### Example request

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

### Successful response — 200 OK

The following response was observed locally for the example request using the indicated logged model. Other models or inputs can return different values.

```json
{
  "prediction": 0,
  "probability": 0.012770727629820311,
  "model_version": "m-6215dba09ce64077bb59887734445acf",
  "mode": "real"
}
```

| Field | Meaning |
| --- | --- |
| `prediction` | `0`: no failure; `1`: failure |
| `probability` | Probability of failure, between 0 and 1 |
| `model_version` | Configured model URI with the `models:/` prefix removed; for logged-model URIs this is an `m-...` identifier, not a Registry version number |
| `mode` | Always `"real"` for this endpoint |

### Prediction behavior and model configuration

The server reads `MODEL_URI` from its environment. This variable is required for prediction and must identify a trusted fitted sklearn Pipeline accessible through the configured MLflow backend. Clients cannot supply a model URI in the request.

`MLFLOW_TRACKING_URI` selects the MLflow backend; its default is `http://127.0.0.1:5000`.

The service builds a single-row DataFrame using the six normalized feature names in the model's expected order. The loaded pipeline performs preprocessing internally; it is not fitted again during prediction.

The trained binary classifier uses class order `[0, 1]`. The service reads the positive-class probability from `predict_proba`, requires an output shape of `(1, 2)`, and rejects non-finite values or probabilities outside `[0, 1]`.

The failure prediction is `1` when the failure probability is greater than or equal to `0.5`; otherwise it is `0`.

The most recently loaded `(model_uri, tracking_uri)` pair is cached in memory per process. Repeated requests for that pair reuse the pipeline without reloading it from MLflow. The first load, a different pair, or an API process restart requires artifact access again. Changing `MODEL_URI` in a terminal does not update a running API process: set the variable and restart the API.

Prediction does not load the training CSV. It uses the model and preprocessing already saved in MLflow. Running `/train` does not automatically update `MODEL_URI` or activate the newly trained model.

### Operational errors

| Status | Detail |
| --- | --- |
| `503` | `The prediction model is unavailable.` |
| `500` | `Prediction failed because of invalid model data.` |
| `503` | `The prediction event could not be stored.` |

The `503` handler covers the missing `MODEL_URI` configuration and handled MLflow, runtime, or operating-system failures. The `500` handler covers `ValueError`, including an incompatible loaded object or invalid probability output. These responses contain a string-valued `detail`, for example:

```json
{
  "detail": "The prediction model is unavailable."
}
```

Unexpected failures may also result in HTTP `500`.

### Persistent prediction events

Each request that passes input validation receives a server-generated UUID.
For successful predictions and handled prediction failures, the API commits
an event to a local SQLite database and returns the UUID in the `X-Request-ID`
response header. The JSON response schema is unchanged.

The event is stored in `prediction_events`, with `request_id` as the unique
primary key and `payload` containing JSON. Event fields are:

| Field | Meaning |
| --- | --- |
| `request_id` | UUID matching the response header |
| `timestamp` | Request processing start time in UTC ISO 8601 format |
| `inputs` | The six validated request features |
| `model_uri` | Configured model URI, or null when configuration is missing |
| `decision_threshold` | `0.5` |
| `status` | `success` or `error` |
| `http_status` | Intended prediction status: `200`, `503`, or `500` |
| `prediction`, `probability`, `model_version`, `mode` | Successful prediction result fields |
| `error` | Public error message for a handled prediction failure |

Internal exception messages and stack traces are not stored in these events.
Requests rejected with `422` do not reach this logging step. Unhandled failures
are not covered by this event logger.

`PREDICTION_LOG_DB` configures the SQLite file. Its default is the repository's
`artifacts/predictions.db`; the parent directory and table are created on demand.
Committed events remain available when the API restarts with the same file.
The generated database must not be committed to Git. Container deployments
must put it on persistent storage to survive container replacement.

If the event cannot be committed, the API returns `503` with
`The prediction event could not be stored.` and an `X-Request-ID` header.
This also takes precedence if persistence fails while recording a handled
prediction error. No event can be guaranteed in a storage-failure case.
There is no automatic retry or retention policy. A separate command exports persisted events to MLflow as described below.

## Prediction artifact export (command-line operation)

`python scripts/export_predictions.py` exports a snapshot of persisted events
to the `predictive-maintenance-predictions` MLflow experiment. Run from the
repository root with `PYTHONPATH=src`. This is not an HTTP endpoint and is not
automatically executed by `/predict`.

| Option | Behavior |
| --- | --- |
| `--database PATH` | Override `PREDICTION_LOG_DB` / the default `artifacts/predictions.db` |
| `--tracking-uri URI` | Override `MLFLOW_TRACKING_URI` / the default `http://127.0.0.1:5000` |
| `--all` | Explicitly create a historical snapshot, including already uploaded events |
| `--model-uri URI` | Select the exact stored `model_uri` |
| `--since TIMESTAMP` | Inclusive event timestamp boundary, with timezone |
| `--until TIMESTAMP` | Exclusive event timestamp boundary, with timezone |

Default mode selects events not included in a completed incremental batch for
the same tracking URI string. Both success and error events are eligible.
Historical snapshots never remove events or alter incremental eligibility.
No matching events means no remote run and output `{"status": "no_matching_events"}`.

Each non-empty batch stores an immutable JSON snapshot and unique `batch_id`
in SQLite before contacting MLflow. Its artifact path is
`predictions/predictions.json`. The document contains `schema_version`,
`batch_id`, `created_at`, `selection`, `event_count`, and an `events` array.
The original event fields are preserved, including `request_id`.

Successful output includes `run_id` and a downloadable URI in the form
`runs:/<run-id>/predictions/predictions.json`. The run has `operation=predict`
and `batch_id` tags, an event count, model URI information, and an export mode.
Any previous artifact remains independently downloadable from MLflow UI.

Incremental bookkeeping is completed only after artifact upload and remote
run completion. Failures leave the local batch pending. Retrying the same
selection reuses its snapshot, batch ID, remote run, and artifact path.
Remote runs can also be recovered by batch tag if creation succeeded before
the run ID was saved locally. A finished remote run allows local recovery
without repeating its upload. This is recovery coordination between two
stores, not a distributed atomic transaction.

An unfinished selection must be retried before changing filters for the same
destination. Multiple remote runs with the same batch tag are reported as an
error rather than silently selected. Exporters are serialized using a separate
local SQLite lock file; event writes remain possible during upload. CLI errors
produce a nonzero exit status. No scheduler, pruning, or deletion is performed.

See the README for command examples and restart/retry instructions.

## POST /train

Loads the configured initial dataset, trains the selected model, evaluates it on the validation partition, and logs the results to MLflow.

### Request fields

All request fields have defaults. Additional fields and explicit null values are rejected.

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `dataset_id` | string | `"default"` | Only `"default"` is accepted |
| `model_type` | string | `"gradient_boosting"` | `"gradient_boosting"` or `"xgboost"` |
| `random_state` | integer | `42` | Only `42` is accepted |

`dataset_id` identifies the configured dataset:

```text
data/raw/predictive_maintenance_5000.csv
```

This path is relative to the repository root. It refers to the root-level data directory, not the Python modules under `src/predictive_maintenance/data/`.

Clients cannot supply an arbitrary file path.

`model_type` selects either Gradient Boosting or XGBoost.

`random_state` is restricted to the numeric literal `42`, matching the fixed initial splitting and model-training configuration. The string `"42"` is not accepted.

### Data preparation and training

The initial dataset must contain exactly 5,000 records with the original raw CSV columns.

The training service:

1. Loads and validates the raw CSV.
2. Normalizes column names.
3. Separates the six input features from the binary target.
4. Creates reproducible, stratified dataset partitions.
5. Trains and evaluates the selected model.
6. Logs the training results to MLflow.

| Partition | Records | Purpose |
| --- | ---: | --- |
| Training | 3,000 | Fit preprocessing and the classifier |
| Validation | 750 | Calculate validation metrics |
| Future | 750 | Reserved for later monitoring and retraining work |
| Final test | 500 | Reserved for final evaluation |

Training uses only the training partition. The current training operation calculates metrics on the validation partition; it does not evaluate the final-test partition.

This splitting procedure is intended for the initial dataset. Future retraining must preserve the fixed validation and final-test partitions.

### Example request

```json
{
  "dataset_id": "default",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

An empty JSON object is also accepted and uses the defaults:

```json
{}
```

The request body itself is required. Omitting it is different from sending `{}` and returns HTTP `422`.

### Successful response — 200 OK

The following response is illustrative. The identifiers and metric values depend on the actual training run.

```json
{
  "status": "completed",
  "training_run_id": "<actual-run-id>",
  "model_type": "gradient_boosting",
  "model_uri": "<actual-model-uri>",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.80,
    "recall": 0.75,
    "f1": 0.77,
    "roc_auc": 0.90,
    "average_precision": 0.85
  },
  "mode": "real"
}
```

| Field | Meaning |
| --- | --- |
| `status` | `"completed"` indicates successful completion |
| `training_run_id` | The actual MLflow training run ID |
| `model_type` | The selected classifier |
| `model_uri` | URI used to load the logged preprocessing and model pipeline |
| `metrics` | Six metrics calculated on the validation partition |
| `mode` | Always `"real"` for this endpoint |

Training is synchronous: the response is returned after the operation finishes. No background job is created.

The operation logs the following to MLflow:

- Model type, training parameters, and dataset partition sizes.
- Validation metrics.
- Git commit and experiment metadata.
- The complete fitted preprocessing and classifier pipeline.

The returned `model_uri` identifies a logged model. It does not by itself indicate that a Model Registry version or champion alias has been created.

Training does not automatically activate the new model for prediction. To use the returned model, configure its `model_uri` as the API process's `MODEL_URI` and restart the API.

### Operational requirements

The operation requires:

- The configured CSV file.
- The project's Python dependencies.
- An accessible MLflow tracking backend and writable artifact storage.

The training code uses the `MLFLOW_TRACKING_URI` environment variable when set. Otherwise, it defaults to:

```text
http://127.0.0.1:5000
```

The request cannot override the MLflow tracking URI.

### Operational errors

The route explicitly handles the following failures:

| Status | Detail |
| --- | --- |
| `503` | `The training dataset is unavailable.` |
| `503` | `The MLflow operation could not be completed.` |
| `500` | `Training failed because of invalid data or configuration.` |

These responses contain a string-valued `detail` field:

```json
{
  "detail": "The training dataset is unavailable."
}
```

Unexpected failures may also result in HTTP `500`.

## Validation errors — 422 Unprocessable Entity

Invalid request bodies return HTTP `422`.

Examples include:

- Missing required prediction fields.
- A missing POST request body.
- Unsupported product types, dataset identifiers, or model types.
- Values outside the allowed ranges.
- Values that cannot be converted to the required numeric types.
- A `random_state` value other than the numeric literal `42`.
- Additional fields not defined by the request schema.
- Explicit null values.

The response contains a `detail` list with information about each validation error.

| Property | Meaning |
| --- | --- |
| `loc` | Location of the invalid value, such as `["body", "torque_nm"]` |
| `type` | Validation error identifier |
| `msg` | Human-readable explanation |

Additional properties may be included. Clients should not depend on the exact wording of `msg`.

Request validation takes place before the endpoint function runs. Invalid training requests do not start training; invalid prediction requests do not invoke the prediction service.

## Current scope and remaining integration

Both `/train` and `/predict` are connected to real services. Model loading and per-process caching are implemented. The retained mock service functions are not called by the current routes.

The following parts are not implemented yet:

- Automatic model activation or Registry-based model selection.

Persistent local prediction logging and JSON prediction artifact export are implemented. Registry-based activation and the wider automated monitoring/retraining workflow belong to later integration work.

The `/health` endpoint checks API responsiveness only.

API unit tests replace training and prediction services with controlled results. Service tests exercise data preparation, model-loading/cache behavior, decision thresholds, missing configuration, and invalid model output while isolating external calls. They do not require the developer's running MLflow server.

Separate local Swagger calls have demonstrated successful training, loading the saved pipeline, and a real prediction. Automated MLflow integration tests additionally upload and download prediction artifacts using isolated storage. These checks do not establish remote CI or deployment status; review the latest quality-check results before merging.

Any changes to the public API contract must also be reflected in the schemas, examples, tests, and this document.
