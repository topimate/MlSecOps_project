# MlSecOps_project
MlSecOps beadando @beni and @anna

```text
MlSecOps_project/
├── app/                         # Streamlit dashboard
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Preprocessed datasets
├── docs/                        # Project documentation
├── infra/                       # Docker and infrastructure configuration
├── src/
│   └── predictive_maintenance/
│       ├── api/                 # REST API endpoints and schemas
│       ├── data/                # Data loading and preprocessing
│       ├── models/              # ML training, prediction and MLflow
│       └── monitoring/          # Drift detection, logging and retraining
├── tests/                       # Unit and integration tests
└── README.md                    # Project overview
```

## Data versioning with DVC

GitHub stores source code, configuration, documentation, tests, and DVC
metadata. The raw dataset is stored in the shared DagsHub DVC remote.

- Storage project: https://dagshub.com/bsumicz/MlSecOps_project
- DVC remote name: `dagshub` (default)
- Tracked metadata: `data/raw.dvc`
- Dataset path: `data/raw/predictive_maintenance_5000.csv`

Do not commit raw dataset files or credentials to Git.
Processed-data versioning will be added when processed datasets are persisted.

### Local setup (PowerShell)

Run commands from the repository root with the project virtual environment
activated. The CI environment uses Python 3.11.

Install the project dependencies and development tools:

```powershell
python -m pip install -r requirements.txt
python -m pip install "dvc[s3]==3.67.1" "pre-commit==4.7.0"
```

Each team member should create a DagsHub account and obtain access to the
storage project. Use your own DagsHub access token.

Enter the token in the Password field of the credential dialog:

```powershell
$dagshubCredential = Get-Credential -UserName "token" -Message "Enter your DagsHub token in the Password field"
```

After confirming the dialog, configure local authentication:

```powershell
dvc remote modify --local dagshub access_key_id ($dagshubCredential.GetNetworkCredential().Password)
dvc remote modify --local dagshub secret_access_key ($dagshubCredential.GetNetworkCredential().Password)
Remove-Variable dagshubCredential
```

Credentials are stored in the Git-ignored `.dvc/config.local` file.
Never commit this file or share your token.

Download the dataset version referenced by the current Git checkout:

```powershell
dvc pull -r dagshub
dvc status
```

### Updating tracked data

After intentionally changing the raw dataset:

```powershell
dvc add data/raw
dvc push -r dagshub
git diff -- data/raw.dvc
```

Review and commit the updated DVC metadata. Upload the corresponding data
objects before pushing the Git commit.

### CI data access

GitHub Actions installs DVC with S3 support and downloads the dataset before
running tests. Authentication uses the `DAGSHUB_TOKEN` repository secret.

Missing credentials, failed downloads, or a missing expected CSV fail the
workflow. Dataset-dependent integration tests are not conditionally skipped.

## Pre-commit checks

Install the Git hook separately in each local clone:

```powershell
python -m pre_commit install
```

Run all checks manually:

```powershell
python -m pre_commit run --all-files
```

Checks cover trailing whitespace, file endings, YAML syntax, merge conflict
markers, large added files, and Flake8 using the existing `.flake8` rules.

If a hook modifies files, review the changes, stage the corrected files,
and retry the commit. Full tests and coverage remain separate checks.

## Running the API locally

Run the following commands from the project root.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start the development server

```powershell
python -m uvicorn predictive_maintenance.api.main:app --app-dir src --reload
```

The `--app-dir src` option makes the application package available for import.
The `--reload` option restarts the development server when source files change.

Once the server is running:

- API base URL: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI specification: http://127.0.0.1:8000/openapi.json

Stop the server with `Ctrl+C`.

## Available endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Check that the API is responding. |
| POST | `/predict` | Validate machine data and return a fixed mock prediction. |
| POST | `/train` | Validate training settings and return a simulated completion result. |

Use **Try it out** in Swagger UI to send the provided example requests.

Successful requests return HTTP `200`.
Invalid request bodies return HTTP `422` with validation details.

## Current mock behavior

The API currently runs without a dataset, trained model, or MLflow service.

- `/health` returns `status: "ok"` and `mode: "mock"`.
- `/predict` always returns prediction `0`, failure probability `0.1`,
  model version `"mock-v1"`, and mode `"mock"` for valid input.
- `/train` returns status `"mock_completed"`, a null training run ID,
  and mode `"mock"`.
- Training does not create or update a model.
- Prediction values are simulated and must not be interpreted as real model results.

## Running tests

```powershell
python -m pytest -v
```

The tests cover the health endpoint, successful mock prediction and training
requests, and invalid request bodies.

A running Uvicorn server is not required: the tests use FastAPI's `TestClient`
to call the application directly.

For request fields, response formats, defaults, and validation rules,
see the [API contract](docs/api_contract.md).
