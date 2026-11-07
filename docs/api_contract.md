# API Contract

This document describes the current mock API.

Base URL for local development: `http://127.0.0.1:8000`

Interactive documentation: `/docs`
OpenAPI specification: `/openapi.json`

POST endpoints accept JSON request bodies.

## GET /health

Checks whether the API is responding. It does not check model availability
or external services.

### Successful response — 200 OK

```json
{
  "status": "ok",
  "mode": "mock"
}
```

## POST /predict

Accepts one machine-data record and returns a fixed mock prediction.

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

The schemas use Pydantic's default type conversion rather than strict
type validation. For example, a numeric string such as `"300.8"` can be
converted to a number. Clients should send measurements as JSON numbers.

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

```json
{
  "prediction": 0,
  "probability": 0.1,
  "model_version": "mock-v1",
  "mode": "mock"
}
```

| Field | Meaning |
| --- | --- |
| `prediction` | `0`: no failure; `1`: failure |
| `probability` | Probability of failure, between 0 and 1 |
| `model_version` | Identifier of the model producing the result |
| `mode` | Execution mode; currently always `"mock"` |

In the current implementation, every valid request returns the response
shown above. The input is validated but is not used to calculate a prediction.

The returned probability is a fixed test value, not a real model estimate.

## POST /train

Accepts training settings and simulates successful completion.

### Request fields

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `dataset_id` | string | `"default"` | Only `"default"` is accepted |
| `random_state` | integer | `42` | Must be at least 0 |

Additional fields and explicit null values are rejected.
Pydantic's default type conversion also applies to this request.

`dataset_id` is a logical dataset identifier, not a file path.
The mock implementation does not load a dataset.

`random_state` is reserved for reproducible training. The mock
implementation validates it but does not use it in a training operation.

### Example request

```json
{
  "dataset_id": "default",
  "random_state": 42
}
```

An empty JSON object is also accepted and uses the defaults:

```json
{}
```

The request body itself is required. Omitting the body is different
from sending an empty JSON object and returns HTTP `422`.

### Successful response — 200 OK

```json
{
  "status": "mock_completed",
  "training_run_id": null,
  "mode": "mock"
}
```

| Field | Meaning |
| --- | --- |
| `status` | `"mock_completed"` indicates simulated completion |
| `training_run_id` | Reserved for a future MLflow run ID; currently `null` |
| `mode` | Execution mode; currently always `"mock"` |

The operation completes during the request. It does not start a background job,
create an MLflow run, train a model, or change subsequent predictions.

## Validation errors — 422 Unprocessable Entity

Invalid request bodies return HTTP `422`.

Examples include:

- Missing required prediction fields.
- A missing POST request body.
- Unsupported product types or dataset identifiers.
- Values outside the allowed ranges.
- Values that cannot be converted to the required types.
- Additional fields not defined by the request schema.

The response contains a `detail` list with information about each
validation error.

Typical error properties:

| Property | Meaning |
| --- | --- |
| `loc` | Location of the invalid value, such as `["body", "torque_nm"]` |
| `type` | Validation error identifier |
| `msg` | Human-readable explanation |

Additional properties may be included. Clients should not depend on
the exact wording of `msg`.

## Mock scope and future integration

The current API requires no dataset, trained model, MLflow service,
or persistent storage.

The request and response structures provide a starting point for real
model integration. Before integration, the team must confirm the feature
mapping, class meanings, preprocessing interface, and training behavior.

Any changes to the public API contract must also be reflected in the
schemas, examples, tests, and this document.
