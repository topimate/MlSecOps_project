"""Load and validate raw predictive maintenance datasets."""

import csv
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Target",
    "Failure Type",
)

NUMERIC_COLUMNS = (
    "UDI",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Target",
)

INTEGER_COLUMNS = (
    "UDI",
    "Rotational speed [rpm]",
    "Tool wear [min]",
    "Target",
)


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate the structure and data types of a raw dataset."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    if not df.columns.is_unique:
        raise ValueError("The dataset contains duplicate column names.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    _validate_numeric_columns(df)

    for column in ("Product ID", "Type", "Failure Type"):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError(f"Column '{column}' must contain strings.")

    target = df["Target"]
    if bool(target.isna().any()) or not bool(target.isin([0, 1]).all()):
        raise ValueError("Target must contain only 0 and 1, without missing values.")

    invalid_types = set(df["Type"].dropna()) - {"L", "M", "H"}
    if invalid_types:
        raise ValueError(f"Unexpected product types: {sorted(invalid_types)}")


def _validate_numeric_columns(df: pd.DataFrame) -> None:
    """Validate numeric and integer-only columns."""
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        values = df[column].dropna()

        if column in INTEGER_COLUMNS and not bool(values.mod(1).eq(0).all()):
            raise ValueError(f"Column '{column}' must contain integers.")


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from a configurable path and validate its schema."""
    path = Path(data_path)

    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), [])

        if len(header) != len(set(header)):
            raise ValueError("The CSV contains duplicate column names.")

        df = pd.read_csv(path, encoding="utf-8-sig")

    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read CSV file: {path}") from exc

    validate_raw_data(df)

    return df
