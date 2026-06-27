"""Data loading and validation.

The cleaning of zero-as-missing values happens *inside* the model pipeline
(see ``pipelines.py``) rather than here, so that the exact same imputation is
applied automatically at inference time. This module just loads and sanity
checks the raw data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Pima Indians Diabetes CSV and validate its schema."""
    path = Path(path) if path is not None else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected the Pima Indians "
            f"Diabetes CSV (diabetes.csv) in the project root."
        )

    df = pd.read_csv(path)

    missing = set(config.RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    if df[config.TARGET].dropna().nunique() != 2:
        raise ValueError("Target column 'Outcome' must be binary (0/1).")

    return df


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model feature matrix X and target vector y."""
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y
