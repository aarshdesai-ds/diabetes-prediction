"""Pipeline behaviour tests: preprocessing correctness and model sanity."""
import numpy as np
import pandas as pd

from src import config, data
from src.pipelines import _zeros_to_nan, build_pipelines


def test_zeros_to_nan_only_targets_invalid_columns():
    df = pd.DataFrame({
        "Glucose": [0, 100],
        "BloodPressure": [0, 70],
        "Insulin": [0, 50],
        "BMI": [0.0, 25.0],
        "DiabetesPedigreeFunction": [0.0, 0.5],  # 0 is valid-ish here, must stay
        "Age": [0, 30],                            # not in ZERO_AS_MISSING
    })
    out = _zeros_to_nan(df)
    assert np.isnan(out.loc[0, "Glucose"])
    assert np.isnan(out.loc[0, "BMI"])
    # Columns NOT in ZERO_AS_MISSING keep their zeros.
    assert out.loc[0, "Age"] == 0
    assert out.loc[0, "DiabetesPedigreeFunction"] == 0.0


def test_all_pipelines_fit_and_predict_proba():
    df = data.load_raw()
    X, y = data.split_X_y(df)
    for name, pipe in build_pipelines().items():
        pipe.fit(X, y)
        proba = pipe.predict_proba(X.head(5))
        assert proba.shape == (5, 2), name
        assert np.all((proba >= 0) & (proba <= 1)), name


def test_pipeline_handles_zero_input_without_error():
    """A patient row full of zeros must not crash inference (zeros -> imputed)."""
    df = data.load_raw()
    X, y = data.split_X_y(df)
    pipe = build_pipelines()["Logistic Regression"].fit(X, y)
    zero_row = pd.DataFrame([{f: 0 for f in config.FEATURES}])
    proba = pipe.predict_proba(zero_row)
    assert proba.shape == (1, 2)
