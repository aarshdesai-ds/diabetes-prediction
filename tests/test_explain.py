"""Tests for the SHAP explainability module."""
import numpy as np

from src import config
from src.explain import explain_one, global_importance

_PATIENT = {
    "Glucose": 148, "BloodPressure": 72, "Insulin": 0,
    "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
}


def test_explain_one_has_all_features():
    exp = explain_one(_PATIENT)
    assert set(exp["contributions"]) == set(config.FEATURES)
    assert 0.0 <= exp["base_value"] <= 1.0


def test_shap_values_reconstruct_the_prediction():
    """base_value + sum(contributions) should equal the model's probability."""
    exp = explain_one(_PATIENT)
    reconstructed = exp["base_value"] + sum(exp["contributions"].values())
    assert np.isclose(reconstructed, exp["prediction"], atol=1e-6)


def test_global_importance_covers_all_features():
    imp = global_importance()
    assert set(imp) == set(config.FEATURES)
    # Permutation importances are finite numbers (can be slightly negative).
    assert all(np.isfinite(v) for v in imp.values())
    # Glucose is the strongest predictor in this dataset; sanity-check it leads.
    assert imp["Glucose"] == max(imp.values())
