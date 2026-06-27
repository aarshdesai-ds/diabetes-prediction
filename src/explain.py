"""SHAP explainability for the deployed model.

The shipped model is a calibrated pipeline (preprocessing + classifier wrapped
in CalibratedClassifierCV), so we use a model-agnostic SHAP KernelExplainer
over ``predict_proba``. With only six features, exact Shapley values are cheap.

Two views are exposed:
  - ``explain_one`` — per-patient feature contributions (local explanation),
  - ``global_importance`` — mean |SHAP| across a sample (global explanation).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from . import config, data
from .inference import load_model

# Exact for 6 features (2**6 = 64 coalitions); cheap and deterministic.
_NSAMPLES = min(2 ** len(config.FEATURES), 512)
_BACKGROUND_K = 10  # k-means summary of the training data used as the baseline


def _predict_proba_np(arr: np.ndarray) -> np.ndarray:
    """SHAP passes a raw ndarray; the pipeline needs named columns."""
    model = load_model()
    return model.predict_proba(pd.DataFrame(arr, columns=config.FEATURES))[:, 1]


@lru_cache(maxsize=1)
def _explainer():
    X, _ = data.split_X_y(data.load_raw())
    background = shap.kmeans(X, _BACKGROUND_K)
    return shap.KernelExplainer(_predict_proba_np, background)


def explain_one(features: dict) -> dict:
    """Return a local explanation for one patient.

    Keys: ``base_value`` (population baseline probability), ``prediction``
    (this patient's probability), and ``contributions`` (per-feature SHAP value,
    positive = pushes risk up).
    """
    explainer = _explainer()
    row = np.array([[features[f] for f in config.FEATURES]], dtype=float)
    shap_values = np.asarray(explainer.shap_values(row, nsamples=_NSAMPLES, silent=True)).reshape(-1)
    base = float(np.ravel(explainer.expected_value)[0])
    return {
        "base_value": base,
        "prediction": float(base + shap_values.sum()),
        "contributions": dict(zip(config.FEATURES, shap_values.tolist(), strict=False)),
    }


@lru_cache(maxsize=1)
def global_importance(sample_size: int = 80) -> dict:
    """Mean absolute SHAP value per feature across a sample (global importance)."""
    X, _ = data.split_X_y(data.load_raw())
    sample = X.sample(min(sample_size, len(X)), random_state=config.RANDOM_STATE)
    explainer = _explainer()
    shap_values = np.asarray(
        explainer.shap_values(sample.to_numpy(), nsamples=_NSAMPLES, silent=True)
    )
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(zip(config.FEATURES, mean_abs.tolist(), strict=False))
