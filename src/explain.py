"""Explainability for the deployed model.

Two complementary views:
  - ``explain_one`` — per-patient SHAP contributions (local explanation). Uses a
    model-agnostic SHAP KernelExplainer over ``predict_proba`` on a single row.
  - ``global_importance`` — permutation importance across the dataset (global
    explanation). Permutation importance is used here rather than batched SHAP
    because it is robust across library versions and environments (batched
    KernelExplainer can break on some shap/numpy combinations).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from . import config, data
from .inference import load_model

_BACKGROUND_K = 10  # k-means summary of the training data used as the SHAP baseline


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
    """Return a local SHAP explanation for one patient.

    Keys: ``base_value`` (population baseline probability), ``prediction``
    (this patient's probability), and ``contributions`` (per-feature SHAP value,
    positive = pushes risk up).
    """
    explainer = _explainer()
    row = np.array([[features[f] for f in config.FEATURES]], dtype=float)
    shap_values = np.asarray(explainer.shap_values(row, silent=True)).reshape(-1)
    base = float(np.ravel(explainer.expected_value)[0])
    return {
        "base_value": base,
        "prediction": float(base + shap_values.sum()),
        "contributions": dict(zip(config.FEATURES, shap_values.tolist(), strict=False)),
    }


@lru_cache(maxsize=1)
def global_importance() -> dict:
    """Permutation importance per feature (drop in ROC AUC when shuffled).

    Robust and model-agnostic. Values are the mean AUC decrease across repeats;
    larger = more important. Small negative values (noise) are possible.
    """
    X, y = data.split_X_y(data.load_raw())
    model = load_model()
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=10,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return dict(zip(config.FEATURES, result.importances_mean.tolist(), strict=False))
