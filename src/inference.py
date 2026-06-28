"""Inference helpers used by the Streamlit app.

The app never trains on a button click. It loads the pre-trained pipeline
artifact and only runs ``predict_proba``. If the artifact is missing (e.g. a
fresh clone that has not run training yet), we train it once and cache it, so
the app still works out of the box.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config
from .train import train_and_select


def _ensure_artifacts() -> None:
    """Train exactly once if any artifact is missing (fresh clone / first request)."""
    if not (config.MODEL_PATH.exists() and config.METRICS_PATH.exists()):
        train_and_select(verbose=False)


@lru_cache(maxsize=1)
def load_model():
    """Load the fitted pipeline, (re)training if the artifact is missing or unusable.

    The retrain-on-failure path also covers the case where the file on disk is an
    un-materialised Git LFS pointer (e.g. a checkout without ``git lfs pull``) or
    a pickle from an incompatible scikit-learn version.
    """
    _ensure_artifacts()
    try:
        return joblib.load(config.MODEL_PATH)
    except Exception:
        train_and_select(verbose=False)
        return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    _ensure_artifacts()
    try:
        return json.loads(config.METRICS_PATH.read_text())
    except json.JSONDecodeError:
        train_and_select(verbose=False)
        return json.loads(config.METRICS_PATH.read_text())


def predict_one(features: dict, threshold: float = config.DEFAULT_THRESHOLD):
    """Predict on a single patient's feature dict.

    Returns ``(label, probability)`` where label is 1 (at risk) / 0, decided by
    comparing the model's diabetic probability to ``threshold``.
    """
    model = load_model()
    row = pd.DataFrame([{f: features[f] for f in config.FEATURES}])
    proba = float(model.predict_proba(row)[0, 1])
    label = int(proba >= threshold)
    return label, proba
