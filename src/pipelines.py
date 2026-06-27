"""Model pipeline definitions.

Each candidate model is a full scikit-learn :class:`~sklearn.pipeline.Pipeline`
that bundles preprocessing with the estimator. This is the key architectural
fix over the original project: because preprocessing lives *inside* the
pipeline, the exact same zero->NaN replacement, imputation, and scaling that
were used in training are applied automatically to every prediction. There is
no way for training and serving to drift apart.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config


def _zeros_to_nan(X):
    """Replace 0 with NaN in the columns where 0 means 'not measured'."""
    X = X.copy()
    cols = [c for c in config.ZERO_AS_MISSING if c in X.columns]
    X[cols] = X[cols].replace(0, np.nan)
    return X


def _build_preprocessor(scale: bool) -> Pipeline:
    """Zero-cleaning + median imputation, optionally followed by scaling."""
    steps = [
        ("zeros_to_nan", FunctionTransformer(_zeros_to_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_pipelines() -> dict[str, Pipeline]:
    """Return the candidate models, each as a complete preprocessing+estimator pipeline.

    ``class_weight="balanced"`` (where supported) addresses the 65/35 class
    imbalance so the models do not simply learn to predict the majority class.
    """
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", _build_preprocessor(scale=True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("prep", _build_preprocessor(scale=False)),
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=rs)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", _build_preprocessor(scale=False)),
                ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=rs)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", _build_preprocessor(scale=False)),
                ("clf", GradientBoostingClassifier(random_state=rs)),
            ]
        ),
    }


def param_grids() -> dict[str, dict]:
    """Hyperparameter search spaces for each model (keys match ``build_pipelines``)."""
    return {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
        "Decision Tree": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
        },
        "Random Forest": {
            "clf__max_depth": [4, 6, 8, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
            "clf__n_estimators": [100, 200, 300],
        },
    }
