"""Train, evaluate, and persist the diabetes risk model.

Run with:  ``python -m src.train``

This script:
  1. loads and splits the data into train/test (stratified),
  2. tunes each candidate model with cross-validated GridSearchCV (ROC AUC),
  3. selects the best model by cross-validated ROC AUC,
  4. evaluates the winner on the held-out test set (the honest number),
  5. saves the fitted pipeline, a metrics JSON, and a model card.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import config, data
from .pipelines import build_pipelines, param_grids


def train_and_select(verbose: bool = True):
    """Run the full training + selection routine. Returns (best_name, best_estimator, metrics)."""
    df = data.load_raw()
    X, y = data.split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    pipelines = build_pipelines()
    grids = param_grids()

    leaderboard = {}
    fitted = {}
    for name, pipe in pipelines.items():
        search = GridSearchCV(
            pipe,
            grids[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        leaderboard[name] = {
            "cv_roc_auc": round(float(search.best_score_), 4),
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        }
        if verbose:
            print(f"  {name:<20} CV ROC AUC = {search.best_score_:.4f}")

    best_name = max(leaderboard, key=lambda n: leaderboard[n]["cv_roc_auc"])
    best_model = fitted[best_name]

    # --- Honest held-out test-set evaluation -------------------------------
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "generated_on": str(date.today()),
        "best_model": best_name,
        "best_params": leaderboard[best_name]["best_params"],
        "leaderboard": leaderboard,
        "test_set": {
            "n_samples": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "precision_diabetic": round(float(precision_score(y_test, y_pred)), 4),
            "recall_diabetic": round(float(recall_score(y_test, y_pred)), 4),
            "f1_diabetic": round(float(f1_score(y_test, y_pred)), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Non-diabetic", "Diabetic"], output_dict=True
            ),
        },
        "features": config.FEATURES,
        "n_total_records": int(len(df)),
    }

    # Refit the winning pipeline on ALL data before shipping, so the deployed
    # model uses every available record. (Metrics above still come from the
    # held-out test set, which is the honest estimate of generalisation.)
    best_model.fit(X, y)

    _save_artifacts(best_model, metrics)
    if verbose:
        t = metrics["test_set"]
        print(f"\nSelected: {best_name}")
        print(f"  Held-out test ROC AUC : {t['roc_auc']}")
        print(f"  Recall (diabetic)     : {t['recall_diabetic']}")
        print(f"  Artifacts written to  : {config.MODELS_DIR}")

    return best_name, best_model, metrics


def _jsonable(v):
    """GridSearch params can contain numpy scalars / None; make them JSON-safe."""
    try:
        return v.item()  # numpy scalar -> python scalar
    except AttributeError:
        return v


def _save_artifacts(model, metrics: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    config.MODEL_CARD_PATH.write_text(_render_model_card(metrics))


def _render_model_card(m: dict) -> str:
    t = m["test_set"]
    rows = "\n".join(
        f"| {name} | {info['cv_roc_auc']} |" for name, info in m["leaderboard"].items()
    )
    return f"""# Model Card — Diabetes Risk Screening

> **Not for clinical use.** This is an educational screening demo trained on a
> small, single-population research dataset. It must not be used to diagnose,
> treat, or make medical decisions about any individual.

- **Generated:** {m['generated_on']}
- **Selected model:** {m['best_model']}
- **Best hyperparameters:** `{m['best_params']}`

## Intended use
Educational demonstration of a responsible ML workflow for tabular clinical
data: a risk *screening* aid, not a diagnostic device.

## Training data
- Pima Indians Diabetes Dataset ({m['n_total_records']} records).
- Population: adult female patients of Pima Indian heritage. Results do **not**
  generalise to other populations, sexes, or age ranges.
- Features used: {", ".join(m['features'])}.
- Biologically impossible zero values (e.g. Glucose=0, BMI=0) are treated as
  missing and median-imputed inside the model pipeline.

## Model selection
Each candidate was tuned with 5-fold cross-validated GridSearchCV on ROC AUC;
the winner was chosen by cross-validated ROC AUC, then evaluated once on a
held-out 30% test set.

| Model | CV ROC AUC |
|---|---|
{rows}

## Held-out test performance ({t['n_samples']} samples)
| Metric | Value |
|---|---|
| ROC AUC | {t['roc_auc']} |
| Accuracy | {t['accuracy']} |
| Precision (diabetic) | {t['precision_diabetic']} |
| Recall (diabetic) | {t['recall_diabetic']} |
| F1 (diabetic) | {t['f1_diabetic']} |

Confusion matrix (rows = actual, cols = predicted): `{t['confusion_matrix']}`

## Limitations & ethical considerations
- Small dataset from a single, narrow population — high risk of bias if applied
  elsewhere.
- Recall on the diabetic class matters most clinically (a missed case is worse
  than a false alarm); the app exposes a decision-threshold slider for this.
- The data is de-identified research data and is used as such. No attempt is or
  should be made to re-identify individuals.
"""


if __name__ == "__main__":
    print("Training diabetes risk models...\n")
    train_and_select(verbose=True)
