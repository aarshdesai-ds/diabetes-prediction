"""Train, evaluate, calibrate, and persist the diabetes risk model.

Run with:  ``python -m src.train``

Pipeline of work:
  1. load + stratified train/test split,
  2. tune each candidate model with cross-validated GridSearchCV (ROC AUC),
  3. select the best model by cross-validated ROC AUC,
  4. estimate a confidence interval on its AUC via repeated stratified CV,
  5. probability-calibrate the winner (Platt scaling) and measure calibration,
  6. evaluate on the held-out test set (the honest numbers),
  7. refit the calibrated model on all data and persist artifacts.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from . import config, data
from .pipelines import build_pipelines, param_grids

CALIBRATION_METHOD = "sigmoid"  # Platt scaling — robust on small datasets
CALIBRATION_CV = 5


def train_and_select(verbose: bool = True):
    """Run the full training + selection routine. Returns (best_name, model, metrics)."""
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
    raw_best_params = {}
    for name, pipe in pipelines.items():
        search = GridSearchCV(pipe, grids[name], scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        raw_best_params[name] = search.best_params_
        leaderboard[name] = {
            "cv_roc_auc": round(float(search.best_score_), 4),
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        }
        if verbose:
            print(f"  {name:<20} CV ROC AUC = {search.best_score_:.4f}")

    best_name = max(leaderboard, key=lambda n: leaderboard[n]["cv_roc_auc"])

    # Unfitted pipeline with the winning hyperparameters, reused below.
    best_pipe = clone(pipelines[best_name]).set_params(**raw_best_params[best_name])

    # --- (4) Confidence interval on AUC via repeated stratified CV ----------
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(clone(best_pipe), X, y, scoring="roc_auc", cv=rcv, n_jobs=-1)
    ci_low, ci_high = np.percentile(cv_scores, [2.5, 97.5])
    repeated_cv = {
        "n_splits": 5, "n_repeats": 10,
        "mean_roc_auc": round(float(cv_scores.mean()), 4),
        "std_roc_auc": round(float(cv_scores.std()), 4),
        "ci95_low": round(float(ci_low), 4),
        "ci95_high": round(float(ci_high), 4),
    }
    if verbose:
        print(f"\nRepeated CV ROC AUC: {repeated_cv['mean_roc_auc']} "
              f"(95% CI {repeated_cv['ci95_low']}–{repeated_cv['ci95_high']})")

    # --- (5) Calibrate the winner, fit on train for honest evaluation ------
    calibrated = CalibratedClassifierCV(
        clone(best_pipe), method=CALIBRATION_METHOD, cv=CALIBRATION_CV
    )
    calibrated.fit(X_train, y_train)

    # Uncalibrated baseline (same pipeline) for a before/after Brier comparison.
    uncalibrated = clone(best_pipe).fit(X_train, y_train)
    brier_uncal = brier_score_loss(y_test, uncalibrated.predict_proba(X_test)[:, 1])

    # Choose the operating threshold on OUT-OF-FOLD training probabilities so the
    # test set stays untouched (no leakage). Pick the highest threshold whose
    # recall still meets the target sensitivity (maximises precision subject to
    # catching enough diabetic cases).
    oof_proba = cross_val_predict(
        CalibratedClassifierCV(clone(best_pipe), method=CALIBRATION_METHOD, cv=CALIBRATION_CV),
        X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    recommended_threshold = _choose_threshold(y_train.to_numpy(), oof_proba)
    if verbose:
        print(f"Recommended threshold ({config.THRESHOLD_STRATEGY} strategy): "
              f"{recommended_threshold:.3f}")

    # --- (6) Held-out test evaluation of the calibrated model --------------
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= recommended_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    brier_cal = brier_score_loss(y_test, y_proba)
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")

    # AUPRC (average precision) — more informative than ROC AUC under class
    # imbalance. The no-skill baseline is the positive prevalence.
    auprc = average_precision_score(y_test, y_proba)
    prevalence = float(y_test.mean())
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
    pr_curve = _downsample_curve(pr_recall, pr_precision, n=60)

    metrics = {
        "generated_on": str(date.today()),
        "best_model": best_name,
        "best_params": leaderboard[best_name]["best_params"],
        "calibration_method": CALIBRATION_METHOD,
        "recommended_threshold": round(float(recommended_threshold), 3),
        "threshold_strategy": config.THRESHOLD_STRATEGY,
        "target_recall": config.TARGET_RECALL,
        "target_precision": config.TARGET_PRECISION,
        "leaderboard": leaderboard,
        "repeated_cv": repeated_cv,
        "test_set": {
            "n_samples": int(len(y_test)),
            "threshold": round(float(recommended_threshold), 3),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "auprc": round(float(auprc), 4),
            "auprc_baseline": round(prevalence, 4),
            "pr_curve": pr_curve,
            "precision_diabetic": round(float(precision_score(y_test, y_pred)), 4),
            "recall_diabetic": round(float(recall_score(y_test, y_pred)), 4),
            "f1_diabetic": round(float(f1_score(y_test, y_pred)), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Non-diabetic", "Diabetic"], output_dict=True
            ),
        },
        "calibration": {
            "brier_uncalibrated": round(float(brier_uncal), 4),
            "brier_calibrated": round(float(brier_cal), 4),
            "reliability_curve": {
                "mean_predicted": [round(float(v), 4) for v in mean_pred],
                "fraction_positive": [round(float(v), 4) for v in frac_pos],
            },
        },
        "features": config.FEATURES,
        "n_total_records": int(len(df)),
    }

    # --- (7) Ship: refit a calibrated model on ALL data --------------------
    final_model = CalibratedClassifierCV(
        clone(best_pipe), method=CALIBRATION_METHOD, cv=CALIBRATION_CV
    )
    final_model.fit(X, y)

    _save_artifacts(final_model, metrics)
    if verbose:
        t = metrics["test_set"]
        print(f"\nSelected & calibrated: {best_name}")
        print(f"  Held-out test ROC AUC : {t['roc_auc']}")
        print(f"  Held-out test AUPRC   : {t['auprc']} (baseline {t['auprc_baseline']})")
        print(f"  Recall (diabetic)     : {t['recall_diabetic']}")
        print(f"  Precision (diabetic)  : {t['precision_diabetic']}")
        print(f"  Brier (uncal -> cal)  : {brier_uncal:.4f} -> {brier_cal:.4f}")
        print(f"  Artifacts written to  : {config.MODELS_DIR}")

    return best_name, final_model, metrics


def _choose_threshold(y_true, proba) -> float:
    """Pick an operating threshold per ``config.THRESHOLD_STRATEGY``.

    - "recall":    highest threshold whose recall >= TARGET_RECALL (most precise
                   point that still catches enough cases) — the screening default.
    - "precision": lowest threshold whose precision >= TARGET_PRECISION (most
                   sensitive point that still avoids too many false alarms).
    - "f1":        threshold that maximises F1 (no error treated as worse).

    Falls back to 0.5 if the target is unattainable.
    """
    strategy = config.THRESHOLD_STRATEGY
    candidates = np.unique(proba)
    positives = int((y_true == 1).sum())

    if strategy == "f1":
        best_t, best_f1 = config.DEFAULT_THRESHOLD, -1.0
        for t in candidates:
            pred = (proba >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return best_t

    best = config.DEFAULT_THRESHOLD
    found = False
    for t in candidates:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        predicted_pos = int((pred == 1).sum())
        recall = tp / positives if positives else 0.0
        precision = tp / predicted_pos if predicted_pos else 1.0
        if strategy == "precision":
            if precision >= config.TARGET_PRECISION and not found:
                best, found = float(t), True  # first (lowest) t meeting precision
        else:  # "recall"
            if recall >= config.TARGET_RECALL:
                best, found = float(t), True  # keep raising t while recall holds
    return best if found else config.DEFAULT_THRESHOLD


def _downsample_curve(x, y, n: int) -> dict:
    """Evenly sample up to ``n`` points from a curve, rounding for compact JSON."""
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return {
        "recall": [round(float(x[i]), 4) for i in idx],
        "precision": [round(float(y[i]), 4) for i in idx],
    }


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
    r = m["repeated_cv"]
    c = m["calibration"]
    rows = "\n".join(
        f"| {name} | {info['cv_roc_auc']} |" for name, info in m["leaderboard"].items()
    )
    return f"""# Model Card — Diabetes Risk Screening

> **Not for clinical use.** This is an educational screening demo trained on a
> small, single-population research dataset. It must not be used to diagnose,
> treat, or make medical decisions about any individual.

- **Generated:** {m['generated_on']}
- **Selected model:** {m['best_model']} (probability-calibrated via {m['calibration_method']} scaling)
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

## Generalisation estimate (repeated CV)
{r['n_splits']}-fold × {r['n_repeats']} repeats stratified CV ROC AUC:
**{r['mean_roc_auc']} ± {r['std_roc_auc']}** (95% CI {r['ci95_low']}–{r['ci95_high']}).

## Held-out test performance ({t['n_samples']} samples)
Evaluated at the recommended operating threshold **{m['recommended_threshold']}**
(chosen on out-of-fold data using the *{m['threshold_strategy']}* strategy).

| Metric | Value |
|---|---|
| ROC AUC | {t['roc_auc']} |
| AUPRC (avg precision) | {t['auprc']} (no-skill baseline {t['auprc_baseline']}) |
| Accuracy | {t['accuracy']} |
| Precision (diabetic) | {t['precision_diabetic']} |
| Recall (diabetic) | {t['recall_diabetic']} |
| F1 (diabetic) | {t['f1_diabetic']} |
| Brier score (lower = better) | {c['brier_calibrated']} (was {c['brier_uncalibrated']} uncalibrated) |

Confusion matrix (rows = actual, cols = predicted): `{t['confusion_matrix']}`

## Limitations & ethical considerations
- Small dataset from a single, narrow population — high risk of bias if applied
  elsewhere.
- Recall on the diabetic class matters most clinically (a missed case is worse
  than a false alarm); the app exposes a decision-threshold slider for this.
- Probabilities are calibrated so the threshold slider is meaningful, but they
  remain estimates from a small sample — treat them as screening signals only.
- The data is de-identified research data and is used as such. No attempt is or
  should be made to re-identify individuals.
"""


if __name__ == "__main__":
    print("Training diabetes risk models...\n")
    train_and_select(verbose=True)
