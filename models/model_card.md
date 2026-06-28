# Model Card — Diabetes Risk Screening

> **Not for clinical use.** This is an educational screening demo trained on a
> small, single-population research dataset. It must not be used to diagnose,
> treat, or make medical decisions about any individual.

- **Generated:** 2026-06-28
- **Selected model:** Random Forest (probability-calibrated via sigmoid scaling)
- **Best hyperparameters:** `{'clf__max_depth': 6, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 5}`

## Intended use
Educational demonstration of a responsible ML workflow for tabular clinical
data: a risk *screening* aid, not a diagnostic device.

## Training data
- Pima Indians Diabetes Dataset (768 records).
- Population: adult female patients of Pima Indian heritage. Results do **not**
  generalise to other populations, sexes, or age ranges.
- Features used: Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age.
- Biologically impossible zero values (e.g. Glucose=0, BMI=0) are treated as
  missing and median-imputed inside the model pipeline.

## Model selection
Each candidate was tuned with 5-fold cross-validated GridSearchCV on ROC AUC;
the winner was chosen by cross-validated ROC AUC, then evaluated once on a
held-out 30% test set.

| Model | CV ROC AUC |
|---|---|
| Logistic Regression | 0.8318 |
| Decision Tree | 0.8153 |
| Random Forest | 0.8417 |
| Gradient Boosting | 0.8341 |

## Generalisation estimate (repeated CV)
5-fold × 10 repeats stratified CV ROC AUC:
**0.8413 ± 0.0281** (95% CI 0.7793–0.8868).

## Held-out test performance (231 samples)
Evaluated at the recommended operating threshold **0.264**
(chosen on out-of-fold data using the *recall* strategy).

| Metric | Value |
|---|---|
| ROC AUC | 0.8414 |
| AUPRC (avg precision) | 0.7393 (no-skill baseline 0.3506) |
| Accuracy | 0.7489 |
| Precision (diabetic) | 0.6018 |
| Recall (diabetic) | 0.8395 |
| F1 (diabetic) | 0.701 |
| Brier score (lower = better) | 0.154 (was 0.1563 uncalibrated) |

Confusion matrix (rows = actual, cols = predicted): `[[105, 45], [13, 68]]`

## Fairness — performance by age band (held-out test set)
Bias audit across age subgroups. Small bands give unstable estimates, which is
itself a caveat worth surfacing.

| Age band | n | Prevalence | Recall | Precision | ROC AUC |
|---|---|---|---|---|---|
| 21–29 | 129 | 0.1938 | 0.64 | 0.4444 | 0.8015 |
| 30–39 | 47 | 0.5957 | 0.8929 | 0.7576 | 0.8233 |
| 40–49 | 29 | 0.5517 | 1.0 | 0.6667 | 0.8558 |
| 50+ | 26 | 0.4615 | 0.9167 | 0.55 | 0.7976 |

**Recall disparity across age bands: 0.36.** A large gap means the
model catches diabetic cases unevenly across ages — a deployment risk. Mitigations
to consider: per-group thresholds, collecting more data for under-served bands, or
reporting subgroup metrics to clinicians alongside each prediction.

## Limitations & ethical considerations
- Small dataset from a single, narrow population — high risk of bias if applied
  elsewhere.
- Recall on the diabetic class matters most clinically (a missed case is worse
  than a false alarm); the app exposes a decision-threshold slider for this.
- Probabilities are calibrated so the threshold slider is meaningful, but they
  remain estimates from a small sample — treat them as screening signals only.
- The data is de-identified research data and is used as such. No attempt is or
  should be made to re-identify individuals.
