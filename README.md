# 🩺 Early Diabetes Risk — Screening Demo

[![CI](https://github.com/aarshdesai-ds/diabetes-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/aarshdesai-ds/diabetes-prediction/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

A small but **complete, responsible** machine-learning project: it trains and
compares several classifiers on the Pima Indians Diabetes Dataset, selects the
best one by honest held-out evaluation, and serves it through an interactive
Streamlit app — with tests, linting, CI/CD, and one-click deployment to three
platforms.

> ⚠️ **Not a medical device.** This is an educational demonstration only. It must
> not be used to diagnose, treat, or make any medical decision about any
> individual. If you have health concerns, consult a qualified clinician.

---

## Why this project is built the way it is

This started as a notebook-style Streamlit app and was re-engineered into the
shape a real ML product takes. The key design decisions:

| Decision | Why it matters in a healthcare-AI role |
|---|---|
| **Train once → save artifact → app does inference only** | The original retrained the model (incl. a 34-fit grid search) on *every button click*. Real products separate training from serving. |
| **Preprocessing lives inside the sklearn `Pipeline`** | Zero-as-missing cleaning, imputation, and scaling are applied identically at train and serve time — no train/serve skew, the #1 cause of silent ML bugs. |
| **Honest held-out test metrics + confidence interval** | The original reported *training* accuracy (overfit). We report ROC AUC, precision, **recall on the diabetic class**, and F1 on a stratified hold-out set, plus a 5×10 repeated-CV confidence interval on the AUC. |
| **Probability calibration** | The winning model is Platt-scaled (`CalibratedClassifierCV`) so its probabilities are trustworthy — reported via Brier score and a reliability curve. This makes the threshold slider meaningful. |
| **Tuned operating threshold (recall-first)** | In screening, a missed diabetic case is worse than a false alarm. The threshold is tuned on out-of-fold data to hit ~85% sensitivity (not a naive 0.5), and the app lets the user adjust it live. |
| **Explainability (SHAP)** | Per-patient SHAP explanations ("why was this person flagged") and a global feature-importance view — model-agnostic, computed over the calibrated model. |
| **Privacy & governance are first-class** | Data provenance, de-identification, intended use, and limitations are documented in a model card and surfaced in the app. |

---

## Data, privacy & responsible use

- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (UCI / Kaggle), originally collected by the US National Institute of Diabetes and Digestive and Kidney Diseases.
- **Population:** 768 adult female patients of Pima Indian heritage. **Results do not generalise** to other sexes, ages, or populations.
- **De-identified research data.** The dataset contains no identifiers and is used strictly as de-identified research data. **No re-identification is performed or supported** — re-identifying de-identified health data is a privacy violation and is out of scope for this project by design.
- **Data quality:** biologically impossible zero values (e.g. `Glucose = 0`, `BMI = 0`) are treated as *missing* and median-imputed inside the model pipeline rather than taken as real readings.

A full **model card** is generated at `models/model_card.md` every time you train.

---

## Architecture

```
diabetes-prediction/
├── streamlit_app.py          # App entry point (navigation + data loading)
├── app/                      # Streamlit UI pages
│   ├── home.py               #   overview, privacy notice, data explorer
│   ├── predict.py            #   sliders → probability → threshold-based label
│   └── plots.py              #   correlation, leaderboard, confusion matrix, importances
├── src/                      # Reusable, testable ML code (no Streamlit deps)
│   ├── config.py             #   paths, schema, feature lists, constants
│   ├── data.py               #   load + validate the raw CSV
│   ├── pipelines.py          #   model pipelines + hyperparameter grids
│   ├── train.py              #   train, select, calibrate, tune threshold, persist
│   ├── inference.py          #   load artifact + single-patient prediction
│   └── explain.py            #   SHAP local + global explanations
├── tests/                    # pytest suite (data, pipeline, inference)
├── models/                   # generated artifacts (gitignored)
├── .github/workflows/        # CI (lint+test+docker) and CD (HF Spaces)
├── Dockerfile                # container for Docker / Cloud Run / HF Spaces
├── diabetes.csv              # dataset
├── requirements.txt          # runtime deps
└── requirements-dev.txt      # + pytest, ruff
```

---

## Quickstart (local)

```bash
pip install -r requirements-dev.txt

# 1. Train the model (writes models/model.joblib, metrics.json, model_card.md)
python -m src.train

# 2. Run the tests + linter
python -m pytest -q
python -m ruff check .

# 3. Launch the app
python -m streamlit run streamlit_app.py   # → http://localhost:8501
```

> Tip: the `python -m` prefix works even if your `Scripts/` directory isn't on
> PATH (common on Windows). Plain `pytest` / `streamlit` work too once it is.

> If you skip step 1, the app trains the model automatically on first request.

---

## Model performance

The training script tunes four model families with 5-fold cross-validated
`GridSearchCV` (scored on ROC AUC), selects the best, calibrates it, tunes the
operating threshold on out-of-fold data, and evaluates once on a stratified 30%
hold-out set. On the current data the winner is **Random Forest** (calibrated),
evaluated at the tuned threshold of **0.264** (chosen for ~85% sensitivity):

| Metric (held-out test set) | Value |
|---|---|
| ROC AUC | 0.84 |
| ROC AUC, 5×10 repeated CV | 0.84 ± 0.03 (95% CI 0.78–0.89) |
| Recall / sensitivity (diabetic) | 0.84 |
| Precision (diabetic) | 0.60 |
| Accuracy | 0.75 |
| Brier score (calibrated) | 0.154 |

The recall/precision trade-off is deliberate: in screening you accept lower
precision to avoid missing diabetic patients. Exact, reproducible numbers are
written to `models/metrics.json` and `models/model_card.md` on every training
run. (The original project reported ~72% *training* accuracy, which overstated
real-world performance.)

---

## Testing & CI/CD

- **`tests/`** — schema/data validation, preprocessing correctness (zeros → imputed), every pipeline fits & emits valid probabilities, end-to-end inference, and threshold monotonicity.
- **`.github/workflows/ci.yml`** — on every push/PR: `ruff` lint → train → `pytest` → app-import smoke test → Docker build.
- **`.github/workflows/deploy-hf.yml`** — on push to `main`: deploys to Hugging Face Spaces (see below).

---

## Deployment

This app is deployable to three platforms, all from the same codebase.

### 1. Streamlit Community Cloud (easiest)
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch `main`, main file `streamlit_app.py`.
4. Deploy. It auto-redeploys on every push. No extra config needed.

### 2. Hugging Face Spaces (automated via CI)
1. Create a Space at [hf.co/new-space](https://huggingface.co/new-space) → **SDK: Docker**.
2. In this GitHub repo's **Settings → Secrets and variables → Actions**, add:
   - Secret `HF_TOKEN` — a write token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Variable `HF_USERNAME` — your HF username
   - Variable `HF_SPACE` — the Space name (e.g. `diabetes-prediction`)
3. Push to `main`. The `deploy-hf` workflow injects the Space metadata and pushes the repo; the Space builds the `Dockerfile`.

### 3. Docker (local, or Google Cloud Run)
```bash
# Build & run locally
docker build -t diabetes-prediction .
docker run -p 8501:8501 diabetes-prediction      # → http://localhost:8501

# Deploy to Google Cloud Run (scales to zero, public URL)
gcloud run deploy diabetes-prediction \
  --source . --region us-central1 --allow-unauthenticated
```
The image trains the model at build time and reads `$PORT` at runtime, so it
works unchanged on Cloud Run and HF Spaces.

---

## Implemented

- ✅ **SHAP** per-prediction explanations (why *this* patient was flagged) + global importance.
- ✅ **Calibration** (`CalibratedClassifierCV`, Platt scaling) so the probability is trustworthy.
- ✅ **Repeated CV** confidence interval on the AUC (5×10 stratified).
- ✅ **Threshold tuning** to a target sensitivity, chosen on out-of-fold data.

## Possible future extensions

- **Model monitoring** hooks (log inputs/predictions for drift detection).
- **XGBoost / LightGBM** as additional candidate models.
- **Fairness slicing** — report metrics across age bands to check for bias.

---

*Data: [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Educational use only — not for clinical decision-making.*
