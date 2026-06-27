"""Central configuration: paths, feature definitions, and constants.

Keeping these in one place means the training script, the app, and the tests
all agree on the schema. If you change a feature name here, everything follows.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "diabetes.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_CARD_PATH = MODELS_DIR / "model_card.md"

# --- Schema ----------------------------------------------------------------
# The raw CSV column names, as shipped by the UCI/Kaggle dataset.
RAW_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

TARGET = "Outcome"

# Features we train on. We keep the same 6 features the original project used,
# but the choice is now documented and centralised.
FEATURES = [
    "Glucose",
    "BloodPressure",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Columns where a value of 0 is biologically impossible and therefore encodes a
# MISSING measurement rather than a real reading. These get imputed.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "Insulin", "BMI"]

# Human-friendly labels for the UI sliders.
FEATURE_LABELS = {
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "Insulin": "Insulin (µU/mL)",
    "BMI": "BMI (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
}

# --- Fairness ---------------------------------------------------------------
# Subgroups for fairness/bias auditing. The dataset is all adult female Pima
# patients, so age is the meaningful slicing axis. (label, low, high_inclusive)
AGE_BANDS = [
    ("21–29", 21, 29),
    ("30–39", 30, 39),
    ("40–49", 40, 49),
    ("50+", 50, 200),
]

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Default decision threshold. In a clinical screening context, missing a
# diabetic patient (false negative) is usually costlier than a false alarm,
# so the app lets the user lower this.
DEFAULT_THRESHOLD = 0.5

# How the operating threshold is chosen (on out-of-fold data, stored in
# metrics.json). For a screening tool we default to "recall" — missing a
# diabetic case is worse than a false alarm. Alternatives let you demonstrate
# a different cost model:
#   "recall"    -> highest threshold that still reaches TARGET_RECALL
#   "precision" -> lowest threshold that still reaches TARGET_PRECISION
#   "f1"        -> threshold that maximises F1 (no asymmetry assumed)
THRESHOLD_STRATEGY = "recall"
TARGET_RECALL = 0.85
TARGET_PRECISION = 0.60
