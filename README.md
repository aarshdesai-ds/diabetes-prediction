# Diabetes Risk Prediction Web App

An interactive, multi-page Streamlit application for early diabetes risk prediction using clinical data, comparing a **Baseline Decision Tree** against a **GridSearchCV-Optimized Decision Tree** on the Pima Indians Diabetes Dataset.

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Dataset Description](#dataset-description)
- [App Architecture](#app-architecture)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

This project analyzes **768 patient records** from the Pima Indians Diabetes Dataset to build and compare two Decision Tree classifiers for early diabetes risk prediction:

1. **Baseline Decision Tree** — Fixed hyperparameters (entropy criterion, max depth 3)
2. **GridSearchCV-Optimized Tree** — Exhaustive search over 34 hyperparameter combinations scored by ROC AUC

The deployed Streamlit app lets users adjust 6 clinical input features via sliders, select a model, and receive an instant diabetes risk prediction alongside a performance score. A dedicated visualization page renders the trained tree structure and confusion matrix for model interpretability.

---

## Research Questions

- Can a shallow (depth-3) Decision Tree provide a useful clinical baseline for diabetes risk?
- How much does hyperparameter tuning via GridSearchCV improve predictive performance over the baseline?
- Which clinical features carry the most predictive signal, and which can be dropped without hurting accuracy?
- Does glucose remain the dominant predictor across both model configurations?

---

## Dataset Description

**Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — Kaggle / UCI Machine Learning Repository

| Property | Value |
|---|---|
| Total records | 768 |
| Features | 8 clinical measurements |
| Target variable | `Outcome` (0 = Non-diabetic, 1 = Diabetic) |
| Class 0 (Non-diabetic) | 500 records (65.1%) |
| Class 1 (Diabetic) | 268 records (34.9%) |
| Patient population | Adult female patients of Pima Indian heritage |

### Feature Inventory

| Feature | Type | Description |
|---|---|---|
| `Pregnancies` | Integer | Number of pregnancies |
| `Glucose` | Integer | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Integer | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Integer | Triceps skin fold thickness (mm) |
| `Insulin` | Integer | 2-hour serum insulin (μU/mL) |
| `BMI` | Float | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Float | Genetic diabetes likelihood score |
| `Age` | Integer | Patient age (years) |
| `Outcome` | Binary | 0 = Non-diabetic, 1 = Diabetic |

### Feature Selection

2 features are dropped before modelling based on low predictive contribution:

| Feature | Reason for Exclusion |
|---|---|
| `Pregnancies` | Weak correlation with outcome; adds noise at shallow tree depths |
| `SkinThickness` | High proportion of zero values; low signal relative to glucose and BMI |

**6 features retained:** Glucose, Blood Pressure, Insulin, BMI, Pedigree Function, Age

---

## App Architecture

The application is split across 4 Python files with a 3-page Streamlit navigation structure.

### Page 1 — Home (`diabetes_home.py`)

- Project overview and educational context about diabetes
- Interactive data explorer: full dataset table, column types, per-column distributions
- Summary statistics via `.describe()`

### Page 2 — Predict Diabetes (`diabetes_predict.py`)

**Baseline Decision Tree (`d_tree_pred`)**
- Criterion: `entropy`
- Max depth: `3`
- Cached with `@st.cache_data()` for fast repeated inference
- Reports training accuracy score

**GridSearchCV Optimized Tree (`grid_tree_pred`)**
- Parameter grid: `criterion` ∈ {`gini`, `entropy`} × `max_depth` ∈ {4, 5, ..., 20} = **34 combinations**
- Scoring: ROC AUC
- Parallelized with `n_jobs=-1`
- Reports best cross-validation ROC AUC score

**Train / Test Split (both models)**
- Test size: 30% → ~230 samples
- Train size: 70% → ~537 samples
- Random state: `1` (fixed for reproducibility)

**User Interface**
- 6 input sliders (Glucose, Blood Pressure, Insulin, BMI, Pedigree Function, Age)
- Model selector dropdown
- Predict button → binary output (diabetic / non-diabetic) + performance score

### Page 3 — Visualise Decision Tree (`diabetes_plots.py`)

- Correlation heatmap across all 8 features + target
- Confusion matrix for the selected model (training data)
- Interactive decision tree diagram rendered via Graphviz (max display depth: 3)

---

## Key Findings

### 1. GridSearchCV Delivers a Meaningful Accuracy Gain
- Baseline Decision Tree (depth 3, entropy): **~72% training accuracy**
- GridSearchCV Best Tree (depth 4–20 tuned): **~78% best cross-validation ROC AUC score**
- A 6-percentage-point improvement from tuning 34 hyperparameter combinations

### 2. Only 6 of 8 Features Are Needed
- Removing `Pregnancies` and `SkinThickness` does not degrade performance
- The 6-feature model matches or exceeds what the full 8-feature model produces at shallow depths
- This reduces overfitting risk and keeps the decision tree interpretable

### 3. Glucose is the Dominant Clinical Predictor
- Glucose appears at the root split of the decision tree in both model configurations
- Its range across the dataset (0–199 mg/dL) provides strong discriminative power
- BMI and Age serve as the most important secondary splits

### 4. Shallow Trees Are Surprisingly Competitive
- A depth-3 tree captures the majority of predictable variance in the dataset
- The class imbalance (65.1% vs 34.9%) means a naive majority classifier achieves ~65% accuracy — both models substantially exceed this baseline
- Deeper GridSearchCV trees gain primarily on the diabetic minority class (class 1 recall)

### 5. Model Performance Summary

| Model | Configuration | Score Type | Score |
|---|---|---|---|
| Decision Tree (baseline) | entropy, max_depth=3 | Training Accuracy | ~72% |
| GridSearchCV Best Tree | gini/entropy, depth 4–20 | Best CV ROC AUC | ~80% |

---

## Project Structure

```
diabetes-prediction/
│
├── diabetes_main.py        # Entry point: page config, data loading, navigation
├── diabetes_home.py        # Page 1: dataset explorer and project overview
├── diabetes_predict.py     # Page 2: model training, prediction, and UI sliders
├── diabetes_plots.py       # Page 3: correlation heatmap, confusion matrix, tree viz
│
├── diabetes.csv            # Pima Indians Diabetes Dataset (768 records, 9 columns)
├── requirements.txt        # Python dependency versions
└── README.md
```

---

## Requirements

```
streamlit>=1.31
pandas>=1.5
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.11
IPython>=8.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aarshdesai-ds/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app:
   ```bash
   streamlit run diabetes_main.py
   ```

4. Open the URL printed in the terminal (default: `http://localhost:8501`) and navigate between the three pages using the sidebar.

---

## Results Summary

| Model | Depth | Criterion | Score Metric | Score | Hyperparameter Combinations |
|---|---|---|---|---|---|
| Decision Tree (baseline) | 3 (fixed) | entropy | Training Accuracy | ~72% | 1 |
| GridSearchCV Best Tree | 4–20 (tuned) | gini / entropy | CV ROC AUC | ~80% | 34 |

**Dataset split:** 70% train (~537 samples) / 30% test (~230 samples), `random_state=1`

**Class distribution:** 500 non-diabetic (65.1%) / 268 diabetic (34.9%) across 768 total records

---

## Ideas for Extension

### 1. Add More Classifiers for Comparison
The current project uses only Decision Trees. Adding Random Forest, Logistic Regression, or XGBoost as selectable models in the dropdown would let users compare interpretability vs. raw performance directly within the same interface.

### 2. Address Class Imbalance
The 65/35 class split means the models may underperform on diabetic patients (the higher-risk class). Applying SMOTE oversampling or class-weight adjustments and re-evaluating precision/recall on class 1 specifically could meaningfully improve clinical utility.

### 3. Full Test-Set Evaluation
Both models currently report training accuracy (baseline) and cross-validation AUC (GridSearch) — neither reports held-out test accuracy. Adding a proper test-set evaluation with a full classification report (precision, recall, F1 per class) would give a more complete picture of generalization.

### 4. Feature Importance Visualization
Decision Trees expose feature importances natively via `feature_importances_`. Adding a bar chart of importances for both models to the visualization page would confirm whether Glucose and BMI dominate across both configurations and help users understand what drives each prediction.

### 5. Threshold Tuning for Clinical Use
The default classification threshold of 0.5 may not be appropriate in a medical context where false negatives (missing a diabetic patient) are more costly than false positives. Adding a threshold slider and a live precision-recall curve to the app would make it more clinically relevant.

### 6. Cross-Validation Stability Analysis
Currently GridSearchCV is run once. Running repeated k-fold cross-validation (e.g., 5×10-fold) and plotting score distributions across folds would reveal how stable the ~80% AUC estimate is and whether the optimized model actually generalizes reliably.

### 7. Hyperparameter Search Beyond Depth
The current grid only tunes `criterion` and `max_depth`. Expanding the search to include `min_samples_split`, `min_samples_leaf`, and `max_features` would give the optimizer more levers and could squeeze additional performance from the tree structure.

### 8. Explainability with SHAP
Adding SHAP (SHapley Additive exPlanations) values for individual predictions would let the app explain *why* a specific input combination is flagged as high risk — a critical feature for any clinical decision-support tool.

---

*Data sourced from [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Original data collected by the National Institute of Diabetes and Digestive and Kidney Diseases.*
