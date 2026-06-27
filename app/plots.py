"""Insights page: correlation, leaderboard, CV confidence interval,
confusion matrix, calibration curve, and global SHAP importance."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from src import config
from src.explain import global_importance
from src.inference import load_metrics


def app(diabetes_df):
    st.title("📊 Model Insights")
    metrics = load_metrics()

    # --- Correlation heatmap ----------------------------------------------
    if st.checkbox("Correlation heatmap", value=True):
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(diabetes_df.corr(numeric_only=True), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax)
        ax.set_title("Feature correlations")
        st.pyplot(fig)

    # --- Model leaderboard + CV confidence interval ------------------------
    if st.checkbox("Model leaderboard & confidence interval", value=True):
        lb = pd.DataFrame(
            {"CV ROC AUC": {k: v["cv_roc_auc"] for k, v in metrics["leaderboard"].items()}}
        ).sort_values("CV ROC AUC", ascending=False)
        st.dataframe(lb, use_container_width=True)
        r = metrics.get("repeated_cv")
        if r:
            st.info(
                f"**{metrics['best_model']}** generalisation estimate "
                f"({r['n_splits']}×{r['n_repeats']} repeated CV): "
                f"ROC AUC **{r['mean_roc_auc']} ± {r['std_roc_auc']}** "
                f"(95% CI {r['ci95_low']}–{r['ci95_high']})."
            )

    # --- Held-out confusion matrix (from saved metrics, no retraining) -----
    if st.checkbox("Confusion matrix (held-out test set)"):
        cm = np.array(metrics["test_set"]["confusion_matrix"])
        thr = metrics["test_set"].get("threshold", config.DEFAULT_THRESHOLD)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Non-diabetic", "Diabetic"]
        ).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Test-set confusion matrix (threshold {thr})")
        st.pyplot(fig)

    # --- Calibration reliability curve ------------------------------------
    if st.checkbox("Calibration curve"):
        cal = metrics.get("calibration", {})
        rc = cal.get("reliability_curve", {})
        if rc:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
            ax.plot(rc["mean_predicted"], rc["fraction_positive"], "o-",
                    color="#2c7fb8", label="Model")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed fraction positive")
            ax.set_title("Reliability curve")
            ax.legend()
            st.pyplot(fig)
            st.caption(
                f"Brier score: **{cal['brier_calibrated']}** calibrated "
                f"(was {cal['brier_uncalibrated']} before calibration; lower is better)."
            )

    # --- Test metrics ------------------------------------------------------
    if st.checkbox("Test-set metrics"):
        t = metrics["test_set"]
        st.json({
            "Operating threshold": t.get("threshold"),
            "ROC AUC": t["roc_auc"],
            "Accuracy": t["accuracy"],
            "Precision (diabetic)": t["precision_diabetic"],
            "Recall (diabetic)": t["recall_diabetic"],
            "F1 (diabetic)": t["f1_diabetic"],
        })

    # --- Global SHAP importance -------------------------------------------
    if st.checkbox("Global feature importance (SHAP)", value=True):
        with st.spinner("Computing SHAP values across a sample…"):
            imp = pd.Series(global_importance()).sort_values()
        imp.index = [config.FEATURE_LABELS.get(f, f) for f in imp.index]
        fig, ax = plt.subplots(figsize=(8, 4))
        imp.plot.barh(ax=ax, color="#2c7fb8")
        ax.set_title("Mean |SHAP| — average impact on the prediction")
        ax.set_xlabel("Mean absolute SHAP value")
        st.pyplot(fig)
        st.caption(
            "Model-agnostic SHAP attribution over the calibrated model. Larger "
            "bars mean the feature moves predictions more, on average."
        )
