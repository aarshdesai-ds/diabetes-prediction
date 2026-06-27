"""Visualisation page: correlation, model leaderboard, confusion matrix, importances."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from src import config
from src.inference import load_metrics, load_model


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

    # --- Model leaderboard -------------------------------------------------
    if st.checkbox("Model leaderboard (cross-validated)", value=True):
        lb = pd.DataFrame(
            {"CV ROC AUC": {k: v["cv_roc_auc"] for k, v in metrics["leaderboard"].items()}}
        ).sort_values("CV ROC AUC", ascending=False)
        st.dataframe(lb, use_container_width=True)
        st.caption(f"Deployed model: **{metrics['best_model']}**")

    # --- Held-out confusion matrix (from saved metrics, no retraining) -----
    if st.checkbox("Confusion matrix (held-out test set)"):
        cm = np.array(metrics["test_set"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Non-diabetic", "Diabetic"]
        ).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("Test-set confusion matrix")
        st.pyplot(fig)

    # --- Test metrics ------------------------------------------------------
    if st.checkbox("Test-set metrics"):
        t = metrics["test_set"]
        st.json({
            "ROC AUC": t["roc_auc"],
            "Accuracy": t["accuracy"],
            "Precision (diabetic)": t["precision_diabetic"],
            "Recall (diabetic)": t["recall_diabetic"],
            "F1 (diabetic)": t["f1_diabetic"],
        })

    # --- Feature importance / coefficients --------------------------------
    if st.checkbox("Feature importance", value=True):
        importances = _feature_importance(load_model())
        if importances is None:
            st.info("This model does not expose feature importances.")
        else:
            imp = pd.Series(importances, index=config.FEATURES).sort_values()
            fig, ax = plt.subplots(figsize=(8, 4))
            imp.plot.barh(ax=ax, color="#2c7fb8")
            ax.set_title("What drives the prediction")
            ax.set_xlabel("Relative importance")
            st.pyplot(fig)


def _feature_importance(model):
    """Pull importances from the classifier step of the fitted pipeline."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    return None
