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

    # --- Precision-Recall curve (AUPRC) -----------------------------------
    if st.checkbox("Precision-Recall curve (AUPRC)"):
        t = metrics["test_set"]
        pr = t.get("pr_curve")
        if pr:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(pr["recall"], pr["precision"], "-", color="#2c7fb8",
                    label=f"Model (AUPRC = {t['auprc']})")
            ax.axhline(t["auprc_baseline"], ls="--", color="grey",
                       label=f"No-skill baseline ({t['auprc_baseline']})")
            ax.set_xlabel("Recall (sensitivity)")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall curve")
            ax.set_ylim(0, 1.02)
            ax.legend()
            st.pyplot(fig)
            st.caption(
                "AUPRC focuses on the diabetic (positive) class and is more "
                "informative than ROC AUC under class imbalance. The no-skill "
                "baseline equals the diabetic prevalence in the test set."
            )

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
            "AUPRC (baseline)": f"{t['auprc']} ({t['auprc_baseline']})",
            "Accuracy": t["accuracy"],
            "Precision (diabetic)": t["precision_diabetic"],
            "Recall (diabetic)": t["recall_diabetic"],
            "F1 (diabetic)": t["f1_diabetic"],
        })

    # --- Fairness: performance by age band --------------------------------
    if st.checkbox("Fairness — performance by age band", value=True):
        fair = metrics.get("fairness", {}).get("by_age", [])
        if fair:
            df_fair = pd.DataFrame(fair).set_index("band")
            st.dataframe(df_fair, use_container_width=True)

            recalls = df_fair["recall"].dropna()
            fig, ax = plt.subplots(figsize=(8, 4))
            recalls.plot.bar(ax=ax, color="#2c7fb8")
            overall = metrics["test_set"]["recall_diabetic"]
            ax.axhline(overall, ls="--", color="#d7191c",
                       label=f"Overall recall ({overall})")
            ax.set_ylabel("Recall (sensitivity)")
            ax.set_ylim(0, 1.05)
            ax.set_title("Diabetic-case recall by age band")
            ax.legend()
            st.pyplot(fig)

            if len(recalls) > 1:
                gap = round(float(recalls.max() - recalls.min()), 3)
                msg = (f"Recall ranges {recalls.min():.2f}–{recalls.max():.2f} "
                       f"across age bands (gap {gap}).")
                (st.warning if gap >= 0.15 else st.info)(
                    msg + " Large gaps mean the model detects diabetes unevenly "
                    "across ages — a bias/deployment risk. Small bands also give "
                    "noisy estimates."
                )

    # --- Global feature importance (permutation) --------------------------
    if st.checkbox("Global feature importance", value=True):
        try:
            with st.spinner("Computing permutation importance…"):
                imp = pd.Series(global_importance()).sort_values()
            imp.index = [config.FEATURE_LABELS.get(f, f) for f in imp.index]
            fig, ax = plt.subplots(figsize=(8, 4))
            imp.plot.barh(ax=ax, color="#2c7fb8")
            ax.set_title("Permutation importance — drop in ROC AUC when shuffled")
            ax.set_xlabel("Mean ROC AUC decrease")
            ax.axvline(0, color="#444", linewidth=0.8)
            st.pyplot(fig)
            st.caption(
                "Model-agnostic permutation importance over the calibrated model: "
                "how much test ROC AUC drops when each feature is randomly shuffled. "
                "Larger = more important. (Per-patient SHAP is on the Predict page.)"
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully in the UI
            st.info(f"Feature importance unavailable ({exc}).")
