"""Prediction page: sliders -> probability -> threshold-based label."""
from __future__ import annotations

import streamlit as st

from src import config
from src.inference import load_metrics, predict_one


def app(diabetes_df):
    st.title("🔮 Predict Diabetes Risk")
    metrics = load_metrics()
    st.caption(
        f"Serving the **{metrics['best_model']}** model "
        f"(held-out test ROC AUC = {metrics['test_set']['roc_auc']}, "
        f"diabetic recall = {metrics['test_set']['recall_diabetic']})."
    )

    st.subheader("Patient measurements")
    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(config.FEATURES):
        col = cols[i % 2]
        series = diabetes_df[feat]
        lo, hi = float(series.min()), float(series.max())
        # Default to the dataset median so the starting point is realistic.
        default = float(series.median())
        if feat in ("BMI", "DiabetesPedigreeFunction"):
            inputs[feat] = col.slider(config.FEATURE_LABELS[feat], lo, hi, default)
        else:
            inputs[feat] = col.slider(
                config.FEATURE_LABELS[feat], int(lo), int(hi), int(default)
            )

    st.subheader("Decision threshold")
    threshold = st.slider(
        "Flag as 'at risk' when the model's probability is at least…",
        min_value=0.05, max_value=0.95, value=config.DEFAULT_THRESHOLD, step=0.05,
        help="Lower the threshold to catch more diabetic cases (higher recall) "
             "at the cost of more false alarms. In screening, missing a case is "
             "usually worse than a false alarm.",
    )

    if st.button("Predict", type="primary"):
        label, proba = predict_one(inputs, threshold=threshold)
        st.metric("Estimated probability of diabetes", f"{proba:.1%}")
        if label == 1:
            st.error(
                f"**Elevated risk** — probability {proba:.1%} is at or above the "
                f"{threshold:.0%} threshold. This is a screening signal, not a "
                f"diagnosis. Recommend follow-up with a clinician."
            )
        else:
            st.success(
                f"**Lower risk** — probability {proba:.1%} is below the "
                f"{threshold:.0%} threshold. This does not rule out diabetes."
            )
        st.caption(
            "Reminder: educational demo only — not a medical device. See the "
            "Home page for data and model limitations."
        )
