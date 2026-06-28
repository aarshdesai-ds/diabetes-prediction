"""Prediction page: sliders -> calibrated probability -> threshold-based label,
with a per-patient SHAP explanation of what drove the result."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src import config
from src.explain import explain_one
from src.inference import load_metrics, predict_one


def app(diabetes_df):
    st.title("🔮 Predict Diabetes Risk")
    metrics = load_metrics()
    rec_threshold = metrics.get("recommended_threshold", config.DEFAULT_THRESHOLD)
    st.caption(
        f"Serving the calibrated **{metrics['best_model']}** model "
        f"(test ROC AUC = {metrics['test_set']['roc_auc']}, "
        f"diabetic recall = {metrics['test_set']['recall_diabetic']} at the "
        f"recommended threshold)."
    )

    st.subheader("Patient measurements")
    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(config.FEATURES):
        col = cols[i % 2]
        series = diabetes_df[feat]
        lo, hi = float(series.min()), float(series.max())
        default = float(series.median())
        if feat in ("BMI", "DiabetesPedigreeFunction"):
            inputs[feat] = col.slider(config.FEATURE_LABELS[feat], lo, hi, default)
        else:
            inputs[feat] = col.slider(
                config.FEATURE_LABELS[feat], int(lo), int(hi), int(default)
            )

    st.subheader("Decision threshold")
    threshold = st.slider(
        "Flag as 'at risk' when the calibrated probability is at least…",
        min_value=0.05, max_value=0.95, value=float(rec_threshold), step=0.01,
        help=f"Default is the tuned threshold ({rec_threshold:.2f}) that reaches "
             f"~{int(metrics.get('target_recall', 0.85) * 100)}% sensitivity. "
             f"Lower it to catch more diabetic cases at the cost of more false alarms.",
    )

    if st.button("Predict", type="primary"):
        label, proba = predict_one(inputs, threshold=threshold)
        st.metric("Estimated probability of diabetes", f"{proba:.1%}")
        if label == 1:
            st.error(
                f"**Elevated risk** — probability {proba:.1%} is at or above the "
                f"{threshold:.0%} threshold. A screening signal, not a diagnosis. "
                f"Recommend follow-up with a clinician."
            )
        else:
            st.success(
                f"**Lower risk** — probability {proba:.1%} is below the "
                f"{threshold:.0%} threshold. This does not rule out diabetes."
            )

        with st.expander("Why this prediction? (SHAP explanation)", expanded=True):
            try:
                with st.spinner("Computing feature contributions…"):
                    _render_explanation(inputs)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully in the UI
                st.info(f"Explanation unavailable for this input ({exc}).")

        st.caption(
            "Reminder: educational demo only — not a medical device. See the "
            "Home page for data and model limitations."
        )


def _render_explanation(inputs: dict):
    exp = explain_one(inputs)
    contribs = exp["contributions"]
    labels = [config.FEATURE_LABELS[f] for f in config.FEATURES]
    values = [contribs[f] for f in config.FEATURES]
    colors = ["#d7191c" if v > 0 else "#2c7fb8" for v in values]

    order = sorted(range(len(values)), key=lambda i: values[i])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([labels[i] for i in order], [values[i] for i in order],
            color=[colors[i] for i in order])
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Contribution to diabetes probability")
    ax.set_title(
        f"Baseline {exp['base_value']:.0%}  →  this patient {exp['prediction']:.0%}"
    )
    st.pyplot(fig)
    st.caption(
        "Red bars push risk **up**, blue bars push it **down**, relative to the "
        "average patient. Bars sum to the gap between the baseline and this "
        "patient's probability."
    )
