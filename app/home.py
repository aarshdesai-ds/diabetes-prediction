"""Home page: project overview, responsible-use notice, and a data explorer."""
from __future__ import annotations

import streamlit as st

DISCLAIMER = (
    "⚠️ **Not a medical device.** This app is an educational demonstration of a "
    "machine-learning workflow. It is **not** a diagnostic tool and must not be "
    "used to make any medical decision. If you have health concerns, consult a "
    "qualified clinician."
)


def app(diabetes_df):
    st.title("🩺 Early Diabetes Risk — Screening Demo")
    st.warning(DISCLAIMER)

    st.markdown(
        "This project trains and compares several classifiers on the **Pima "
        "Indians Diabetes Dataset** and serves the best one through this app. "
        "It is built as a small but complete, *responsible* ML workflow: data "
        "validation, leak-free preprocessing pipelines, honest held-out "
        "evaluation, explainability, and CI/CD."
    )

    with st.expander("About the data & privacy", expanded=False):
        st.markdown(
            "- **Source:** Pima Indians Diabetes Dataset (UCI / Kaggle), "
            "originally collected by the US National Institute of Diabetes and "
            "Digestive and Kidney Diseases.\n"
            "- **Population:** adult female patients of Pima Indian heritage — "
            "results do **not** generalise to other groups.\n"
            "- **De-identified:** the data contains no identifiers. It is used "
            "strictly as de-identified research data; no re-identification is "
            "performed or supported.\n"
            "- Biologically impossible zero values (e.g. Glucose = 0) are "
            "treated as *missing* and imputed inside the model pipeline."
        )

    st.subheader("Explore the dataset")
    with st.expander("View raw data"):
        st.dataframe(diabetes_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.checkbox("Show column names"):
            st.write(list(diabetes_df.columns))
    with c2:
        if st.checkbox("Show data types"):
            st.write(diabetes_df.dtypes.astype(str))
    with c3:
        if st.checkbox("Show a single column"):
            col = st.selectbox("Column", tuple(diabetes_df.columns))
            st.write(diabetes_df[col])

    if st.checkbox("Show summary statistics"):
        st.dataframe(diabetes_df.describe(), use_container_width=True)
