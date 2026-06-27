"""Streamlit entry point.

Run locally with:  ``streamlit run streamlit_app.py``

This is the file Streamlit Community Cloud and Hugging Face Spaces look for.
It wires up navigation and loads the dataset once (cached).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import home, plots, predict
from src import config

st.set_page_config(
    page_title="Early Diabetes Risk — Screening Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH)


def main() -> None:
    df = load_data()
    pages = {
        "Home": home,
        "Predict Diabetes Risk": predict,
        "Model Insights": plots,
    }
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", tuple(pages.keys()))
    st.sidebar.caption("Educational demo — not for clinical use.")
    pages[choice].app(df)


if __name__ == "__main__":
    main()
