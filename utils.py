"""Shared CSS, cached loaders, and helpers used across all pages."""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from hijri_converter import Hijri, Gregorian

# Absolute paths anchored to this file's location — read_csv/joblib.load with
# a relative path resolve against the working directory `streamlit run` was
# launched FROM, not against this file's folder, which breaks the moment
# someone launches it from a different directory (VS Code, a different
# terminal, deployment, etc.). Anchoring to __file__ makes it launch-location
# independent.
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

TEAL = "#0E3D3D"
MINT = "#3ECFB2"
MINT_BG = "#F0FAF8"


def inject_theme():
    st.markdown(f"""
    <style>
        .stApp {{ background-color: #FFFFFF; }}
        h1, h2, h3 {{ color: {TEAL}; font-weight: 700; }}

        div[data-testid="stMetric"] {{
            background-color: {MINT_BG};
            border: 1px solid #CDEDE6;
            border-radius: 14px;
            padding: 16px 20px;
        }}
        div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {{ color: {TEAL}; }}

        .stButton>button {{
            background-color: {TEAL};
            color: #FFFFFF;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
        }}
        .stButton>button:hover {{ background-color: #145C5C; color: #FFFFFF; }}

        div[data-testid="stAlertContainer"] {{ border-radius: 12px; }}
        section[data-testid="stSidebar"] {{ background-color: {MINT_BG}; }}

        /* Style the auto-generated multipage nav links as rounded pills */
        [data-testid="stSidebarNav"] a {{
            border-radius: 10px;
            margin: 2px 8px;
            padding: 8px 12px;
        }}
        [data-testid="stSidebarNav"] a:hover {{ background-color: #DFF5F0; }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_hijri_data():
    return pd.read_csv(DATA_PROCESSED / "walmart_with_hijri_features.csv", parse_dates=["Date"])


@st.cache_data
def load_stores():
    stores_path = DATA_RAW / "stores.csv"
    if not stores_path.exists():
        stores_path = BASE_DIR / "data" / "stores.csv"
    return pd.read_csv(stores_path)


@st.cache_resource
def load_pipeline():
    pipeline = joblib.load(MODELS_DIR / "crescentiq_pipeline.pkl")
    feature_cols = joblib.load(MODELS_DIR / "crescentiq_feature_columns.pkl")
    return pipeline, feature_cols


@st.cache_data
def load_business_impact():
    return pd.read_csv(RESULTS_DIR / "business_impact.csv").iloc[0]


@st.cache_data
def load_model_comparison():
    return pd.read_csv(RESULTS_DIR / "model_comparison.csv", index_col=0)


@st.cache_data
def load_hijri_comparison():
    return pd.read_csv(RESULTS_DIR / "baseline_vs_hijri_comparison.csv", index_col=0)


@st.cache_data
def load_dept_breakdown():
    return pd.read_csv(RESULTS_DIR / "improvement_by_dept.csv")


def hijri_features_for_date(date: pd.Timestamp) -> dict:
    """Compute days_to/since Hijri features for a single arbitrary date —
    used at prediction time, same logic as training."""
    g = Gregorian(date.year, date.month, date.day).to_hijri()
    candidates = []
    for hy in [g.year - 1, g.year, g.year + 1]:
        candidates.append(("ramadan_start", pd.Timestamp(*Hijri(hy, 9, 1).to_gregorian().datetuple())))
        candidates.append(("eid_fitr", pd.Timestamp(*Hijri(hy, 10, 1).to_gregorian().datetuple())))
        candidates.append(("eid_adha", pd.Timestamp(*Hijri(hy, 12, 10).to_gregorian().datetuple())))
    events = pd.DataFrame(candidates, columns=["event", "date"])

    feats = {}
    for name, prefix in [("ramadan_start", "ramadan"), ("eid_fitr", "eid_fitr"), ("eid_adha", "eid_adha")]:
        ev_dates = events.loc[events["event"] == name, "date"]
        after = ev_dates[ev_dates >= date]
        before = ev_dates[ev_dates <= date]
        feats[f"days_to_{prefix}"] = (after.min() - date).days if not after.empty else 999
        feats[f"days_since_{prefix}"] = (date - before.max()).days if not before.empty else 999

    feats["is_ramadan_window"] = (0 <= feats["days_since_ramadan"] <= 29) or (0 <= feats["days_to_ramadan"] <= 6)
    feats["is_pre_eid_window"] = (0 <= feats["days_to_eid_fitr"] <= 6) or (0 <= feats["days_to_eid_adha"] <= 6)
    return feats