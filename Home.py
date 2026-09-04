import streamlit as st
from utils import inject_theme

st.set_page_config(page_title="CrescentIQ", layout="wide", page_icon="🌙")
inject_theme()

st.title("🌙 CrescentIQ")
st.subheader("Hijri-aware demand forecasting for MENA retail & manufacturing")

st.markdown("""
Standard demand forecasting tools assume Gregorian-calendar seasonality — but
Ramadan and the two Eids follow the lunar Hijri calendar, shifting ~10-11 days
earlier every Gregorian year. That breaks the year-over-year seasonality
patterns that enterprise tools rely on.

**CrescentIQ** adds a Hijri-calendar feature layer (built on the official
Umm al-Qura calendar) on top of a classical ML pipeline, and measures its
real, quantified contribution to forecast accuracy.

**Use the sidebar to navigate:**
- **Analysis** — sales by market, and how the Ramadan/Eid season affects them
- **Predict** — get a live sales prediction from the trained model
- **Business Impact** — what the Hijri feature engineering is worth, in dollars
""")