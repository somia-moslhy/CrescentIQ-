import streamlit as st
import pandas as pd
from utils import inject_theme, load_pipeline, load_stores, hijri_features_for_date

st.set_page_config(page_title="CrescentIQ — Predict", layout="wide", page_icon="🔮")
inject_theme()
st.title("🔮 Predict Weekly Sales")

pipeline, feature_cols = load_pipeline()
stores = load_stores()

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.selectbox("Store", options=sorted(stores["Store"].unique()))
        store_type = stores.loc[stores["Store"] == store, "Type"].iloc[0]
        st.caption(f"Store Type: {store_type}")
        dept = st.number_input("Department", min_value=1, max_value=99, value=1)
    with col2:
        date = st.date_input("Date")
        temperature = st.number_input("Temperature (°F)", value=60.0)
        fuel_price = st.number_input("Fuel Price", value=3.5)
    with col3:
        cpi = st.number_input("CPI", value=180.0)
        unemployment = st.number_input("Unemployment (%)", value=7.5)
        is_holiday = st.checkbox("Is US federal holiday week?")

    predict_clicked = st.button("Predict Weekly Sales", type="primary")

if predict_clicked:
    hijri_feats = hijri_features_for_date(pd.Timestamp(date))
    row = {
        "Store": store, "Dept": dept, "Type": store_type,
        "Temperature": temperature, "Fuel_Price": fuel_price,
        "CPI": cpi, "Unemployment": unemployment, "IsHoliday": int(is_holiday),
        **hijri_feats,
    }
    X = pd.DataFrame([row])[feature_cols]
    prediction = pipeline.predict(X)[0]

    st.success(f"Predicted Weekly Sales: **${prediction:,.0f}**")
    if hijri_feats["is_ramadan_window"] or hijri_feats["is_pre_eid_window"]:
        st.info("This date falls inside a Ramadan/Eid demand window — the model has factored this in.")