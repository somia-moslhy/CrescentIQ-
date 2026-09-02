import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from hijri_converter import Hijri, Gregorian

st.set_page_config(page_title="CrescentIQ", layout="wide")


@st.cache_data
def load_hijri_data():
    return pd.read_csv("data/processed/walmart_with_hijri_features.csv", parse_dates=["Date"])


@st.cache_data
def load_stores():
    return pd.read_csv("data/stores.csv")


@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("models/crescentiq_pipeline.pkl")
    feature_cols = joblib.load("models/crescentiq_feature_columns.pkl")
    return pipeline, feature_cols


def hijri_features_for_date(date: pd.Timestamp) -> dict:
    """Compute the same days_to/since Hijri features used in training,
    for a single arbitrary date — used at prediction time."""
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


page = st.sidebar.radio("Navigation", ["Analysis", "Predict"])

# ============================================================
if page == "Analysis":
    st.title("CrescentIQ — Sales Analysis by Market")
    df = load_hijri_data()
    stores = load_stores()

    # "Market" = Store, the closest unit to an independent market in this dataset
    market_sales = df.groupby("Store")["Weekly_Sales"].sum().reset_index().sort_values("Weekly_Sales", ascending=False)
    st.plotly_chart(
        px.bar(market_sales, x="Store", y="Weekly_Sales", title="Total sales by market (Store)"),
        use_container_width=True,
    )

    st.subheader("Does the season (Ramadan/Eid) affect sales? — per market")
    df["season"] = df["is_ramadan_window"] | df["is_pre_eid_window"]
    df["season"] = df["season"].map({True: "Ramadan/Eid window", False: "Regular period"})

    selected_markets = st.multiselect(
        "Select markets to compare", options=sorted(df["Store"].unique()),
        default=sorted(df["Store"].unique())[:5],
    )
    filtered = df[df["Store"].isin(selected_markets)]
    season_avg = filtered.groupby(["Store", "season"])["Weekly_Sales"].mean().reset_index()
    st.plotly_chart(
        px.bar(season_avg, x="Store", y="Weekly_Sales", color="season", barmode="group",
               title="Avg Weekly_Sales: Ramadan/Eid window vs regular period, by market"),
        use_container_width=True,
    )

    st.subheader("Sales over time for a single market")
    single_market = st.selectbox("Market (Store)", options=sorted(df["Store"].unique()))
    trend = df[df["Store"] == single_market].sort_values("Date")
    fig_trend = px.line(trend, x="Date", y="Weekly_Sales", title=f"Weekly sales over time — Store {single_market}")
    ramadan_dates = trend.loc[trend["is_ramadan_window"], "Date"]
    for d in ramadan_dates:
        fig_trend.add_vline(x=d, line_color="green", opacity=0.15)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Shaded green lines mark weeks falling inside a Ramadan window.")

    # Overall verdict, not just per-market
    overall_avg = df.groupby("season")["Weekly_Sales"].mean()
    diff_pct = (overall_avg["Ramadan/Eid window"] - overall_avg["Regular period"]) / overall_avg["Regular period"] * 100
    st.metric("Overall: Ramadan/Eid vs regular period sales", f"{diff_pct:+.1f}%")

# ============================================================
elif page == "Predict":
    st.title("CrescentIQ — Predict Weekly Sales")
    pipeline, feature_cols = load_pipeline()
    stores = load_stores()

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

    if st.button("Predict Weekly Sales", type="primary"):
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