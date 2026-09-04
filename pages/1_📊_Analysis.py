import streamlit as st
import plotly.express as px
from utils import inject_theme, load_hijri_data, TEAL, MINT

st.set_page_config(page_title="CrescentIQ — Analysis", layout="wide", page_icon="📊")
inject_theme()
st.title("📊 Sales Analysis by Market")

df = load_hijri_data()

market_sales = df.groupby("Store")["Weekly_Sales"].sum().reset_index().sort_values("Weekly_Sales", ascending=False)
st.plotly_chart(
    px.bar(market_sales, x="Store", y="Weekly_Sales", title="Total sales by market (Store)",
           color_discrete_sequence=[TEAL]),
    use_container_width=True,
)

st.markdown("#### Does the season (Ramadan/Eid) affect sales? — per market")
df["season"] = (df["is_ramadan_window"] | df["is_pre_eid_window"]).map(
    {True: "Ramadan/Eid window", False: "Regular period"})

selected_markets = st.multiselect(
    "Select markets to compare", options=sorted(df["Store"].unique()),
    default=sorted(df["Store"].unique())[:5],
)
filtered = df[df["Store"].isin(selected_markets)]
season_avg = filtered.groupby(["Store", "season"])["Weekly_Sales"].mean().reset_index()
st.plotly_chart(
    px.bar(season_avg, x="Store", y="Weekly_Sales", color="season", barmode="group",
           title="Avg Weekly_Sales: Ramadan/Eid window vs regular period, by market",
           color_discrete_sequence=[TEAL, MINT]),
    use_container_width=True,
)

st.markdown("#### Do MarkDown promotions move sales?")
markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
if markdown_cols:
    df["has_markdown"] = df[markdown_cols].sum(axis=1) > 0
    markdown_avg = df.groupby("has_markdown")["Weekly_Sales"].mean().reset_index()
    markdown_avg["has_markdown"] = markdown_avg["has_markdown"].map({True: "Promo week", False: "No promo"})
    st.plotly_chart(
        px.bar(markdown_avg, x="has_markdown", y="Weekly_Sales",
               title="Avg Weekly_Sales: promo (MarkDown active) vs no promo",
               color_discrete_sequence=[TEAL]),
        use_container_width=True,
    )

st.markdown("#### Sales over time for a single market")
single_market = st.selectbox("Market (Store)", options=sorted(df["Store"].unique()))
trend = df[df["Store"] == single_market].sort_values("Date")
fig_trend = px.line(trend, x="Date", y="Weekly_Sales", title=f"Weekly sales over time — Store {single_market}",
                     color_discrete_sequence=[TEAL])
# Shade Ramadan periods with ONE vrect per contiguous window, not a vline
# per row — looping add_vline is O(n^2) and can hang (see earlier fix).
ramadan_weeks = trend.loc[trend["is_ramadan_window"], "Date"].sort_values()
if not ramadan_weeks.empty:
    # split into contiguous blocks (one per year) wherever the gap exceeds 60 days
    gaps = ramadan_weeks.diff().dt.days.fillna(0)
    block_id = (gaps > 60).cumsum()
    for _, block in ramadan_weeks.groupby(block_id):
        fig_trend.add_vrect(x0=block.min(), x1=block.max(), fillcolor=MINT, opacity=0.15, line_width=0)
st.plotly_chart(fig_trend, use_container_width=True)

overall_avg = df.groupby("season")["Weekly_Sales"].mean()
diff_pct = (overall_avg["Ramadan/Eid window"] - overall_avg["Regular period"]) / overall_avg["Regular period"] * 100
st.metric("Overall: Ramadan/Eid vs regular period sales", f"{diff_pct:+.1f}%")