import streamlit as st
import plotly.express as px
from utils import (inject_theme, load_business_impact, load_model_comparison,
                    load_hijri_comparison, load_dept_breakdown, TEAL, MINT)

st.set_page_config(page_title="CrescentIQ — Business Impact", layout="wide", page_icon="💰")
inject_theme()
st.title("💰 What the Hijri Feature Engineering Is Worth")

impact = load_business_impact()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg error before Hijri features", f"${impact['mae_no_hijri_usd']:,.0f}")
with col2:
    st.metric("Avg error with Hijri features", f"${impact['mae_with_hijri_usd']:,.0f}",
               delta=f"-${impact['mae_reduction_per_prediction_usd']:,.0f}", delta_color="normal")
with col3:
    st.metric("Total forecast error reduced (test period)", f"${impact['total_forecast_error_reduced_usd']:,.0f}")

st.caption(
    "This is an illustrative proxy, not a real P&L figure: it's the drop in average "
    f"forecast error (MAE) × {int(impact['n_ramadan_eid_predictions_in_test']):,} "
    "store-department-week predictions made during Ramadan/Eid windows in the test set. "
    "A real financial estimate would need actual holding-cost and stockout-cost "
    "assumptions per unit."
)

st.markdown("---")
st.markdown("#### How we got here")

comparison = load_hijri_comparison()
st.plotly_chart(
    px.bar(comparison.reset_index().rename(columns={"index": "version"}),
           x="version", y="ramadan/eid_window_rmse",
           title="Ramadan/Eid window RMSE — with vs without Hijri features",
           color_discrete_sequence=[TEAL], text_auto=".0f"),
    use_container_width=True,
)

models_df = load_model_comparison()
st.plotly_chart(
    px.bar(models_df.reset_index().rename(columns={"index": "model"}),
           x="model", y="overall_rmse", title="Model comparison — overall RMSE (lower is better)",
           color_discrete_sequence=[TEAL], text_auto=".0f"),
    use_container_width=True,
)

st.markdown("#### Where the value concentrates")
per_dept = load_dept_breakdown()
top_n = st.slider("Show top N departments", 5, 30, 15)
st.plotly_chart(
    px.bar(per_dept.head(top_n), x="Dept", y="improvement_%",
           title=f"RMSE improvement % by Department (top {top_n})",
           color_discrete_sequence=[MINT]),
    use_container_width=True,
)
st.caption("Departments with the highest improvement are where the Hijri features "
           "capture the strongest real seasonal signal — worth highlighting by name in the pitch.")