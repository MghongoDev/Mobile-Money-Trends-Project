"""Streamlit dashboard – the main UI for the Mobile Money Trends project.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mobile_money_project.analysis import country_summary, top_countries
from mobile_money_project.data import load_mobile_money_data
from mobile_money_project.etl import build_mobile_money_forecast, run_mobile_money_etl
from mobile_money_project.modeling import (
    compare_models,
    explain_model,
    tune_hyperparameters,
)

st.set_page_config(
    page_title="Mobile Money Adoption",
    
    layout="wide",
)

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data ...")
def load_data(use_api: bool) -> tuple[pd.DataFrame, dict]:
    return run_mobile_money_etl(use_api=use_api)


@st.cache_data(show_spinner="Training model ...")
def train_model(_df: pd.DataFrame, horizon: int) -> dict:
    return build_mobile_money_forecast(_df, forecast_horizon=horizon)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📱 Mobile Money Trends")
    st.caption("Pure-Python analytics & forecasting")

    use_api = st.checkbox("Fetch live data from Our World in Data", value=False)
    horizon = st.slider("Forecast horizon (years)", 1, 25, 12)
    st.divider()
    st.markdown("### About")
    st.markdown(
        "This dashboard is written entirely in Python "
        "(Streamlit + pandas + scikit-learn + plotly)."
    )
    st.markdown("Data: [Our World in Data](https://ourworldindata.org)")

# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------
try:
    df, summary = load_data(use_api=use_api)
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

model_result = train_model(df, horizon=horizon)
metrics = model_result["metrics"]
forecast = model_result["forecast"]
backtest = model_result["backtest"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Mobile Money Adoption Dashboard")
st.subheader("Trends, analysis and forecasting – 100% Python")

kpis = st.columns(6)
kpis[0].metric("Records", f"{summary['time_periods']:,}")
kpis[1].metric("Countries", f"{summary['countries']}")
kpis[2].metric("Year range", summary["year_range"])
kpis[3].metric(f"{metrics['target']} R²", f"{metrics['r2']:.3f}")
kpis[4].metric("MAE", f"{metrics['mae']:.3f}")
kpis[5].metric("RMSE", f"{metrics['rmse']:.3f}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_global, tab_country, tab_model, tab_forecast, tab_raw = st.tabs(
    ["🌍 Global trends", "🏳️ Country explorer", "🤖 Model", "🔮 Forecast", "📄 Raw data"]
)

# ---- Global trends ---------------------------------------------------------
with tab_global:
    st.markdown("### Global average adoption over time")
    numeric = df.select_dtypes(include="number").columns.tolist()
    annual = df.groupby("year", as_index=False)[numeric].mean() if "year" in df.columns else df
    melt = annual.melt(
        id_vars="year",
        value_vars=[c for c in ["mobile_money_share", "financial_institution_share"] if c in annual.columns],
        var_name="metric", value_name="share (%)",
    )
    fig = px.line(melt, x="year", y="share (%)", color="metric", markers=True,
                  color_discrete_map={
                      "mobile_money_share": "#2563eb",
                      "financial_institution_share": "#10b981",
                  })
    fig.update_layout(height=420, xaxis_title="Year", yaxis_title="Share of adults (%)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top countries by latest mobile-money share")
        top = top_countries(df, n=15)
        if not top.empty:
            mob_col = [c for c in top.columns if "mobile" in c.lower() and "share" in c.lower()]
            mob_col = mob_col[0] if mob_col else top.columns[-1]
            fig2 = px.bar(top.sort_values(mob_col), x=mob_col, y="country", orientation="h",
                          color=mob_col, color_continuous_scale="Blues")
            fig2.update_layout(height=450, yaxis_title="", xaxis_title=f"{mob_col} (%)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No country column found.")

    with c2:
        st.markdown("#### Digital inclusion index over time")
        if "digital_inclusion_index" in annual.columns:
            fig3 = px.area(annual, x="year", y="digital_inclusion_index",
                           color_discrete_sequence=["#14b8a6"])
            fig3.update_layout(height=450, yaxis_title="Inclusion index (0–1)")
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Global summary statistics"):
        st.json(summary)

# ---- Country explorer ------------------------------------------------------
with tab_country:
    countries = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
    choice = st.selectbox("Pick a country", ["__all__ (Global average)"] + countries)
    country_key = None if choice.startswith("__all__") else choice
    records, csum = country_summary(df, country_key)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Periods", csum["time_periods"])
    col_b.metric("Mobile trend slope", f"{csum['mobile_trend_slope']:.3f}")
    col_c.metric("Final mobile share", f"{csum['final_mobile_share']:.2f}%")
    col_d.metric("Latest account gap", f"{csum['latest_account_gap']:.2f}")

    st.markdown("##### Account shares")
    if "year" in records.columns:
        m = records.melt(id_vars="year",
                         value_vars=[c for c in ["mobile_money_share", "financial_institution_share"]
                                     if c in records.columns],
                         var_name="metric", value_name="share (%)")
        fig = px.line(m, x="year", y="share (%)", color="metric", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    if {"year", "mobile_growth_pct", "financial_growth_pct"}.issubset(records.columns):
        st.markdown("##### Year-over-year growth rates")
        gm = records.melt(id_vars="year",
                          value_vars=["mobile_growth_pct", "financial_growth_pct"],
                          var_name="metric", value_name="growth")
        gm["growth"] = gm["growth"].replace([np.inf, -np.inf], np.nan)
        fig2 = px.bar(gm.dropna(), x="year", y="growth", color="metric", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(records, use_container_width=True)

    @st.cache_data
    def _csv(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download country data as CSV",
        data=_csv(records),
        file_name=f"mobile_money_{country_key or 'global'}.csv",
        mime="text/csv",
    )

# ---- Model tab -------------------------------------------------------------
with tab_model:
    st.markdown("### Model performance")
    st.write(f"**Target:** `{metrics['target']}`  ")
    st.write(f"**Features:** {', '.join(metrics['features_used'])}")
    st.write(f"**Train / test split:** {metrics['train_size']} / {metrics['test_size']} rows (time-based)")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['mae']:.4f}")
    m2.metric("RMSE", f"{metrics['rmse']:.4f}")
    m3.metric("R²", f"{metrics['r2']:.4f}")

    st.markdown("#### Backtest: actual vs predicted")
    if {"actual", "prediction"}.issubset(backtest.columns):
        if "year" in backtest.columns:
            bt = backtest.sort_values("year")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["year"], y=bt["actual"], name="Actual", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=bt["year"], y=bt["prediction"], name="Predicted", mode="lines+markers"))
            fig.update_layout(height=400, xaxis_title="Year", yaxis_title=metrics["target"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(backtest, x="actual", y="prediction", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Compare against other regressors"):
        with st.spinner("Comparing models ..."):
            comparison = compare_models(df)
        st.dataframe(pd.DataFrame(comparison).T.style.format("{:.4f}"))

    with st.expander("Feature importance"):
        imp = explain_model(df, model_result["pipeline"])["feature_importance"]
        imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
        fig = px.bar(imp_df.sort_values("importance", ascending=True),
                     x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="Tealgrn")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tune hyper-parameters (quick random search)"):
        if st.button("Run 20-iteration random search"):
            with st.spinner("Tuning ..."):
                best = tune_hyperparameters(df, n_trials=20)
            st.json(best)

# ---- Forecast tab ----------------------------------------------------------
with tab_forecast:
    st.markdown(f"### {horizon}-year forecast for `{metrics['target']}`")
    target_col = f"forecast_{metrics['target']}"
    if "year" in forecast.columns:
        fig = px.line(forecast, x="year", y=target_col, markers=True,
                      color_discrete_sequence=["#ef4444"])
        fig.update_layout(height=420, xaxis_title="Year", yaxis_title=target_col)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast, use_container_width=True)
    st.download_button(
        "Download forecast CSV",
        data=forecast.to_csv(index=False).encode(),
        file_name="mobile_money_forecast.csv",
        mime="text/csv",
    )

# ---- Raw data tab ----------------------------------------------------------
with tab_raw:
    st.markdown("### Prepared dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download prepared dataset as CSV",
        data=df.to_csv(index=False).encode(),
        file_name="mobile_money_prepared.csv",
        mime="text/csv",
    )
