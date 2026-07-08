"""Trend summaries and high-level analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _slope(series: pd.Series) -> float:
    """Slope of a linear regression of *series* vs its index."""
    s = series.dropna().reset_index(drop=True)
    if len(s) < 2:
        return 0.0
    x = np.arange(len(s)).reshape(-1, 1)
    model = LinearRegression().fit(x, s.values.reshape(-1, 1))
    return float(model.coef_[0, 0])


def _find_col(df: pd.DataFrame, *needles: str) -> str | None:
    for needle in needles:
        for col in df.columns:
            if needle in col.lower():
                return col
    return None


def summarize_mobile_money_trends(df: pd.DataFrame) -> dict:
    """Return a global/country summary dict (same shape for API use)."""
    work = df.copy()
    mobile_col = _find_col(work, "mobile_money_share", "mobile")
    financial_col = _find_col(work, "financial_institution_share", "financial", "bank")

    summary: dict = {"time_periods": int(len(work))}

    if "mobile_growth_pct" in work.columns:
        finite = work["mobile_growth_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        summary["average_mobile_growth_pct"] = float(finite.mean()) if len(finite) else 0.0
    else:
        summary["average_mobile_growth_pct"] = 0.0

    if "financial_growth_pct" in work.columns:
        finite = work["financial_growth_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        summary["average_financial_growth_pct"] = float(finite.mean()) if len(finite) else 0.0
    else:
        summary["average_financial_growth_pct"] = 0.0

    if mobile_col:
        summary["mobile_trend_slope"] = _slope(work[mobile_col])
        summary["final_mobile_share"] = float(work[mobile_col].dropna().iloc[-1]) if work[mobile_col].notna().any() else 0.0
    else:
        summary["mobile_trend_slope"] = 0.0
        summary["final_mobile_share"] = 0.0

    if financial_col:
        summary["financial_trend_slope"] = _slope(work[financial_col])
        summary["final_financial_share"] = float(work[financial_col].dropna().iloc[-1]) if work[financial_col].notna().any() else 0.0
    else:
        summary["financial_trend_slope"] = 0.0
        summary["final_financial_share"] = 0.0

    if "account_gap" in work.columns:
        summary["latest_account_gap"] = float(work["account_gap"].dropna().iloc[-1]) if work["account_gap"].notna().any() else 0.0

    summary["countries"] = int(work["country"].nunique()) if "country" in work.columns else 1
    if "year" in work.columns and work["year"].notna().any():
        summary["year_range"] = f"{int(work['year'].min())}-{int(work['year'].max())}"
    else:
        summary["year_range"] = "N/A"

    return summary


def country_summary(df: pd.DataFrame, country: str) -> tuple[pd.DataFrame, dict]:
    """Return (records, summary) for one country, or the global aggregate."""
    if country == "__all__" or country is None:
        records = df.copy()
        numeric_cols = records.select_dtypes(include="number").columns.tolist()
        if "year" in records.columns and numeric_cols:
            agg = records.groupby("year", as_index=False)[numeric_cols].mean()
            return agg, summarize_mobile_money_trends(records)
        return records, summarize_mobile_money_trends(records)

    subset = df[df["country"] == country].copy() if "country" in df.columns else df.copy()
    return subset.sort_values("year") if "year" in subset.columns else subset, summarize_mobile_money_trends(subset)


def top_countries(df: pd.DataFrame, year: int | None = None, n: int = 10) -> pd.DataFrame:
    """Return top-N countries by latest mobile-money share."""
    work = df.copy()
    if "country" not in work.columns:
        return pd.DataFrame()
    if year is not None and "year" in work.columns:
        work = work[work["year"] == year]
    if work.empty:
        return pd.DataFrame()

    mobile_col = _find_col(work, "mobile_money_share", "mobile") or "mobile_money_share"
    # Keep one row per country = latest year available
    latest_idx = work.groupby("country")["year"].idxmax() if "year" in work.columns else work.index
    latest = work.loc[latest_idx].sort_values(mobile_col, ascending=False)
    keep = [c for c in ["country", "country_code", "year", mobile_col,
                        "financial_institution_share", "account_gap"] if c in latest.columns]
    return latest[keep].head(n).reset_index(drop=True)
