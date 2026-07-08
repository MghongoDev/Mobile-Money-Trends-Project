"""Cleaning and feature engineering for the mobile money dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, *needles: str) -> str | None:
    """Return the first column whose (lower-cased) name contains any needle."""
    for needle in needles:
        for col in df.columns:
            if needle in col.lower():
                return col
    return None


def prepare_mobile_money_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, sort and engineer features on the raw OWID-style dataframe."""
    df = df.copy()

    # --- date/year normalisation -------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["year"] = df["date"].dt.year
    elif "year" in df.columns:
        sort_cols = [c for c in ["country", "year"] if c in df.columns]
        df = df.sort_values(sort_cols)

    mobile_col = _find_col(df, "mobile_money_share", "mobile")
    financial_col = _find_col(df, "financial_institution_share", "financial", "bank")

    def _group_pct_change(series: pd.Series) -> pd.Series:
        if "country" in df.columns:
            return series.groupby(df["country"]).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["mobile_growth_pct"] = _group_pct_change(df[mobile_col]) if mobile_col else 0.0
    df["financial_growth_pct"] = _group_pct_change(df[financial_col]) if financial_col else 0.0

    # Account ratio (mobile / financial), safe against division by zero
    if mobile_col and financial_col:
        fin_safe = df[financial_col].replace(0, np.nan)
        df["account_ratio"] = (df[mobile_col] / fin_safe).fillna(0.0)
    else:
        df["account_ratio"] = 0.0

    # Account gap = financial - mobile (in percentage points)
    if mobile_col and financial_col:
        df["account_gap"] = df[financial_col] - df[mobile_col]
    else:
        df["account_gap"] = 0.0

    # Digital inclusion index: 50/50 blend of normalised mobile & financial shares
    inclusion = []
    if mobile_col:
        series = df[mobile_col]
        if series.max() > 1.5:  # percentages on 0-100 scale
            series = series / 100.0
        inclusion.append(series.clip(0, 1) * 0.5)
    if financial_col:
        series = df[financial_col]
        if series.max() > 1.5:
            series = series / 100.0
        inclusion.append(series.clip(0, 1) * 0.5)
    if inclusion:
        df["digital_inclusion_index"] = sum(inclusion)
    else:
        df["digital_inclusion_index"] = 0.5

    # Trend factor in [0, 1] over the time range
    if "year" in df.columns:
        ymin, ymax = df["year"].min(), df["year"].max()
        span = max(1, ymax - ymin)
        df["trend_factor"] = (df["year"] - ymin) / span
    else:
        df["trend_factor"] = np.linspace(0, 1, len(df))

    return df
