from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def summarize_mobile_money_trends(df: pd.DataFrame) -> dict:
    """Return a trend summary for mobile money and financial institution accounts."""
    working = df.copy()
    
    # Find mobile and financial columns
    mobile_cols = [col for col in df.columns if "mobile" in col.lower()]
    financial_cols = [col for col in df.columns if "financial" in col.lower() or "bank" in col.lower()]
    
    mobile_col = mobile_cols[0] if mobile_cols else None
    financial_col = financial_cols[0] if financial_cols else None
    
    summary = {
        "time_periods": len(working),
        "average_mobile_growth_pct": float(working["mobile_growth_pct"].mean()),
        "average_financial_growth_pct": float(working["financial_growth_pct"].mean()),
    }
    
    # Calculate trends if columns exist
    if mobile_col and mobile_col in working.columns:
        working_with_index = working[[mobile_col]].reset_index(drop=True)
        working_with_index["index"] = np.arange(len(working_with_index))
        try:
            mobile_model = LinearRegression().fit(
                working_with_index[["index"]], 
                working_with_index[[mobile_col]]
            )
            summary["mobile_trend_slope"] = float(mobile_model.coef_[0, 0])
            summary["final_mobile_share"] = float(working[mobile_col].iloc[-1])
        except Exception:
            summary["mobile_trend_slope"] = 0.0
            summary["final_mobile_share"] = float(working[mobile_col].iloc[-1] if mobile_col in working.columns else 0)
    
    if financial_col and financial_col in working.columns:
        working_with_index = working[[financial_col]].reset_index(drop=True)
        working_with_index["index"] = np.arange(len(working_with_index))
        try:
            financial_model = LinearRegression().fit(
                working_with_index[["index"]], 
                working_with_index[[financial_col]]
            )
            summary["financial_trend_slope"] = float(financial_model.coef_[0, 0])
            summary["final_financial_share"] = float(working[financial_col].iloc[-1])
        except Exception:
            summary["financial_trend_slope"] = 0.0
            summary["final_financial_share"] = float(working[financial_col].iloc[-1] if financial_col in working.columns else 0)
    
    if "account_gap" in working.columns:
        summary["latest_account_gap"] = int(working["account_gap"].iloc[-1])
    
    summary["countries"] = int(working["country"].nunique()) if "country" in working.columns else 1
    summary["year_range"] = f"{int(working['year'].min())}-{int(working['year'].max())}" if "year" in working.columns else "N/A"
    
    return summary
