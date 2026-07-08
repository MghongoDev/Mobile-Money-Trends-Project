"""High-level ETL orchestration (pure-Python, no external workflow engine)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import summarize_mobile_money_trends
from .data import LOCAL_CSV, load_mobile_money_data
from .modeling import forecast_mobile_money, train_mobile_money_forecast
from .preprocessing import prepare_mobile_money_data


def extract(path: str | Path | None = None, use_api: bool = True) -> pd.DataFrame:
    return load_mobile_money_data(path or LOCAL_CSV, use_api=use_api)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_mobile_money_data(df)


def load(df: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    """Persist the transformed frame to CSV and return it (useful for caching)."""
    target = Path(path) if path else LOCAL_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return df


def run_mobile_money_etl(
    path: str | Path | None = None,
    use_api: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Extract -> Transform -> summarize. Returns (prepared_df, summary_dict)."""
    raw = extract(path, use_api=use_api)
    prepared = transform(raw)
    summary = summarize_mobile_money_trends(prepared)
    return prepared, summary


def build_mobile_money_forecast(
    df: pd.DataFrame,
    forecast_horizon: int = 12,
) -> dict:
    """Train and forecast. Returns dict with pipeline/backtest/metrics/forecast."""
    pipeline, backtest, _y_test, metrics = train_mobile_money_forecast(df)
    forecast_df = forecast_mobile_money(df, pipeline, forecast_horizon=forecast_horizon)
    return {
        "pipeline": pipeline,
        "backtest": backtest,
        "metrics": metrics,
        "forecast": forecast_df,
    }
