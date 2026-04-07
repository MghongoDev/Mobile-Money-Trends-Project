from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import summarize_mobile_money_trends
from .data import load_mobile_money_data
from .modeling import forecast_mobile_money, train_mobile_money_forecast
from .preprocessing import prepare_mobile_money_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_mobile_money_data.csv"


def extract_mobile_money_data(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    return load_mobile_money_data(str(data_path))


def transform_mobile_money_data(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_mobile_money_data(df)


def load_mobile_money_dataset(path: str | Path | None = None) -> pd.DataFrame:
    raw_df = extract_mobile_money_data(path)
    return transform_mobile_money_data(raw_df)


def run_mobile_money_etl(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    df = load_mobile_money_dataset(path)
    summary = summarize_mobile_money_trends(df)
    return df, summary


def build_mobile_money_forecast(df: pd.DataFrame, forecast_horizon: int = 12) -> dict:
    pipeline, backtest, y_test, metrics = train_mobile_money_forecast(df)
    forecast_df = forecast_mobile_money(df, pipeline, forecast_horizon)
    return {
        "pipeline": pipeline,
        "backtest": backtest,
        "metrics": metrics,
        "forecast": forecast_df,
    }
