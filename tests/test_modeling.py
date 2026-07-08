import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mobile_money_project.modeling import (
    compare_models,
    forecast_mobile_money,
    train_mobile_money_forecast,
)


def _make_frame(n_years: int = 20) -> pd.DataFrame:
    years = np.arange(2004, 2004 + n_years)
    mobile = 1 + 0.6 * np.arange(n_years)            # rising adoption
    fin = 20 + 0.2 * np.arange(n_years)
    return pd.DataFrame({
        "country": ["Testia"] * n_years,
        "year": years,
        "only_mobile_money_account": mobile,
        "mobile_money_share": mobile + 0.5,
        "financial_institution_share": fin,
    })


def test_train_and_forecast():
    df = _make_frame()
    # pre-process first so the model has the engineered features
    from mobile_money_project.preprocessing import prepare_mobile_money_data
    prepared = prepare_mobile_money_data(df)

    pipe, backtest, y_test, metrics = train_mobile_money_forecast(prepared)
    assert isinstance(pipe, Pipeline)
    assert {"mae", "rmse", "r2", "features_used", "target"} <= metrics.keys()
    assert metrics["r2"] > 0.5  # toy data should fit well
    assert len(backtest) == metrics["test_size"]

    future = forecast_mobile_money(prepared, pipe, forecast_horizon=5)
    assert len(future) == 5
    assert "year" in future.columns


def test_compare_models():
    from mobile_money_project.preprocessing import prepare_mobile_money_data
    prepared = prepare_mobile_money_data(_make_frame(30))
    comparison = compare_models(prepared)
    assert {"Linear Regression", "Random Forest", "Gradient Boosting"} <= comparison.keys()
    for name, scores in comparison.items():
        assert {"mae", "rmse", "r2"} <= scores.keys()
