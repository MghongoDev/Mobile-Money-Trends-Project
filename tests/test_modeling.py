import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mobile_money_project.modeling import train_mobile_money_forecast


def test_train_mobile_money_forecast():
    """Test model training."""
    # Create sample data
    sample_data = pd.DataFrame({
        "year": list(range(2010, 2020)),
        "trend_factor": list(range(10)),
        "mobile_money_share": [0.1 + i*0.01 for i in range(10)],
        "financial_institution_share": [0.3 + i*0.005 for i in range(10)]
    })
    pipeline, backtest, y_test, metrics = train_mobile_money_forecast(sample_data)
    assert isinstance(pipeline, Pipeline)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics