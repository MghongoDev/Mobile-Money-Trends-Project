"""Mobile money adoption analysis package."""

from .data import generate_synthetic_mobile_money_data, load_mobile_money_data
from .preprocessing import prepare_mobile_money_data
from .analysis import summarize_mobile_money_trends
from .modeling import train_mobile_money_forecast, forecast_mobile_money

__all__ = [
    "generate_synthetic_mobile_money_data",
    "load_mobile_money_data",
    "prepare_mobile_money_data",
    "summarize_mobile_money_trends",
    "train_mobile_money_forecast",
    "forecast_mobile_money",
]
