import pandas as pd
import pytest

from mobile_money_project.etl import extract, transform, run_mobile_money_etl


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return extract(use_api=False)


def test_extract_returns_dataframe(raw_frame):
    assert isinstance(raw_frame, pd.DataFrame)
    assert not raw_frame.empty
    assert "year" in raw_frame.columns


def test_transform_adds_features(raw_frame):
    prepared = transform(raw_frame)
    for col in ("mobile_growth_pct", "financial_growth_pct",
                "account_ratio", "digital_inclusion_index", "trend_factor"):
        assert col in prepared.columns


def test_run_etl(raw_frame):
    df, summary = run_mobile_money_etl(use_api=False)
    assert "r2" not in summary  # summary dict is analytics, not model
    assert "time_periods" in summary
    assert "countries" in summary
    assert summary["time_periods"] == len(df)
