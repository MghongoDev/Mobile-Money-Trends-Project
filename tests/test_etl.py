import pandas as pd
import pytest
from pathlib import Path

from mobile_money_project.etl import extract_mobile_money_data, transform_mobile_money_data


def test_extract_mobile_money_data():
    """Test data extraction from API or fallback."""
    df = extract_mobile_money_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "year" in df.columns


def test_transform_mobile_money_data():
    """Test data transformation."""
    # Create sample data
    sample_data = pd.DataFrame({
        "country": ["TestCountry"],
        "year": [2020],
        "mobile_money_share": [0.1],
        "financial_institution_share": [0.3]
    })
    transformed = transform_mobile_money_data(sample_data)
    assert "mobile_growth_pct" in transformed.columns
    assert "digital_inclusion_index" in transformed.columns