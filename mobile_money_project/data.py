import os

import pandas as pd
import requests


def fetch_mobile_money_data_from_api() -> pd.DataFrame:
    """Fetch real mobile money adoption data from Our World in Data API."""
    url = "https://ourworldindata.org/grapher/share-adults-bank-account-financial-institution-mobile-money.csv?v=1&csvType=full&useColumnShortNames=true"
    headers = {"User-Agent": "Mobile Money Analysis Project/1.0"}
    
    try:
        df = pd.read_csv(url, storage_options={"User-Agent": headers["User-Agent"]})
        
        # Standardize column names
        df = df.rename(columns={
            "entity": "country",
            "code": "country_code",
        })
        
        # Calculate derived metrics
        # Assume: financial_institution_share = only_financial_institution_account + both_accounts
        # Assume: mobile_money_share = only_mobile_money_account + both_accounts
        df["financial_institution_share"] = (
            df.get("only_financial_institution_account", 0) + df.get("both_accounts", 0)
        )
        df["mobile_money_share"] = (
            df.get("only_mobile_money_account", 0) + df.get("both_accounts", 0)
        )
        
        # Keep essential columns
        essential_cols = ["country", "country_code", "year"]
        metric_cols = [col for col in df.columns if col not in essential_cols and df[col].notna().any()]
        
        df = df[essential_cols + metric_cols].copy()
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df.sort_values(["country", "year"]).reset_index(drop=True)
        
        print(f"[INFO] Successfully fetched {len(df)} records from Our World in Data")
        return df
    except Exception as e:
        print(f"[WARNING] Failed to fetch from API: {e}. Using fallback sample data.")
        return None


def generate_synthetic_mobile_money_data(
    start_date: str = "2016-01-01",
    periods: int = 120,
    freq: str = "MS",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic monthly dataset for mobile money analysis (fallback)."""
    import numpy as np
    
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    months = np.arange(periods)

    population = 45_000_000 + months * 155_000 + np.random.normal(0, 190_000, periods)
    population = np.clip(population, 44_000_000, None)

    internet_penetration = np.clip(0.24 + 0.008 * months + np.random.normal(0, 0.01, periods), 0.1, 0.95)
    smartphone_penetration = np.clip(0.18 + 0.010 * months + np.random.normal(0, 0.01, periods), 0.08, 0.9)
    policy_support_index = np.clip(30 + 0.18 * months + np.random.normal(0, 3.0, periods), 25, 100)

    mobile_share = np.clip(0.03 + 0.0009 * months + 0.0003 * policy_support_index, 0.04, 0.65)
    financial_share = np.clip(0.18 + 0.0004 * months + 0.00035 * internet_penetration, 0.18, 0.85)

    mobile_money_accounts = np.round(population * mobile_share).astype(int)
    financial_institution_accounts = np.round(population * financial_share).astype(int)
    account_gap = financial_institution_accounts - mobile_money_accounts

    data = pd.DataFrame(
        {
            "date": dates,
            "population": np.round(population).astype(int),
            "internet_penetration": np.round(internet_penetration, 4),
            "smartphone_penetration": np.round(smartphone_penetration, 4),
            "policy_support_index": np.round(policy_support_index, 1),
            "mobile_money_accounts": mobile_money_accounts,
            "financial_institution_accounts": financial_institution_accounts,
            "account_gap": account_gap,
        }
    )
    return data


def load_mobile_money_data(path: str = "data/sample_mobile_money_data.csv") -> pd.DataFrame:
    """Load the mobile money dataset from API or disk fallback."""
    # Try to fetch from API first
    df = fetch_mobile_money_data_from_api()
    
    if df is not None and not df.empty:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return df
    
    # Fallback to disk if available
    if os.path.exists(path):
        print(f"[INFO] Loaded data from {path}")
        return pd.read_csv(path)
    
    # Generate synthetic data as last resort
    print("[INFO] Generating synthetic fallback data")
    df = generate_synthetic_mobile_money_data()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df 
