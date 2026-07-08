"""Data loading: fetch from Our World in Data API or fall back to bundled CSV."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://ourworldindata.org/grapher/"
    "share-adults-bank-account-financial-institution-mobile-money.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)
USER_AGENT = "mobile-money-trends/1.0 (pure-python)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = PROJECT_ROOT / "data" / "sample_mobile_money_data.csv"


def fetch_mobile_money_data_from_api(url: str = DATA_URL) -> pd.DataFrame | None:
    """Fetch live data from Our World in Data. Returns None on failure."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return _normalize(df)
    except Exception as exc:  # pragma: no cover - network-dependent
        print(f"[data] API fetch failed ({exc}); falling back to local CSV")
        return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and compute derived share columns."""
    rename = {}
    if "Entity" in df.columns:
        rename["Entity"] = "country"
    if "entity" in df.columns:
        rename["entity"] = "country"
    if "Code" in df.columns:
        rename["Code"] = "country_code"
    if "code" in df.columns:
        rename["code"] = "country_code"
    if "Year" in df.columns:
        rename["Year"] = "year"
    if "year" in df.columns:
        rename["year"] = "year"
    df = df.rename(columns=rename)

    if "only_financial_institution_account" not in df.columns:
        # Fallback: older OWID used different column names
        for old, new in {
            "fin1a.t.d": "only_financial_institution_account",
            "fin1a.t.m": "only_mobile_money_account",
            "fin1a.t": "both_accounts",
        }.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

    def _col(name: str):
        return df[name] if name in df.columns else 0

    both = _col("both_accounts")
    only_fin = _col("only_financial_institution_account")
    only_mob = _col("only_mobile_money_account")

    df["financial_institution_share"] = only_fin + both
    df["mobile_money_share"] = only_mob + both

    essential = [c for c in ["country", "country_code", "year"] if c in df.columns]
    others = [c for c in df.columns if c not in essential and df[c].notna().any()]
    df = df[essential + others].copy()

    if "year" in df.columns:
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        sort_cols = [c for c in ["country", "year"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def load_local_csv(path: Path = LOCAL_CSV) -> pd.DataFrame:
    """Load the bundled CSV shipped with the repo."""
    df = pd.read_csv(path)
    return _normalize(df)


def load_mobile_money_data(
    path: str | Path = LOCAL_CSV,
    use_api: bool = True,
    persist: bool = True,
) -> pd.DataFrame:
    """Load data: live API first, then local CSV fallback.

    Parameters
    ----------
    path : path to local CSV (used as cache / fallback)
    use_api : whether to attempt a network fetch
    persist : save successful API fetches back to the local CSV cache
    """
    path = Path(path)
    df: pd.DataFrame | None = None

    if use_api:
        df = fetch_mobile_money_data_from_api()

    if df is None or df.empty:
        df = load_local_csv(path)

    if persist and path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        except OSError:
            pass

    return df
