import pandas as pd


def prepare_mobile_money_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the mobile money dataset for analysis and modeling."""
    df = df.copy()
    
    # Handle date column if it exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]
    elif "year" in df.columns:
        if "country" in df.columns:
            df = df.sort_values(["country", "year"])
        else:
            df = df.sort_values("year")
        
    # Identify account columns more flexibly
    mobile_cols = [col for col in df.columns if "mobile" in col.lower()]
    bank_cols = [col for col in df.columns if "bank" in col.lower()]
    financial_cols = [col for col in df.columns if "financial" in col.lower()]
    
    mobile_col = mobile_cols[0] if mobile_cols else None
    financial_col = financial_cols[0] if financial_cols else (bank_cols[0] if bank_cols else None)
    
    # Calculate growth percentages if columns exist
    if mobile_col and mobile_col in df.columns:
        if "country" in df.columns:
            df["mobile_growth_pct"] = df.groupby("country")[mobile_col].pct_change().fillna(0).values
        else:
            df["mobile_growth_pct"] = df[mobile_col].pct_change().fillna(0)
    else:
        df["mobile_growth_pct"] = 0.0
        
    if financial_col and financial_col in df.columns:
        if "country" in df.columns:
            df["financial_growth_pct"] = df.groupby("country")[financial_col].pct_change().fillna(0).values
        else:
            df["financial_growth_pct"] = df[financial_col].pct_change().fillna(0)
    else:
        df["financial_growth_pct"] = 0.0
    
    # Calculate account ratio
    if mobile_col and financial_col:
        df["account_ratio"] = df[mobile_col] / df[financial_col].replace(0, 1)
    else:
        df["account_ratio"] = 0.0
    
    # Create digital inclusion index from available share columns
    inclusion_components = []
    for col in df.columns:
        col_lower = col.lower()
        if "mobile" in col_lower and "share" in col_lower:
            # Assume values are percentages (0-100) from Our World in Data
            values = df[col] / 100 if df[col].max() > 1 else df[col]
            inclusion_components.append(values * 0.5)
        elif "financial" in col_lower and "share" in col_lower:
            values = df[col] / 100 if df[col].max() > 1 else df[col]
            inclusion_components.append(values * 0.5)
    
    if inclusion_components:
        df["digital_inclusion_index"] = sum(inclusion_components)
    elif mobile_col or financial_col:
        # Fallback: use average of available columns
        available_share_cols = [col for col in df.columns if "share" in col.lower() or "account" in col.lower()]
        if available_share_cols:
            df["digital_inclusion_index"] = df[available_share_cols].mean(axis=1, skipna=True).fillna(0.5)
        else:
            df["digital_inclusion_index"] = 0.5
    else:
        df["digital_inclusion_index"] = 0.5
    
    # Create trend factor
    if "year" in df.columns:
        min_year = df["year"].min()
        max_year = df["year"].max()
        df["trend_factor"] = (df["year"] - min_year) / max(1, (max_year - min_year))
    else:
        df["trend_factor"] = df.index / max(1, len(df) - 1)
    
    return df
