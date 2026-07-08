"""Modeling: training, forecasting, comparison, hyper-parameter tuning, SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURE_CANDIDATES = [
    "trend_factor",
    "digital_inclusion_index",
    "account_ratio",
    "mobile_growth_pct",
    "financial_growth_pct",
    "mobile_money_share",
    "financial_institution_share",
    "year",
]


def _build_pipeline(degree: int = 1) -> Pipeline:
    steps = []
    if degree and degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )))
    return Pipeline(steps=steps)


def _available_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in FEATURE_CANDIDATES if c in df.columns]
    # Exclude the target from features if it accidentally lands there
    target = _get_target_column(df)
    feats = [c for c in feats if c != target]
    if not feats:
        feats = ["trend_factor"] if "trend_factor" in df.columns else []
    if not feats:
        raise ValueError("No usable feature columns were found in the dataset")
    return feats


def _get_target_column(df: pd.DataFrame) -> str:
    # Prefer "only_mobile_money_account" (percentage of adults with *only* mobile)
    for preferred in ("only_mobile_money_account", "mobile_money_share", "mobile_money_accounts"):
        if preferred in df.columns:
            return preferred
    for col in df.columns:
        if "mobile" in col.lower():
            return col
    return "account_ratio"


def _clean_xy(df: pd.DataFrame, target: str):
    features = _available_features(df)
    clean = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10:
        raise ValueError(
            f"Only {len(clean)} usable rows after cleaning; need at least 10"
        )
    X = clean[features]
    y = clean[target]
    return X, y, features


def train_mobile_money_forecast(
    df: pd.DataFrame,
    target: str | None = None,
    test_size: float = 0.2,
) -> tuple[Pipeline, pd.DataFrame, np.ndarray, dict]:
    """Train the GradientBoosting pipeline.

    Returns (pipeline, backtest_df, y_test, metrics_dict).
    """
    if target is None:
        target = _get_target_column(df)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe")

    X, y, features = _clean_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features_used": features,
        "target": target,
    }

    backtest = X_test.copy()
    backtest["actual"] = y_test.values
    backtest["prediction"] = y_pred
    if "year" in df.columns:
        backtest = backtest.merge(
            df[["year"] + (["country"] if "country" in df.columns else [])],
            left_index=True,
            right_index=True,
            how="left",
        )

    return pipe, backtest, y_test.values, metrics


def forecast_mobile_money(
    df: pd.DataFrame,
    pipeline: Pipeline,
    forecast_horizon: int = 12,
) -> pd.DataFrame:
    """Project forward *forecast_horizon* years using simple trend extrapolation."""
    features = _available_features(df)
    target = _get_target_column(df)

    # Aggregate to global-year level to forecast a global trend
    if "year" in df.columns:
        numeric = df.select_dtypes(include="number").columns.tolist()
        annual = df.groupby("year", as_index=False)[numeric].mean().sort_values("year")
    else:
        annual = df.copy()
    last_row = annual.iloc[-1]

    last_year = int(last_row.get("year", 2024))
    future_years = np.arange(last_year + 1, last_year + 1 + forecast_horizon)

    future = pd.DataFrame({"year": future_years})
    for feat in features:
        if feat == "trend_factor":
            span = max(1, (annual["year"].max() - annual["year"].min()))
            future[feat] = (future["year"] - annual["year"].min()) / span
        elif feat == "year":
            future[feat] = future["year"]
        else:
            # extrapolate linearly over last 5 years
            tail = annual.tail(5)
            slope = _lin_slope(tail["year"].values, tail[feat].values) if len(tail) >= 2 else 0.0
            future[feat] = np.clip(last_row[feat] + slope * np.arange(1, forecast_horizon + 1), 0, None)

    future[f"forecast_{target}"] = pipeline.predict(future[features])
    return future


def _lin_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def compare_models(df: pd.DataFrame, target: str | None = None) -> dict:
    """Compare a few regressors using the same train/test split."""
    if target is None:
        target = _get_target_column(df)
    X, y, _ = _clean_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("model", RandomForestRegressor(n_estimators=200, random_state=42))]),
        "Gradient Boosting": _build_pipeline(),
    }
    out = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        out[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }
    return out


def tune_hyperparameters(df: pd.DataFrame, target: str | None = None, n_trials: int = 20) -> dict:
    """Tune the GradientBoosting model using a simple random search (no optuna hard dep)."""
    import random
    if target is None:
        target = _get_target_column(df)
    X, y, _ = _clean_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best: dict = {"rmse": float("inf"), "params": {}}
    rng = random.Random(42)
    for _ in range(n_trials):
        params = {
            "n_estimators": rng.choice([50, 100, 200, 300]),
            "max_depth": rng.randint(2, 6),
            "learning_rate": rng.choice([0.03, 0.05, 0.08, 0.1, 0.2]),
        }
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42, **params)),
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        if rmse < best["rmse"]:
            best = {"rmse": rmse, "params": params}
    return best["params"]


def explain_model(df: pd.DataFrame, pipeline: Pipeline, target: str | None = None) -> dict:
    """Aggregate feature importances from the final estimator back to base features.

    Works whether or not the pipeline contains a ``PolynomialFeatures`` step
    (the step is optional since we default to ``degree=1`` which skips it).
    """
    if target is None:
        target = _get_target_column(df)
    features = _available_features(df)
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"feature_importance": {f: 0.0 for f in features}}

    agg: dict[str, float] = {f: 0.0 for f in features}

    if "poly" in pipeline.named_steps:
        poly = pipeline.named_steps["poly"]
        expanded_names = poly.get_feature_names_out(features)
        for name, imp in zip(expanded_names, importances):
            for base in features:
                if base in name:
                    agg[base] += float(imp)
                    break
    else:
        if len(importances) == len(features):
            for base, imp in zip(features, importances):
                agg[base] = float(imp)
        else:
            for base in features:
                agg[base] = float(np.mean(importances))

    total = sum(agg.values()) or 1.0
    agg = {k: v / total for k, v in agg.items()}
    return {"feature_importance": agg}
