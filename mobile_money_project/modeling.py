from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import optuna
import shap


def _build_forecast_pipeline(degree: int = 2) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)),
        ]
    )


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Dynamically identify available feature columns."""
    available_features = []
    
    potential_features = [
        "trend_factor",
        "internet_penetration",
        "smartphone_penetration",
        "policy_support_index",
        "digital_inclusion_index",
        "account_ratio",
    ]
    
    for feat in potential_features:
        if feat in df.columns:
            available_features.append(feat)
    
    # If we don't have enough features, add year if available
    if len(available_features) < 3 and "year" in df.columns:
        available_features.append("year")
    
    return available_features if available_features else ["trend_factor"]


def _get_target_column(df: pd.DataFrame) -> str:
    """Dynamically identify target column for modeling."""
    mobile_cols = [col for col in df.columns if "mobile" in col.lower()]
    return mobile_cols[0] if mobile_cols else "account_ratio"


def train_mobile_money_forecast(
    df: pd.DataFrame,
    target: str | None = None,
    test_size: float = 0.2,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict]:
    """Train a regression forecast model for mobile money adoption."""
    if target is None:
        target = _get_target_column(df)
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    
    features = _get_feature_columns(df)
    if not features:
        raise ValueError("No suitable feature columns found in data")
    
    # Remove rows with missing values in features or target
    df_clean = df[features + [target]].dropna()
    
    if len(df_clean) < 10:
        raise ValueError(f"Not enough data points after cleaning: {len(df_clean)}")
    
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    pipeline = _build_forecast_pipeline(degree=2)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(pipeline.score(X_test, y_test)),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features_used": features,
        "target": target,
    }

    results = X_test.copy()
    results["actual"] = y_test.values
    results["prediction"] = y_pred

    return pipeline, results, y_test, metrics


def forecast_mobile_money(
    df: pd.DataFrame,
    pipeline: Pipeline,
    forecast_horizon: int = 12,
) -> pd.DataFrame:
    """Produce a future forecast for mobile money adoption."""
    last = df.iloc[-1].copy()
    
    features = _get_feature_columns(df)
    
    # Create future forecast dataframe
    months = np.arange(1, forecast_horizon + 1)
    future_data = {}
    
    for feat in features:
        if feat == "trend_factor":
            future_data[feat] = last["trend_factor"] + months / 12
        elif "internet" in feat.lower():
            future_data[feat] = np.clip(last.get(feat, 0.5) + 0.005 * months, 0, 1)
        elif "smartphone" in feat.lower():
            future_data[feat] = np.clip(last.get(feat, 0.4) + 0.006 * months, 0, 1)
        elif "policy" in feat.lower():
            future_data[feat] = np.clip(last.get(feat, 50) + 0.4 * months, 0, 100)
        elif "digital" in feat.lower():
            future_data[feat] = np.clip(last.get(feat, 0.5) + 0.003 * months, 0, 1)
        elif "ratio" in feat.lower():
            future_data[feat] = last.get(feat, 0.5)
        elif "year" in feat.lower():
            future_data[feat] = last.get("year", 2024) + (months // 12)
        else:
            future_data[feat] = last.get(feat, 0.5)
    
    future = pd.DataFrame(future_data)
    
    # Add date info if available
    if "year" in df.columns:
        future["year"] = last.get("year", 2024) + (months // 12)
    if "country" in df.columns:
        future["country"] = last.get("country", "Global")
    
    target = _get_target_column(df)
    future[f"forecast_{target}"] = pipeline.predict(future[features]).round().astype(int)
    
    return future


def tune_hyperparameters(df: pd.DataFrame, target: str | None = None) -> dict:
    """Tune hyperparameters using Optuna."""
    if target is None:
        target = _get_target_column(df)
    
    features = _get_feature_columns(df)
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    return study.best_params


def compare_models(df: pd.DataFrame, target: str | None = None) -> dict:
    """Compare different models."""
    if target is None:
        target = _get_target_column(df)
    
    features = _get_feature_columns(df)
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': model.score(X_test, y_test)
        }
    
    return results


def explain_model(df: pd.DataFrame, pipeline: Pipeline, target: str | None = None) -> dict:
    """Generate SHAP explanations."""
    if target is None:
        target = _get_target_column(df)
    
    features = _get_feature_columns(df)
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    
    # Get the model from pipeline
    model = pipeline.named_steps['model']
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Feature importance
    feature_importance = dict(zip(features, np.abs(shap_values).mean(axis=0)))
    
    return {
        'feature_importance': feature_importance,
        'shap_values': shap_values.tolist(),
        'feature_names': features
    }
