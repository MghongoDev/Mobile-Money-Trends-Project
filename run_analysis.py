from pathlib import Path

import pandas as pd

from mobile_money_project.analysis import summarize_mobile_money_trends
from mobile_money_project.data import load_mobile_money_data
from mobile_money_project.modeling import forecast_mobile_money, train_mobile_money_forecast
from mobile_money_project.preprocessing import prepare_mobile_money_data


def main() -> None:
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "sample_mobile_money_data.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load data from API or fallback
    df = load_mobile_money_data(str(data_path))
    
    print(f"\n[INFO] Raw dataset shape: {df.shape}")
    print(f"[INFO] Columns: {', '.join(df.columns.tolist())}")
    print(f"\n[INFO] First few rows:")
    print(df.head())
    
    # Prepare data
    df = prepare_mobile_money_data(df)

    print(f"\n[INFO] Prepared dataset shape: {df.shape}")

    # Trend summary
    try:
        trend_summary = summarize_mobile_money_trends(df)
        print("\n[INFO] Trend Summary:")
        for name, value in trend_summary.items():
            print(f"  {name}: {value}")
    except Exception as e:
        print(f"[WARNING] Could not generate trend summary: {e}")

    # Train model
    try:
        pipeline, backtest, y_test, metrics = train_mobile_money_forecast(df)
        print("\n[INFO] Model Training Results:")
        for metric, value in metrics.items():
            if metric not in ["features_used"]:
                print(f"  {metric}: {value}")
        
        backtest_path = results_dir / "backtest_results.csv"
        backtest.to_csv(backtest_path, index=False)
        print(f"\n[INFO] Backtest results saved to {backtest_path}")

        # Generate forecast
        forecast = forecast_mobile_money(df, pipeline, forecast_horizon=12)
        forecast_path = results_dir / "mobile_money_forecast.csv"
        forecast.to_csv(forecast_path, index=False)
        print(f"[INFO] Forecast saved to {forecast_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not train model: {e}")
        print("[INFO] This may be due to data format or missing required columns")

    print("\n[INFO] Mobile money adoption analysis complete.")


if __name__ == "__main__":
    main()
