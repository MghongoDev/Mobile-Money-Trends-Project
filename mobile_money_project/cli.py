"""Command-line entry point: run the full ETL + model pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .etl import run_mobile_money_etl, build_mobile_money_forecast


def _print_summary(summary: dict) -> None:
    print("\n=== Global Trend Summary ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mobile-money", description="Run the mobile money ETL + forecast pipeline")
    parser.add_argument("--data", type=Path, default=None, help="Path to input CSV (default: bundled)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for outputs")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (years)")
    parser.add_argument("--no-api", action="store_true", help="Skip network fetch; use local CSV only")
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[etl] loading data (use_api={not args.no_api}) ...")
    df, summary = run_mobile_money_etl(path=args.data, use_api=not args.no_api)
    _print_summary(summary)

    print("\n[model] training forecast model ...")
    result = build_mobile_money_forecast(df, forecast_horizon=args.horizon)
    metrics = result["metrics"]
    print("=== Model Metrics ===")
    print(f"  target      : {metrics['target']}")
    print(f"  features    : {', '.join(metrics['features_used'])}")
    print(f"  train/test  : {metrics['train_size']} / {metrics['test_size']}")
    print(f"  MAE         : {metrics['mae']:.4f}")
    print(f"  RMSE        : {metrics['rmse']:.4f}")
    print(f"  R^2         : {metrics['r2']:.4f}")

    backtest_path = args.results_dir / "backtest_results.csv"
    forecast_path = args.results_dir / "mobile_money_forecast.csv"
    metrics_path = args.results_dir / "metrics.json"

    result["backtest"].to_csv(backtest_path, index=False)
    result["forecast"].to_csv(forecast_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    print(f"\n[out] backtest -> {backtest_path}")
    print(f"[out] forecast -> {forecast_path}")
    print(f"[out] metrics  -> {metrics_path}")
    print("[done] pipeline complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
