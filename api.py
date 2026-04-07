from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from pathlib import Path

from mobile_money_project.etl import run_mobile_money_etl, build_mobile_money_forecast
from mobile_money_project.analysis import summarize_mobile_money_trends

app = FastAPI(title="Mobile Money Adoption API", version="1.0.0")

DATA_PATH = Path(__file__).parent / "data" / "sample_mobile_money_data.csv"

@app.get("/")
def read_root():
    return {"message": "Mobile Money Adoption API", "docs": "/docs"}

@app.get("/summary")
def get_summary():
    """Get global summary statistics."""
    df, summary = run_mobile_money_etl(str(DATA_PATH))
    return summary

@app.get("/forecast")
def get_forecast(horizon: int = 12):
    """Get forecast for mobile money adoption."""
    df, _ = run_mobile_money_etl(str(DATA_PATH))
    forecast_result = build_mobile_money_forecast(df, forecast_horizon=horizon)
    return {
        "metrics": forecast_result["metrics"],
        "forecast": forecast_result["forecast"].to_dict(orient="records")
    }

@app.get("/country/{country}")
def get_country_data(country: str):
    """Get data and summary for a specific country."""
    df, _ = run_mobile_money_etl(str(DATA_PATH))
    if "country" not in df.columns:
        raise HTTPException(status_code=404, detail="Country data not available")
    country_df = df[df["country"] == country]
    if country_df.empty:
        raise HTTPException(status_code=404, detail="Country not found")
    summary = summarize_mobile_money_trends(country_df)
    return {
        "summary": summary,
        "data": country_df.to_dict(orient="records")
    }

@app.get("/dashboard")
def get_dashboard():
    """Serve the generated dashboard HTML."""
    html_path = Path(__file__).parent / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not generated. Run dashboard.py first.")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


def main():
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()