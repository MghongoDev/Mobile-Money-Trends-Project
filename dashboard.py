from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import pandas as pd

from mobile_money_project.analysis import summarize_mobile_money_trends
from mobile_money_project.etl import build_mobile_money_forecast, run_mobile_money_etl

DATA_PATH = Path(__file__).parent / "data" / "sample_mobile_money_data.csv"
HTML_PATH = Path(__file__).parent / "dashboard.html"


def get_country_options(df: pd.DataFrame) -> list[str]:
    if "country" not in df.columns:
        return ["All countries"]
    countries = sorted(df["country"].dropna().unique().tolist())
    return ["All countries"] + countries


def aggregate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if "year" not in df.columns:
        return df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        return df.copy()

    return df.groupby("year", as_index=False)[numeric_cols].mean().reset_index(drop=True)


def build_country_data(df: pd.DataFrame) -> dict[str, dict]:
    result: dict[str, dict] = {}
    countries = get_country_options(df)

    for country in countries:
        if country == "All countries":
            subset = df.copy()
            records = aggregate_by_year(df)
        else:
            subset = df[df["country"] == country].copy()
            records = subset.sort_values("year") if "year" in subset.columns else subset

        records_json = records.fillna("").to_dict(orient="records")
        summary = summarize_mobile_money_trends(subset if not subset.empty else df)

        result[country] = {
            "records": records_json,
            "summary": summary,
        }

    return result


def safe_json_dumps(data: object) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)


def build_dashboard_html(
    country_data: dict[str, dict],
    forecast_data: list[dict],
    backtest_data: list[dict],
    metrics: dict,
    target_name: str,
) -> str:
    countries = list(country_data.keys())
    default_country = "All countries"
    template = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mobile Money Adoption Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 16px; background: #f5f7fb; color: #202124; }
        .topbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; }
        .topbar h1 { margin: 0; font-size: 2rem; }
        .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
        .card strong { display: block; margin-bottom: 8px; color: #4b5563; }
        .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; margin-bottom: 24px; }
        .table-section { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 24px; }
        table { border-collapse: collapse; width: 100%; margin-top: 12px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #e5e7eb; }
        th { background: #f9fafb; color: #111827; }
        select, button { padding: 10px 12px; border-radius: 8px; border: 1px solid #d1d5db; font-size: 1rem; }
        .metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-top: 16px; }
        .metric-box { background: #111827; color: white; padding: 14px 16px; border-radius: 10px; }
        .metric-box span { display: block; font-size: 1.6rem; margin-top: 6px; }
        .note { color: #374151; margin-top: 8px; }
        .download-link { margin-top: 16px; display: inline-block; }
    </style>
</head>
<body>
    <div class="topbar">
        <div>
            <h1>Mobile Money Adoption Dashboard</h1>
            <p>Interactive HTML dashboard generated from the project ETL pipeline.</p>
        </div>
        <div>
            <label for="countrySelect"><strong>Country / dataset:</strong></label>
            <select id="countrySelect"></select>
        </div>
    </div>

    <div class="card-grid" id="summaryCards"></div>

    <div class="charts">
        <div class="card">
            <h3>Share of adults with accounts</h3>
            <canvas id="shareChart"></canvas>
        </div>
        <div class="card">
            <h3>Account gap over time</h3>
            <canvas id="gapChart"></canvas>
        </div>
    </div>

    <div class="charts">
        <div class="card">
            <h3>Growth percentage trends</h3>
            <canvas id="growthChart"></canvas>
        </div>
        <div class="card">
            <h3>Digital inclusion index</h3>
            <canvas id="inclusionChart"></canvas>
        </div>
    </div>

    <div class="table-section">
        <h2>Forecast metrics</h2>
        <div class="metrics-row">
            <div class="metric-box"><strong>Target</strong><span>__TARGET_NAME__</span></div>
            <div class="metric-box"><strong>MAE</strong><span>__MAE__</span></div>
            <div class="metric-box"><strong>RMSE</strong><span>__RMSE__</span></div>
            <div class="metric-box"><strong>R²</strong><span>__R2__</span></div>
        </div>
        <p class="note">Forecast and backtest are built from the full prepared dataset.</p>
    </div>

    <div class="table-section">
        <h2>Forecast output</h2>
        <div id="forecastTable"></div>
    </div>

    <div class="table-section">
        <h2>Backtest results</h2>
        <div id="backtestTable"></div>
    </div>

    <div class="table-section">
        <h2>Raw data preview</h2>
        <div id="rawTable"></div>
    </div>

    <div class="table-section">
        <button id="downloadCsv">Download current country data as CSV</button>
    </div>

    <script>
        const countryData = __COUNTRY_DATA__;
        const forecastData = __FORECAST_DATA__;
        const backtestData = __BACKTEST_DATA__;
        const countries = __COUNTRIES__;
        const defaultCountry = "__DEFAULT_COUNTRY__";

        const countrySelect = document.getElementById("countrySelect");
        const summaryCards = document.getElementById("summaryCards");
        const forecastTable = document.getElementById("forecastTable");
        const backtestTable = document.getElementById("backtestTable");
        const rawTable = document.getElementById("rawTable");
        const downloadCsv = document.getElementById("downloadCsv");

        const shareChartCtx = document.getElementById("shareChart");
        const gapChartCtx = document.getElementById("gapChart");
        const growthChartCtx = document.getElementById("growthChart");
        const inclusionChartCtx = document.getElementById("inclusionChart");

        let shareChart, gapChart, growthChart, inclusionChart;

        function createChart(ctx, labels, datasets, title) {
            return new Chart(ctx, {
                type: "line",
                data: { labels, datasets },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: "top" },
                        title: { display: true, text: title }
                    },
                    scales: {
                        x: { title: { display: true, text: "Year" } },
                        y: { beginAtZero: true }
                    }
                }
            });
        }

        function buildTableHtml(rows) {
            if (!rows || rows.length === 0) {
                return "<p>No data available.</p>";
            }
            const columns = Object.keys(rows[0]);
            let html = "<table><thead><tr>";
            columns.forEach(col => html += `<th>${col}</th>`);
            html += "</tr></thead><tbody>";
            rows.forEach(row => {
                html += "<tr>";
                columns.forEach(col => html += `<td>${row[col] ?? ""}</td>`);
                html += "</tr>";
            });
            html += "</tbody></table>";
            return html;
        }

        function formatNumber(value) {
            if (value === null || value === undefined || value === "") {
                return "N/A";
            }
            return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
        }

        function renderSummary(summary) {
            summaryCards.innerHTML = "";
            const metrics = [
                ["Periods", summary.time_periods],
                ["Mobile trend slope", summary.mobile_trend_slope],
                ["Financial trend slope", summary.financial_trend_slope],
                ["Countries", summary.countries],
                ["Final mobile share", summary.final_mobile_share],
                ["Final financial share", summary.final_financial_share],
                ["Latest account gap", summary.latest_account_gap],
                ["Year range", summary.year_range],
            ];
            metrics.forEach(([label, value]) => {
                if (value === undefined) return;
                const card = document.createElement("div");
                card.className = "card";
                card.innerHTML = `<strong>${label}</strong><span>${formatNumber(value)}</span>`;
                summaryCards.appendChild(card);
            });
        }

        function getNumericValue(row, key) {
            const value = row[key];
            return value === null || value === undefined || value === "" ? NaN : Number(value);
        }

        function prepareChartData(records, key) {
            const labels = [];
            const values = [];
            records.forEach(row => {
                const label = row.year ?? row.date ?? labels.length + 1;
                labels.push(label);
                values.push(getNumericValue(row, key));
            });
            return { labels, values };
        }

        function updateCharts(records) {
            const shareDatasets = [];
            const shareKeys = ["mobile_money_share", "financial_institution_share"];
            const shareColors = ["#2563eb", "#10b981"];
            shareKeys.forEach((key, index) => {
                const data = prepareChartData(records, key);
                shareDatasets.push({
                    label: key,
                    data: data.values,
                    borderColor: shareColors[index],
                    backgroundColor: shareColors[index],
                    tension: 0.3,
                });
            });
            const labels = records.map(row => row.year ?? row.date ?? "");

            if (shareChart) shareChart.destroy();
            shareChart = createChart(shareChartCtx, labels, shareDatasets, "Share of mobile money and financial accounts");

            const gapData = prepareChartData(records, "account_gap");
            if (gapChart) gapChart.destroy();
            gapChart = createChart(gapChartCtx, gapData.labels, [{ label: "Account gap", data: gapData.values, borderColor: "#ef4444", backgroundColor: "#ef4444", tension: 0.3 }], "Account gap over time");

            const growthDatasets = [];
            const growthKeys = ["mobile_growth_pct", "financial_growth_pct"];
            const growthColors = ["#f97316", "#8b5cf6"];
            growthKeys.forEach((key, index) => {
                const data = prepareChartData(records, key);
                growthDatasets.push({ label: key, data: data.values, borderColor: growthColors[index], backgroundColor: growthColors[index], tension: 0.3 });
            });
            if (growthChart) growthChart.destroy();
            growthChart = createChart(growthChartCtx, labels, growthDatasets, "Growth percentage trends");

            const inclusionData = prepareChartData(records, "digital_inclusion_index");
            if (inclusionChart) inclusionChart.destroy();
            inclusionChart = createChart(inclusionChartCtx, inclusionData.labels, [{ label: "Digital inclusion index", data: inclusionData.values, borderColor: "#14b8a6", backgroundColor: "#14b8a6", tension: 0.3 }], "Digital inclusion index over time");
        }

        function updateTables(records) {
            rawTable.innerHTML = buildTableHtml(records.slice(0, 50));
        }

        function renderForecastTable() {
            forecastTable.innerHTML = buildTableHtml(forecastData);
        }

        function renderBacktestTable() {
            backtestTable.innerHTML = buildTableHtml(backtestData);
        }

        function populateCountrySelect() {
            countrySelect.innerHTML = countries.map(country => `<option value="${country}">${country}</option>`).join("");
            countrySelect.value = defaultCountry;
        }

        function updateDashboard(country) {
            const countryInfo = countryData[country] || countryData[defaultCountry];
            renderSummary(countryInfo.summary);
            updateCharts(countryInfo.records);
            updateTables(countryInfo.records);
        }

        function downloadCsvFromRecords(records) {
            if (!records.length) return;
            const columns = Object.keys(records[0]);
            const csvRows = [columns.join(",")];
            records.forEach(row => {
                const values = columns.map(col => {
                    const item = row[col];
                    return typeof item === "string" ? `"${item.replace(/"/g, '""')}"` : item;
                });
                csvRows.push(values.join(","));
            });
            const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `mobile_money_data_${countrySelect.value.replace(/\W+/g, "_")}.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }

        countrySelect.addEventListener("change", () => updateDashboard(countrySelect.value));
        downloadCsv.addEventListener("click", () => downloadCsvFromRecords(countryData[countrySelect.value].records));

        populateCountrySelect();
        renderForecastTable();
        renderBacktestTable();
        updateDashboard(defaultCountry);
    </script>
</body>
</html>
"""
    html = template.replace("__COUNTRY_DATA__", safe_json_dumps(country_data))
    html = html.replace("__FORECAST_DATA__", safe_json_dumps(forecast_data))
    html = html.replace("__BACKTEST_DATA__", safe_json_dumps(backtest_data))
    html = html.replace("__COUNTRIES__", safe_json_dumps(countries))
    html = html.replace("__DEFAULT_COUNTRY__", default_country)
    html = html.replace("__TARGET_NAME__", target_name)
    html = html.replace("__MAE__", f"{metrics.get('mae', 0):.3f}")
    html = html.replace("__RMSE__", f"{metrics.get('rmse', 0):.3f}")
    html = html.replace("__R2__", f"{metrics.get('r2', 0):.3f}")
    return html


def main() -> None:
    df, _ = run_mobile_money_etl(str(DATA_PATH))
    country_data = build_country_data(df)

    forecast_result = build_mobile_money_forecast(df, forecast_horizon=12)
    forecast_df = forecast_result["forecast"].fillna("")
    backtest_df = forecast_result["backtest"].fillna("")
    metrics = forecast_result["metrics"]
    target_name = metrics.get("target", "forecast")

    html = build_dashboard_html(
        country_data=country_data,
        forecast_data=forecast_df.to_dict(orient="records"),
        backtest_data=backtest_df.to_dict(orient="records"),
        metrics=metrics,
        target_name=target_name,
    )

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {HTML_PATH}")
    webbrowser.open_new_tab(HTML_PATH.as_uri())


if __name__ == "__main__":
    main()
