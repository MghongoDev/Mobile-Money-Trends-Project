# Mobile Money Adoption Analysis

This is a machine learning project analyzing trends in mobile money and financial institution account adoption using real data from **Our World in Data**.

## Key Findings

### Adoption Trends (2014-2022)

**Mobile Money** shows explosive growth globally:
- **2014**: 6.08% average adoption
- **2022**: 18.89% average adoption  
- **Growth**: +210.6% over 8 years

**Financial Institution Accounts** experienced modest decline:
- **2014**: 37.69% average adoption
- **2022**: 32.05% average adoption
- **Change**: -15.0%

This suggests mobile money is expanding the financial inclusion frontier, particularly in regions underserved by traditional banking.

### Top Countries by Mobile Money Adoption (2022)

1. **Eswatini**: 56.7%
2. **Lesotho**: 45.9%
3. **Botswana**: 36.6%
4. **Democratic Republic of Congo**: 22.7%
5. **Madagascar**: 19.0%

Most adoption leaders are in sub-Saharan Africa, where mobile networks enabled rapid financial inclusion.

### Model Insights

The gradient boosting regression model trained on 211 samples achieved excellent predictive accuracy on 53 test cases:

| Metric | Value |
|--------|-------|
| **R-squared** | 0.9880 |
| **MAE** | 0.5063 |
| **RMSE** | 1.0881 |

High R² indicates mobile money adoption patterns are highly predictable using historical trends, digital inclusion metrics, and account ratios.

### Dataset Overview

- **Data Source**: Our World in Data
- **Total Records**: 264
- **Countries**: 103
- **Time Period**: 2014-2022

## Data Source

The project automatically fetches adoption data from:
- **API**: ourworldindata.org (share-adults-bank-account-financial-institution-mobile-money)
- **Metrics**: 
  - Mobile money account adoption share (%)
  - Financial institution account adoption share (%)
  - Dual account ownership

## Project Structure

- `mobile_money_project/`
  - `data.py`: API fetching from Our World in Data + caching
  - `preprocessing.py`: feature engineering and data cleaning
  - `analysis.py`: trend summaries and statistics
  - `modeling.py`: gradient boosting regression forecasting
- `run_analysis.py`: main analysis pipeline
- `notebooks/mobile_money_trends.ipynb`: interactive exploration
- `data/`: auto-downloaded datasets
- `results/`: forecasts and model metrics

## Getting Started

1. Install dependencies:

```bash
uv sync
```

2. Run the pipeline:

```bash
python main.py
```

Output:
- `results/backtest_results.csv` - model validation metrics
- `results/mobile_money_forecast.csv` - 12-year forward forecast
- `data/sample_mobile_money_data.csv` - cached API data

3. Open the notebook:

```bash
notebooks/mobile_money_trends.ipynb
```

4. Generate the interactive dashboard HTML:

```bash
python dashboard.py
```

This command builds `dashboard.html` in the project root and opens it in your browser.

## Interactive Dashboard

The dashboard is generated from Python and HTML only, using the same ETL pipeline as the analysis script.
- Explore mobile money vs financial account share trends
- Inspect digital inclusion and account gaps
- View forecast metrics and backtest results
- Filter by country and download filtered CSV output

## Key Features

- Real-world adoption data from 100+ countries
- Automatic API caching to disk
- Feature engineering (growth rates, ratios, digital inclusion indices)
- Gradient boosting model with 98.8% test R²
- 12-year forward forecasting capability
- Interactive visualizations in Jupyter notebook
