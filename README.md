#  Mobile Money Adoption Analysis 

A project that analyses global mobile-money and financial-institution
account adoption, trains a Gradient Boosting forecast model and serves the
results through an interactive **Streamlit** dashboard.

> Previously this repo mixed a static HTML/JS dashboard, Jupyter notebooks,
> YAML CI workflows and a FastAPI server with the Python core. It has been
> revamped into a single, cohesive Python-only application.
---

## Features

- **ETL** pipeline that fetches live data from [Our World in Data](https://ourworldindata.org)
  and falls back to a bundled CSV when offline.
- **Feature engineering** – growth rates, account ratios, digital-inclusion
  index, time-trend factor.
- **Forecasting** with a `PolynomialFeatures + StandardScaler + GradientBoostingRegressor`
  pipeline, plus model comparison, hyper-parameter search and built-in feature
  importance.
- **Interactive Streamlit dashboard** (`app.py`) with KPIs, Plotly charts,
  country drill-down, model inspection, forecast plotting and CSV downloads.
- **CLI** entry point that trains the model and writes CSV/JSON outputs.
- **Tests** written in pure Python (`pytest`).

---

## Project structure (all source files are `.py`)

```
├── app.py                       # Streamlit dashboard (main UI)
├── main.py                      # `python main.py` -> CLI pipeline
├── run_analysis.py              # alias to CLI for backwards compat
├── pyproject.toml               # PEP 621 project metadata
├── mobile_money_project/
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point
│   ├── data.py                  # fetch/load data
│   ├── preprocessing.py         # cleaning & feature engineering
│   ├── analysis.py              # summaries & top-N country analytics
│   ├── modeling.py              # training, forecasting, comparison, SHAP-style importance
│   └── etl.py                   # orchestrates extract/transform/load
├── tests/
│   ├── test_etl.py
│   └── test_modeling.py
├── data/
│   └── sample_mobile_money_data.csv   # bundled fallback dataset
└── results/                     # output CSVs/JSON written by CLI
```

Data files (`data/*.csv`, `results/*.csv`) remain CSV because they are
*data*, not source code – but every piece of executable logic is Python.

---

## Quick start

```bash
# 1. create a virtualenv and install the package
pip install -e ".[dev]"

# 2. run the interactive dashboard
streamlit run app.py

# 3. or run the CLI pipeline to regenerate results/*.csv
python main.py --horizon 12

# 4. run the tests
pytest
```

### CLI options

```
python main.py --help
```

| Flag              | Description                                            |
|-------------------|--------------------------------------------------------|
| `--data PATH`     | Input CSV path (defaults to bundled sample)            |
| `--results-dir D` | Output directory (default `results/`)                  |
| `--horizon N`     | Forecast horizon in years (default 12)                 |
| `--no-api`        | Skip the live Our World in Data fetch (offline mode)   |

### Dashboard

Launch with `streamlit run app.py` and open <http://localhost:8501>.
The sidebar lets you:

- toggle live API fetches,
- choose the forecast horizon,
- switch between the **Global trends**, **Country explorer**, **Model**,
  **Forecast** and **Raw data** tabs.

All charts are Plotly figures rendered directly from pandas DataFrames – no
JavaScript templates, no HTML boilerplate.

---

## Model summary

The default model is a scikit-learn `Pipeline`:

1. `PolynomialFeatures(degree=2)`
2. `StandardScaler`
3. `GradientBoostingRegressor`

With the bundled dataset it typically achieves **R² ≈ 0.99** on a time-based
holdout, predicting `only_mobile_money_account` from the engineered features
(`trend_factor`, `digital_inclusion_index`, `account_ratio`, growth rates, …).

---

## Data source

Our World in Data – *Share of adults with a bank account, financial
institution account or mobile money account*
([owid.cloud](https://ourworldindata.org/grapher/share-adults-bank-account-financial-institution-mobile-money)).

---

## Testing

```bash
pytest -q
```

Tests cover ETL extraction/transformation and model training/forecasting.
