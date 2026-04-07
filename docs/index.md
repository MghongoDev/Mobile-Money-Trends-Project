# Mobile Money Adoption Project Documentation

## Architecture

The project follows an ETL (Extract, Transform, Load) pipeline:

1. **Extract**: Fetch data from Our World in Data API and optionally World Bank API
2. **Transform**: Clean, preprocess, and engineer features
3. **Load**: Prepare data for analysis and modeling

## API Endpoints

- `GET /`: Root endpoint
- `GET /summary`: Global summary statistics
- `GET /forecast?horizon=12`: Forecast data
- `GET /country/{country}`: Country-specific data
- `GET /dashboard`: Generated HTML dashboard

## Model Details

- **Algorithm**: Gradient Boosting Regressor with polynomial features
- **Features**: Trend factor, internet penetration, smartphone penetration, etc.
- **Evaluation**: MAE, RMSE, R²

## Deployment

Run the API locally with uvicorn:

```bash
uvicorn api:app --reload
```