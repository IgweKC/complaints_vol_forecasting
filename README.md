# complaints_vol_forecasting
  **Objective**: Build a forecasting approach that predicts the daily num 90 days after the final date in the dataset (2025-12-31 to 2026-03-31).



## Approach



1. **Data cleaning**: I Dropped `centered_7d_mean` (leakage — uses future values t+1..t+3). Reindexed to full daily calendar (43 gap-days filled). Impute missing exogenous variables via forward-fill + median. Interpolate 53 missing target values for lag features only; flag and exclude from training loss.

2. **Feature engineering**: I generated 19 strictly causal features  calendar (dow, month, week-of-year, day-of-month, trend), causal lags (1, 7, 14, 28 days), trailing rolling mean/std (7d, 28d), and exogenous passthrough (staffing, backlog, media, channel mix, bank holidays).

3. **Model**: 
- Base Model: Seasonal Naive (m=7)  
- SARIMAX(1,1,1)(1,1,0,7) with exogenous variables
- LightGBM with quantile loss (q=0.1/0.5/0.9 for point forecast + 80% Prediction Interval).

4. **Evaluation**: Walk-forward backtest, 3 folds × 90-day test windows. Metrics: MAE (primary), RMSE, MASE, sMAPE, pinball loss, empirical PI coverage.

5. **Result**: LightGBM wins with mean MAE ≈ 25.1 complaints/day across folds, beating SARIMAX (25.9) and Seasonal Naive (29.5).



## How to Run



```bash
# Clone and set up
git clone https://github.com/IgweKC/complaints_vol_forecasting.git

cd complaints_vol_forecasting

python -m venv venv

# Windows:

.\venv\Scripts\activate

# macOS/Linux:

# source venv/bin/activate

pip install -r requirements.txt



# Run the full pipeline (backtest + 90-day forecast)

python scripts/run_forecast.py


```



**Outputs** are written to `reports/`:

- `forecast_90d.csv` : 90-day daily forecast with 80% prediction intervals

- `backtest_metrics.csv` : per-fold per-model evaluation metrics

- `figures/` : backtest fan chart, residual ACF, day-of-week bias, forecast plot



Step-by-step naration: `notebooks/forecast.ipynb` walks through EDA, feature engineering, backtest, and forecast with comments.




Stucture:
complaints_vol_forecasting
    - data
        - daily_records.csv          # source data
    - notebook
        - 
    - report (automatically generated)
    - Scripts
        - run_forecast.py             # single entry point 
    - src/complaints_forecast
        -__init__.py
        - io.py                       # load, clean, reindex, impute, leakage drop
        - features.py                 # calendar, causal lags, trailing rolling stats      
        - splits.py                   # rolling-origin walk-forward splitter
        - metrics.py                  # MAE, RMSE, MASE, pinball, coverage
        - models.py                   # SeasonalNaive, SARIMAX, LightGBM
        - forecast.py                 # future frame construction, 90-day 
    - requirements.txt
    - README.md
