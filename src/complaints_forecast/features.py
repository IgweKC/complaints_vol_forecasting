"""
Feature engineering for the complaints forecasting models.
- Calendar: day-of-week, month, week-of-year, day-of-month, year trend.
- Lags: complaints at t-1, t-7, t-14, t-28.
- Trailing and rolling statistics
- Exogenous variable
"""

from __future__ import annotations
import pandas as pd

from complaints_forecast.io import EXOG_COLS, TARGET

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features."""
    df = df.copy()
    idx = df.index
    df["dow"] = idx.dayofweek          # 0=Mon .. 6=Sun
    df["month"] = idx.month
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["day_of_month"] = idx.day
    df["days_since_start"] = (idx - idx.min()).days
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Uses only past values of the target."""
    df = df.copy()
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing rolling mean/std — window ends at t-1 (shift(1))."""
    df = df.copy()
    for w in ROLLING_WINDOWS:
        rolled = df[TARGET].shift(1).rolling(window=w, min_periods=w)
        df[f"roll_mean_{w}"] = rolled.mean()
        df[f"roll_std_{w}"] = rolled.std()
    return df


def build_feature_matrix(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters:
    df : DataFrame with DatetimeIndex (include `complaints`)
    drop_na : if True, drop rows where lag and rolling features are NaN
   
    Returns:
    DataFrame with all features + target + is_imputed mask.
    """
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    if drop_na:
        df = df.dropna(subset=[f"lag_{LAG_DAYS[-1]}", f"roll_mean_{ROLLING_WINDOWS[-1]}"])

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to be used as model inputs (X)."""
    non_features = {
        TARGET,
        "complaints_winsorised",
        "is_imputed",
        "row_id",
        "centered_7d_mean",
    }
    return [c for c in df.columns if c not in non_features]
