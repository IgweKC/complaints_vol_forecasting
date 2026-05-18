"""
Produce the 90-day forward forecast under stated exogenous assumptions.

Exogenous strategy for the forecast horizon:
- is_weekend, bank_holiday_flag: derived from calendar (known future).
- staffing_level_fte, backlog_days, media_mentions, channel_mix_index:
  held at trailing 28-day median of the training data (base scenario).
  In production, ops should plug in their own forward assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays as hol

from complaints_forecast.io import TARGET, EXOG_COLS
from complaints_forecast.features import (
    add_calendar_features,
    LAG_DAYS,
    ROLLING_WINDOWS,
)


def build_future_frame(
    hist_df: pd.DataFrame,
    horizon: int = 90,
    country: str = "GB",
) -> pd.DataFrame:
    """
    Construct a DataFrame for the next ``horizon`` days after the end of
    ``hist_df``, filled with deterministic calendar features and base-scenario
    exogenous values.
    """
    last_date = hist_df.index.max()
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D", name="date")
    future = pd.DataFrame(index=future_idx)

    # Calendar-derived (known future)
    future["is_weekend"] = future.index.dayofweek.isin([5, 6]).astype(int)

    uk_holidays = hol.country_holidays(country, years=[d.year for d in future.index])
    future["bank_holiday_flag"] = future.index.map(lambda d: int(d in uk_holidays))

    # Operational exog: trailing 28-day median as base scenario
    trailing = hist_df.tail(28)
    for col in ["staffing_level_fte", "backlog_days", "media_mentions", "channel_mix_index"]:
        future[col] = trailing[col].median()

    future["is_imputed"] = 0
    return future


def build_future_features_for_lgbm(
    hist_df: pd.DataFrame,
    future_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    For LightGBM (direct multi-step with lag features), we need to fill
    lag and rolling columns from the history.  Because this is a direct
    model, we use the *actual* historical values for lags (no recursion).

    For steps beyond the longest lag (28), the lag features point into
    the forecast horizon itself — here we fall back to the last available
    historical value (a conservative assumption).
    """
    combined = pd.concat([hist_df[[TARGET]], future_frame[[]]]).copy()

    for lag in LAG_DAYS:
        combined[f"lag_{lag}"] = combined[TARGET].shift(lag) if TARGET in combined.columns else np.nan

    # For future rows, fill any remaining NaN lags from the historical tail
    for lag in LAG_DAYS:
        col = f"lag_{lag}"
        if combined[col].isna().any():
            combined[col] = combined[col].ffill()

    for w in ROLLING_WINDOWS:
        shifted = combined[TARGET].shift(1) if TARGET in combined.columns else pd.Series(dtype=float)
        combined[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        combined[f"roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0)

    future_feats = future_frame.copy()
    future_feats = add_calendar_features(future_feats)

    # Use days_since_start relative to the original history start
    future_feats["days_since_start"] = (future_feats.index - hist_df.index.min()).days

    for col in [f"lag_{l}" for l in LAG_DAYS] + \
               [f"roll_mean_{w}" for w in ROLLING_WINDOWS] + \
               [f"roll_std_{w}" for w in ROLLING_WINDOWS]:
        future_feats[col] = combined.loc[future_frame.index, col].values

    return future_feats
