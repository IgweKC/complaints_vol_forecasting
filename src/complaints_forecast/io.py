"""
Load, clean, and validate data.

Firt Decision Note:
- Drop `centered_7d_mean` (leakage: it uses future values t+1..t+3).
- Reindex to a full daily DatetimeIndex so lag features are calendar-correct.
- Impute exogenous columns with forward-fill and then compute median (I made 
this decission because operational variables don't change drastically).
- Interpolate missing `complaints` only for downstream lag and rolling features;
  return an `is_imputed` mask so the training loop can exclude those rows. 
"""

# Imports useful packages 
from __future__ import annotations

#logging for debuging 
import logging
from pathlib import Path

#for calculations
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# group variables for easy handling
LEAKAGE_COLS = ["centered_7d_mean"]

''' These are other input variables besides the historical target'''
EXOG_COLS = [
    "is_weekend",
    "bank_holiday_flag",
    "staffing_level_fte",
    "backlog_days",
    "media_mentions",
    "channel_mix_index",
]

TARGET = "complaints"

def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV or Excel and return a DataFrame indexed by date."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name="daily records")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning:
    1. Drop leakage columns.
    2. Reindex to continuous daily dates.
    3. Impute exogenous variables
    4. Interpolate target for lag features and flag imputed rows.
    5. Perform Winsorisation at P01/P99 for linear models.
    """
    df = df.copy()

    # 1. Drop leakage
    for col in LEAKAGE_COLS:
        if col in df.columns:
            logger.info("Dropping leakage column: %s", col)
            df = df.drop(columns=[col])

    # 2. Reindex to full daily range
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    n_gaps = len(full_idx) - len(df)
    if n_gaps > 0:
        logger.info("Reindexing: filling %d missing calendar dates", n_gaps)
    df = df.reindex(full_idx)

    # Drop row_id — not a feature
    if "row_id" in df.columns:
        df = df.drop(columns=["row_id"])

    # 3. Calendar features that can be derived for any date
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    # 4. Impute exog: forward-fill then median
    for col in EXOG_COLS:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                logger.info("Imputing %s: %d missing values (ffill+median)", col, n_miss)
                df[col] = df[col].ffill()
                df[col] = df[col].fillna(df[col].median())

    # 5. Target: mark imputed, then interpolate for lag/rolling feature construction
    df["is_imputed"] = df[TARGET].isna().astype(int)
    n_target_miss = df["is_imputed"].sum()
    if n_target_miss > 0:
        logger.info("Interpolating %d missing target values (flagged as is_imputed)", n_target_miss)
        df[TARGET] = df[TARGET].interpolate(method="linear")

    # 6. Winsorised copy for linear models
    p01, p99 = df[TARGET].quantile(0.01), df[TARGET].quantile(0.99)
    df["complaints_winsorised"] = df[TARGET].clip(lower=p01, upper=p99)

    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """ load + clean"""
    return clean(load_raw(path))
