"""
Walk-forward (rolling-origin) cross-validation splitter.

Reason I chose walk-forward, not random k-fold:
  Random splits leak future information into training data

Configuration: 3 folds x 90-day test windows. I assume that this is a good configuration 
because it is a good balance between having enough data to train the model and having enough data to test the model.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


HORIZON = 90
N_FOLDS = 3


def walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    horizon: int = HORIZON,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    produce (train, test) DataFrames for each fold.
   
    """
    n = len(df)
    for fold_idx in range(n_folds):
        test_end = n - fold_idx * horizon
        test_start = test_end - horizon
        if test_start <= 0:
            raise ValueError(
                f"Fold {fold_idx}: test_start={test_start} <= 0. "
                f"Reduce n_folds or horizon."
            )
        train = df.iloc[:test_start]
        test = df.iloc[test_start:test_end]
        yield train, test
