"""
Forecast evaluation metrics.

Primary: MAE .
Secondary: RMSE .
Scale-free: MASE vs. seasonal-naive m=7 (gate: MASE < 1).
Probabilistic: pinball loss and empirical PI coverage.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 7) -> float:
    """
    Mean Absolute Scaled Error relative to a seasonal-naive baseline
    with period *m*.  MASE < 1 means the model beats naive.
    """
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.inf
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile / pinball loss."""
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)))


def pi_coverage(y_true: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray) -> float:
    """Empirical prediction-interval coverage."""
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    y_lo: np.ndarray | None = None,
    y_hi: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all metrics in one call; PI metrics only if bounds supplied."""
    results = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, m=7),
    }
    if y_lo is not None and y_hi is not None:
        results["pinball_10"] = pinball_loss(y_true, y_lo, 0.1)
        results["pinball_90"] = pinball_loss(y_true, y_hi, 0.9)
        results["coverage_80"] = pi_coverage(y_true, y_lo, y_hi)
    return results
