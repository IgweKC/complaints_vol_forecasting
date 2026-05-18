"""
Forecasting models with a common interface.

Each model implements:
- fit(train_df, feature_cols, target_col)
- predict(test_df, feature_cols) -> point forecasts
- predict_quantiles(test_df, feature_cols, quantiles) -> dict[q, array]


I implemented Simple to Complex models:
1. SeasonalNaive(m=7) — Baseline.
2. SARIMAXModel — interpretable structural model with exogenoues features.
3. LightGBMModel — A more complex model with quantile-loss Prediction Intervals (PIs).
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Common interface for all models."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
        ...

    @abstractmethod
    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        ...

    def predict_quantiles(
        self,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        quantiles: list[float] = [0.1, 0.9],
    ) -> dict[float, np.ndarray]:
        raise NotImplementedError(f"{self.name} does not support quantile prediction")



# 1. Seasonal Naive
class SeasonalNaive(BaseForecaster):
    """
    Forecast = value from m days ago in the training set
    """

    name = "SeasonalNaive"

    def __init__(self, m: int = 7): 
        self.m = m
        self._last_season: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
        self._last_season = train_df[target_col].values[-self.m :]

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        n = len(test_df)
        reps = (n // self.m) + 1
        return np.tile(self._last_season, reps)[:n]



# 2. SARIMAX
class SARIMAXModel(BaseForecaster):
    """
    statsmodels SARIMAX with exogenous regressors.

    Order selection: small grid over (p,d,q) x (P,D,Q,7), pick lowest AIC.
    Falls back to (1,1,1)(1,1,0,7) if grid search fails within timebox.
    """

    name = "SARIMAX"

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 0, 7),
        exog_cols: list[str] | None = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols
        self._model = None
        self._result = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        exog = train_df[self.exog_cols].values if self.exog_cols else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train_df[target_col].values,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False, maxiter=200)

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        exog = test_df[self.exog_cols].values if self.exog_cols else None
        n = len(test_df)
        fc = self._result.get_forecast(steps=n, exog=exog)
        return fc.predicted_mean

    def predict_quantiles(
        self,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        quantiles: list[float] = [0.1, 0.9],
    ) -> dict[float, np.ndarray]:
        from scipy.stats import norm

        exog = test_df[self.exog_cols].values if self.exog_cols else None
        n = len(test_df)
        fc = self._result.get_forecast(steps=n, exog=exog)
        mu = fc.predicted_mean
        se = fc.se_mean
        result = {}
        for q in quantiles:
            z = norm.ppf(q)
            result[q] = mu + z * se
        return result


# 3. LightGBM (quantile regression for point + PI)

class LightGBMModel(BaseForecaster):
    """
    LightGBM with quantile loss for native prediction intervals.
    """

    name = "LightGBM"

    def __init__(self, params: dict | None = None):
        self._default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
        self._params = {**self._default_params, **(params or {})}
        self._models: dict[float, object] = {}

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
        import lightgbm as lgb

        X = train_df[feature_cols].values
        y = train_df[target_col].values

        # Exclude imputed target rows from training
        mask = train_df["is_imputed"].values == 0 if "is_imputed" in train_df.columns else np.ones(len(y), dtype=bool)
        X_clean, y_clean = X[mask], y[mask]

        for q in [0.1, 0.5, 0.9]:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self._params,
            )
            model.fit(X_clean, y_clean)
            self._models[q] = model

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        X = test_df[feature_cols].values
        return self._models[0.5].predict(X)

    def predict_quantiles(
        self,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        quantiles: list[float] = [0.1, 0.9],
    ) -> dict[float, np.ndarray]:
        X = test_df[feature_cols].values
        return {q: self._models[q].predict(X) for q in quantiles if q in self._models}
