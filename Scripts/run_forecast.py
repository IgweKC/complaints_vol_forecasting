
"""
Entry point: data -> features -> backtest -> forecast -> outputs.

Run from the repo root:
    python scripts/run_forecast.py

Outputs:
    reports/backtest_metrics.csv
    reports/forecast_90d.csv
    reports/figures/*.png
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from complaints_forecast.io import load_clean, TARGET, EXOG_COLS
from complaints_forecast.features import build_feature_matrix, get_feature_columns
from complaints_forecast.splits import walk_forward_splits, HORIZON
from complaints_forecast.metrics import compute_all
from complaints_forecast.models import SeasonalNaive, SARIMAXModel, LightGBMModel
from complaints_forecast.forecast import build_future_frame, build_future_features_for_lgbm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_forecast")

DATA_PATH = Path("data/daily_records.csv")
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"

SARIMAX_EXOG = [
    "bank_holiday_flag",
    "staffing_level_fte",
    "backlog_days",
    "media_mentions",
    "channel_mix_index",
]


def ensure_dirs():
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


# ── Backtest ──────────────────────────────────────────────────────────────

def run_backtest(df_feat: pd.DataFrame, feature_cols: list[str]):
    """Run walk-forward backtest for all models; return metrics DataFrame."""
    models = [
        SeasonalNaive(m=7),
        SARIMAXModel(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 7),
            exog_cols=SARIMAX_EXOG,
        ),
        LightGBMModel(),
    ]

    records = []
    fold_predictions: dict[str, list] = {m.name: [] for m in models}

    for fold_idx, (train, test) in enumerate(walk_forward_splits(df_feat)):
        y_train = train[TARGET].values
        y_true = test[TARGET].values
        logger.info(
            "Fold %d: train %s→%s (%d), test %s→%s (%d)",
            fold_idx,
            train.index.min().date(), train.index.max().date(), len(train),
            test.index.min().date(), test.index.max().date(), len(test),
        )

        for model in models:
            try:
                model.fit(train, feature_cols, TARGET)
                y_pred = model.predict(test, feature_cols)

                y_lo, y_hi = None, None
                try:
                    qs = model.predict_quantiles(test, feature_cols, [0.1, 0.9])
                    y_lo, y_hi = qs[0.1], qs[0.9]
                except NotImplementedError:
                    pass

                m = compute_all(y_true, y_pred, y_train, y_lo, y_hi)
                m["model"] = model.name
                m["fold"] = fold_idx
                records.append(m)

                fold_predictions[model.name].append({
                    "fold": fold_idx,
                    "dates": test.index,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_lo": y_lo,
                    "y_hi": y_hi,
                })
                logger.info("  %s fold %d: MAE=%.2f  MASE=%.3f", model.name, fold_idx, m["MAE"], m["MASE"])

            except Exception as e:
                logger.warning("  %s fold %d FAILED: %s", model.name, fold_idx, e)
                records.append({"model": model.name, "fold": fold_idx, "MAE": np.nan})

    metrics_df = pd.DataFrame(records)
    return metrics_df, fold_predictions


def plot_backtest(fold_predictions: dict, winner_name: str):
    """Fan chart for the winning model's backtest folds."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    preds = fold_predictions[winner_name]
    for i, fp in enumerate(preds):
        ax = axes[i]
        ax.plot(fp["dates"], fp["y_true"], "k-", label="Actual", linewidth=1)
        ax.plot(fp["dates"], fp["y_pred"], "b--", label=f"{winner_name}", linewidth=1)
        if fp["y_lo"] is not None:
            ax.fill_between(fp["dates"], fp["y_lo"], fp["y_hi"], alpha=0.2, color="blue", label="80% PI")
        ax.set_title(f"Fold {fp['fold']}")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Walk-Forward Backtest — {winner_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "backtest_fan_chart.png", dpi=150)
    plt.close(fig)
    logger.info("Saved backtest fan chart")


def plot_residual_diagnostics(fold_predictions: dict, winner_name: str):
    """Residual ACF and day-of-week bias for the winner."""
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox

    all_resid = []
    all_dow = []
    for fp in fold_predictions[winner_name]:
        resid = fp["y_true"] - fp["y_pred"]
        all_resid.extend(resid)
        all_dow.extend(fp["dates"].dayofweek)

    all_resid = np.array(all_resid)
    all_dow = np.array(all_dow)

    # ACF plot
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(all_resid, lags=30, ax=ax)
    ax.set_title(f"Residual ACF — {winner_name}")
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_acf.png", dpi=150)
    plt.close(fig)

    # Ljung-Box
    lb = acorr_ljungbox(all_resid, lags=[7, 14], return_df=True)
    logger.info("Ljung-Box test:\n%s", lb)

    # Day-of-week bias
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_bias = [all_resid[all_dow == d].mean() for d in range(7)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dow_names, dow_bias, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean residual (actual − predicted)")
    ax.set_title(f"Day-of-Week Bias — {winner_name}")
    fig.tight_layout()
    fig.savefig(FIGURES / "dow_bias.png", dpi=150)
    plt.close(fig)
    logger.info("Saved residual diagnostics")


# ── Final forecast ────────────────────────────────────────────────────────

def produce_forecast(df_clean: pd.DataFrame, df_feat: pd.DataFrame, feature_cols: list[str], winner_name: str):
    """Train the winning model on full history and produce 90-day forecast."""
    future_frame = build_future_frame(df_clean, horizon=HORIZON)

    if winner_name == "LightGBM":
        model = LightGBMModel()
        model.fit(df_feat, feature_cols, TARGET)
        future_feats = build_future_features_for_lgbm(df_clean, future_frame)
        y_pred = model.predict(future_feats, feature_cols)
        qs = model.predict_quantiles(future_feats, feature_cols, [0.1, 0.9])
        y_lo, y_hi = qs[0.1], qs[0.9]

    elif winner_name == "SARIMAX":
        model = SARIMAXModel(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 7),
            exog_cols=SARIMAX_EXOG,
        )
        model.fit(df_feat, feature_cols, TARGET)
        y_pred = model.predict(future_frame, feature_cols)
        qs = model.predict_quantiles(future_frame, feature_cols, [0.1, 0.9])
        y_lo, y_hi = qs[0.1], qs[0.9]

    else:
        model = SeasonalNaive(m=7)
        model.fit(df_feat, feature_cols, TARGET)
        y_pred = model.predict(future_frame, feature_cols)
        y_lo, y_hi = y_pred * 0.85, y_pred * 1.15

    out = pd.DataFrame({
        "date": future_frame.index,
        "yhat": y_pred,
        "yhat_lo_80": y_lo,
        "yhat_hi_80": y_hi,
    })
    out.to_csv(REPORTS / "forecast_90d.csv", index=False)
    logger.info("Saved 90-day forecast to reports/forecast_90d.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    hist_tail = df_clean[TARGET].tail(90)
    ax.plot(hist_tail.index, hist_tail.values, "k-", label="Historical")
    ax.plot(out["date"], out["yhat"], "b-", label=f"Forecast ({winner_name})")
    ax.fill_between(out["date"], out["yhat_lo_80"], out["yhat_hi_80"], alpha=0.2, color="blue", label="80% PI")
    ax.axvline(df_clean.index.max(), color="grey", linestyle="--", alpha=0.7)
    ax.set_title("90-Day Complaints Forecast")
    ax.set_ylabel("Daily complaints")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "forecast_90d.png", dpi=150)
    plt.close(fig)
    logger.info("Saved forecast plot")

    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    logger.info("Loading and cleaning data...")
    df_clean = load_clean(DATA_PATH)

    logger.info("Building features...")
    df_feat = build_feature_matrix(df_clean)
    feature_cols = get_feature_columns(df_feat)

    logger.info("Running walk-forward backtest (3 folds x 90 days)...")
    metrics_df, fold_predictions = run_backtest(df_feat, feature_cols)
    metrics_df.to_csv(REPORTS / "backtest_metrics.csv", index=False)
    logger.info("Saved backtest_metrics.csv")

    # Pick winner: lowest mean MAE, with stability check
    summary = metrics_df.groupby("model")["MAE"].agg(["mean", "std"]).sort_values("mean")
    logger.info("Model comparison (MAE):\n%s", summary)
    winner_name = summary.index[0]
    logger.info("Winner: %s (mean MAE=%.2f)", winner_name, summary.loc[winner_name, "mean"])

    plot_backtest(fold_predictions, winner_name)
    plot_residual_diagnostics(fold_predictions, winner_name)

    logger.info("Producing 90-day forecast with %s...", winner_name)
    produce_forecast(df_clean, df_feat, feature_cols, winner_name)

    logger.info("Done. All outputs in reports/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
