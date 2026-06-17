"""
03_arima_sarima.py
ARIMA/SARIMA grid search and evaluation on the 12 sample SKUs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as scipy_stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])
daily = (demand.groupby(["Date", "SKU_ID", "ABC_Class", "Category"])["Quantity_Demanded"]
         .sum().reset_index())

all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
TRAIN_END = pd.Timestamp("2023-09-30")
TEST_START = pd.Timestamp("2023-10-01")

# ── Load sample SKUs and EDA results ─────────────────────────────────────────
sample_df = pd.read_csv(OUTPUTS_DIR / "sample_skus.csv")
eda_df = pd.read_csv(OUTPUTS_DIR / "eda_results.csv")
sample_skus = sample_df["sku_id"].tolist()
print(f"Sample SKUs: {len(sample_skus)}")

def get_sku_series(sku_id):
    s = daily[daily["SKU_ID"] == sku_id].set_index("Date")["Quantity_Demanded"]
    s = s.reindex(all_dates, fill_value=0)
    s.index.name = "Date"
    return s

def compute_metrics(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mae = np.mean(np.abs(actual - forecast))
    return mape, rmse, mae

# ── ARIMA grid search ────────────────────────────────────────────────────────
p_range = [0, 1, 2]
d_range = [0, 1]
q_range = [0, 1, 2]
FALLBACK_ORDER = (1, 1, 1)

print("\nStarting ARIMA/SARIMA grid search...")
print(f"Grid: p={p_range}, d={d_range}, q={q_range} => {len(p_range)*len(d_range)*len(q_range)} combos per SKU")

records = []

for sku_id in sample_skus:
    print(f"\n--- {sku_id} ---")
    series = get_sku_series(sku_id)
    train_s = series[series.index <= TRAIN_END].astype(float)
    test_s = series[series.index >= TEST_START].astype(float)
    actual = test_s.values

    eda_row = eda_df[eda_df["SKU_ID"] == sku_id]
    weekly_seasonal = False
    if len(eda_row) > 0:
        weekly_seasonal = float(eda_row["Weekly_F_pval"].values[0]) < 0.05

    # ── ARIMA grid search ────────────────────────────────────────────────────
    best_aic = np.inf
    best_order = FALLBACK_ORDER
    best_arima = None

    for p, d, q in product(p_range, d_range, q_range):
        try:
            mod = sm.tsa.ARIMA(train_s.values, order=(p, d, q), trend="n")
            fit = mod.fit(method="innovations_mle")
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
                best_arima = fit
        except Exception:
            continue

    if best_arima is None:
        # Full fallback
        try:
            mod = sm.tsa.ARIMA(train_s.values, order=FALLBACK_ORDER, trend="n")
            best_arima = mod.fit(method="innovations_mle")
            best_order = FALLBACK_ORDER
            best_aic = best_arima.aic
        except Exception as e:
            print(f"  ARIMA fallback failed: {e}")
            records.append({
                "sku_id": sku_id, "model_type": "ARIMA", "best_order": str(FALLBACK_ORDER),
                "aic": np.nan, "ljung_box_pval": np.nan, "mape": np.nan, "rmse": np.nan, "mae": np.nan
            })
            continue

    print(f"  Best ARIMA: {best_order}, AIC={best_aic:.2f}")

    # ARIMA forecast
    arima_fc = best_arima.forecast(steps=len(actual))
    mape_a, rmse_a, mae_a = compute_metrics(actual, arima_fc)

    # Ljung-Box on ARIMA residuals
    resid = best_arima.resid
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_pval = float(lb["lb_pvalue"].values[0])

    records.append({
        "sku_id": sku_id, "model_type": "ARIMA", "best_order": str(best_order),
        "aic": round(best_aic, 2), "ljung_box_pval": round(lb_pval, 4),
        "mape": round(mape_a, 3), "rmse": round(rmse_a, 3), "mae": round(mae_a, 3)
    })
    print(f"  ARIMA MAPE={mape_a:.1f}%, RMSE={rmse_a:.2f}, LB p={lb_pval:.4f}")

    # ── SARIMA (if seasonal or always try) ───────────────────────────────────
    p_b, d_b, q_b = best_order
    try:
        smod = sm.tsa.SARIMAX(
            train_s.values,
            order=(p_b, d_b, q_b),
            seasonal_order=(1, 0, 1, 7),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sfit = smod.fit(method="lbfgs", disp=False)
        sarima_fc = sfit.forecast(steps=len(actual))
        mape_s, rmse_s, mae_s = compute_metrics(actual, sarima_fc)
        lb_s = acorr_ljungbox(sfit.resid, lags=[10], return_df=True)
        lb_pval_s = float(lb_s["lb_pvalue"].values[0])
        sarima_aic = sfit.aic
        print(f"  SARIMA({p_b},{d_b},{q_b})(1,0,1,7) AIC={sarima_aic:.2f}, "
              f"MAPE={mape_s:.1f}%, LB p={lb_pval_s:.4f}")
        records.append({
            "sku_id": sku_id, "model_type": "SARIMA",
            "best_order": f"({p_b},{d_b},{q_b})(1,0,1,7)",
            "aic": round(sarima_aic, 2), "ljung_box_pval": round(lb_pval_s, 4),
            "mape": round(mape_s, 3), "rmse": round(rmse_s, 3), "mae": round(mae_s, 3)
        })
        sarima_resid = sfit.resid
        sarima_fc_vals = sarima_fc
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        sarima_resid = resid
        sarima_fc_vals = arima_fc

    # ── Residual diagnostics figure ───────────────────────────────────────────
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ARIMA Residual Diagnostics — {sku_id}", fontsize=12, fontweight="bold")

        # Residual plot
        axes[0, 0].plot(resid, linewidth=0.7, color="steelblue")
        axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0, 0].set_title("Residuals")
        axes[0, 0].set_ylabel("Residual")

        # Histogram
        axes[0, 1].hist(resid, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].set_xlabel("Residual")

        # Q-Q plot
        scipy_stats.probplot(resid, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")

        # Residual ACF
        from statsmodels.tsa.stattools import acf
        lags = min(25, len(resid) // 2 - 1)
        acf_vals = acf(resid, nlags=lags, fft=True)
        ci = 1.96 / np.sqrt(len(resid))
        axes[1, 1].bar(range(len(acf_vals)), acf_vals, color="darkorange", alpha=0.7)
        axes[1, 1].axhline(ci, color="red", linestyle="--", linewidth=0.8)
        axes[1, 1].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        axes[1, 1].set_title("Residual ACF")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"arima_residuals_{sku_id}.png", dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Residual plot failed: {e}")
        plt.close("all")

    # ── Forecast plot with CI ─────────────────────────────────────────────────
    try:
        fc_ci = best_arima.get_forecast(steps=len(actual)).summary_frame()
        lower = fc_ci["mean_ci_lower"].values
        upper = fc_ci["mean_ci_upper"].values
        fc_mean = fc_ci["mean"].values

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(train_s.index[-90:], train_s.values[-90:], label="Train (last 90d)",
                color="steelblue", linewidth=0.8)
        ax.plot(test_s.index, actual, label="Actual", color="black", linewidth=0.8)
        ax.plot(test_s.index, fc_mean, label="ARIMA Forecast", color="darkorange", linewidth=1.0)
        ax.fill_between(test_s.index, lower, upper, alpha=0.2, color="darkorange", label="95% CI")
        ax.set_title(f"ARIMA{best_order} Forecast — {sku_id}", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_ylabel("Quantity Demanded")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"arima_forecast_{sku_id}.png", dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Forecast plot failed: {e}")
        plt.close("all")

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUTS_DIR / "arima_results.csv", index=False)
print(f"\nSaved outputs/arima_results.csv ({len(results_df)} rows)")

# ── Print summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("ARIMA/SARIMA RESULTS — PER SKU")
print("=" * 85)
print(f"{'SKU_ID':<16} {'Type':<8} {'Order':<22} {'AIC':<10} {'LB_p':<8} {'MAPE%':<8} {'RMSE':<8} {'MAE'}")
print("-" * 85)
for _, r in results_df.iterrows():
    print(f"{r['sku_id']:<16} {r['model_type']:<8} {str(r['best_order']):<22} "
          f"{r['aic']:<10.2f} {r['ljung_box_pval']:<8.4f} "
          f"{r['mape']:<8.2f} {r['rmse']:<8.2f} {r['mae']:.2f}")
print("=" * 85)

print("\nOverall averages by model type:")
avg = results_df.groupby("model_type")[["mape", "rmse", "mae"]].mean().round(3)
print(avg.to_string())
print("\nDone — 03_arima_sarima.py complete.")
