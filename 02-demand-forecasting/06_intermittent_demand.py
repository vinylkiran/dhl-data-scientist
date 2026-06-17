"""
06_intermittent_demand.py
Croston's method and SBA variant for intermittent demand SKUs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Croston's method (from scratch) ──────────────────────────────────────────
def croston(demand, alpha=0.1):
    """
    Croston's method for intermittent demand forecasting.
    Separates demand size (a) from inter-demand interval (p).
    Returns forecast = a / p (constant level forecast).
    """
    demand = np.array(demand, dtype=float)
    n = len(demand)

    # Find first non-zero
    first_nonzero = np.where(demand > 0)[0]
    if len(first_nonzero) == 0:
        return np.zeros(1)

    # Initialize
    idx0 = first_nonzero[0]
    a = demand[idx0]       # level of non-zero demand
    p = 1.0                # inter-demand interval
    q = 1                  # periods since last demand

    forecasts_train = []
    for i in range(n):
        if i < idx0:
            forecasts_train.append(np.nan)
            continue
        if demand[i] > 0:
            # Update
            a = alpha * demand[i] + (1 - alpha) * a
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
        forecasts_train.append(a / p)

    # One-step-ahead forecast for holdout: the last fitted value
    forecast_value = a / p
    return forecast_value

def croston_sba(demand, alpha=0.1):
    """
    Syntetos-Boylan Approximation: multiply Croston forecast by (1 - alpha/2).
    """
    fc = croston(demand, alpha)
    return fc * (1 - alpha / 2)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
demand_raw = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])
daily = (demand_raw.groupby(["Date", "SKU_ID", "ABC_Class", "Category"])["Quantity_Demanded"]
         .sum().reset_index())

all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
n_days = len(all_dates)
TRAIN_END = pd.Timestamp("2023-09-30")
TEST_START = pd.Timestamp("2023-10-01")
n_train_days = len(all_dates[all_dates <= TRAIN_END])

# ── Load eval SKUs ─────────────────────────────────────────────────────────────
eval_df = pd.read_csv(OUTPUTS_DIR / "eval_skus.csv")
eval_skus = eval_df["sku_id"].tolist()

# ── Identify intermittent SKUs in eval set ────────────────────────────────────
def get_sku_series(sku_id):
    s = daily[daily["SKU_ID"] == sku_id].set_index("Date")["Quantity_Demanded"]
    s = s.reindex(all_dates, fill_value=0)
    s.index.name = "Date"
    return s

print("Identifying intermittent SKUs...")
intermittent_skus = []
sku_zero_frac = {}

for sku_id in eval_skus:
    s = get_sku_series(sku_id)
    train_s = s[s.index <= TRAIN_END]
    zero_frac = (train_s == 0).sum() / len(train_s)
    sku_zero_frac[sku_id] = zero_frac
    if zero_frac > 0.40:
        intermittent_skus.append(sku_id)

print(f"Intermittent SKUs (>40% zeros in train): {len(intermittent_skus)} / {len(eval_skus)}")
pct_intermittent = len(intermittent_skus) / len(eval_skus) * 100
print(f"  = {pct_intermittent:.1f}% of eval catalogue")

# ABC class breakdown
sku_abc = daily.groupby("SKU_ID")[["ABC_Class"]].first().reset_index()
interm_abc = sku_abc[sku_abc["SKU_ID"].isin(intermittent_skus)]["ABC_Class"].value_counts()
print(f"Intermittent SKU ABC breakdown:\n{interm_abc}")

# ── Metrics helper ─────────────────────────────────────────────────────────────
def compute_metrics(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mae = np.mean(np.abs(actual - forecast))
    bias = np.mean(forecast - actual)
    return mape, rmse, mae, bias

# ── Evaluate Croston and SBA ───────────────────────────────────────────────────
print(f"\nEvaluating Croston and SBA on {len(intermittent_skus)} intermittent SKUs...")
records = []

for sku_id in intermittent_skus:
    s = get_sku_series(sku_id)
    train_s = s[s.index <= TRAIN_END]
    test_s = s[s.index >= TEST_START]
    actual = test_s.values
    n_test = len(actual)

    abc_cls = sku_abc[sku_abc["SKU_ID"] == sku_id]["ABC_Class"].values
    abc_cls = abc_cls[0] if len(abc_cls) > 0 else "C"

    # Croston — one scalar forecast applied uniformly to test period
    fc_croston_val = croston(train_s.values, alpha=0.1)
    fc_croston = np.full(n_test, fc_croston_val)

    fc_sba_val = croston_sba(train_s.values, alpha=0.1)
    fc_sba = np.full(n_test, fc_sba_val)

    # Naive
    last_val = float(train_s.iloc[-1])
    fc_naive = np.full(n_test, last_val)

    # Seasonal Naive (period=7)
    last_season = train_s.values[-7:].astype(float)
    fc_sn = np.array([last_season[i % 7] for i in range(n_test)])

    for model_name, fc in [
        ("Croston", fc_croston),
        ("Croston_SBA", fc_sba),
        ("Naive", fc_naive),
        ("SeasonalNaive", fc_sn),
    ]:
        m, r, a, b = compute_metrics(actual, fc)
        records.append({
            "sku_id": sku_id, "abc_class": abc_cls, "model": model_name,
            "mape": round(m, 3), "rmse": round(r, 3), "mae": round(a, 3), "bias": round(b, 3)
        })

# Also grab SES from baseline_results for these intermittent SKUs
baseline = pd.read_csv(OUTPUTS_DIR / "baseline_results.csv")
ses_interm = baseline[
    (baseline["model"] == "SES") & (baseline["sku_id"].isin(intermittent_skus))
][["sku_id", "abc_class", "model", "mape", "rmse", "mae", "bias"]].copy()
records_df = pd.DataFrame(records)
results_df = pd.concat([records_df, ses_interm], ignore_index=True)

results_df.to_csv(OUTPUTS_DIR / "intermittent_demand_results.csv", index=False)
print(f"Saved outputs/intermittent_demand_results.csv ({len(results_df)} rows)")

# ── Print comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INTERMITTENT DEMAND — MODEL COMPARISON")
print("=" * 70)
summary = results_df.groupby("model")[["mape", "rmse", "mae", "bias"]].mean().round(3)
print(summary.to_string())
print("=" * 70)

best_mape = summary["mape"].idxmin()
print(f"\nBest method by MAPE: {best_mape}")
print(f"Croston bias: {summary.loc['Croston', 'bias']:.3f}")
print(f"Naive bias:   {summary.loc['Naive', 'bias']:.3f}")
print(f"SES bias:     {summary.loc['SES', 'bias']:.3f}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
print("\nGenerating comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
models_ord = ["Naive", "SeasonalNaive", "SES", "Croston", "Croston_SBA"]
mape_vals = [summary.loc[m, "mape"] if m in summary.index else np.nan for m in models_ord]
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]

bars = ax.bar(models_ord, mape_vals, color=colors, alpha=0.85, edgecolor="white")
ax.set_title("Intermittent Demand — Mean MAPE by Method", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean MAPE (%)", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
for bar, val in zip(bars, mape_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "intermittent_demand_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved figures/intermittent_demand_comparison.png")

# ── Key findings ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print(f"  Intermittent SKUs: {len(intermittent_skus)} ({pct_intermittent:.1f}% of eval catalogue)")
print(f"  ABC class breakdown: {dict(interm_abc)}")
print(f"  Best method overall: {best_mape}")
croston_bias = summary.loc["Croston", "bias"]
naive_bias = summary.loc["Naive", "bias"]
print(f"  Bias reduction vs Naive: {naive_bias:.3f} (Naive) -> {croston_bias:.3f} (Croston)")
if abs(croston_bias) < abs(naive_bias):
    print("  Croston DOES reduce bias vs Naive on intermittent series.")
else:
    print("  Bias pattern: Naive and Croston comparable on this dataset.")
print("\nDone — 06_intermittent_demand.py complete.")
