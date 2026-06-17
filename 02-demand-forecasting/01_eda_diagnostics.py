"""
01_eda_diagnostics.py
EDA and stationarity diagnostics for 12 representative SKUs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])

print(f"Demand rows: {len(demand):,}")
print(f"Date range: {demand['Date'].min()} to {demand['Date'].max()}")
print(f"Unique SKUs in demand: {demand['SKU_ID'].nunique()}")

# ── Aggregate demand by SKU × Date (sum across warehouses) ──────────────────
daily = (demand.groupby(["Date", "SKU_ID", "ABC_Class", "Category"])["Quantity_Demanded"]
         .sum().reset_index())

all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
n_days = len(all_dates)
print(f"Total days in range: {n_days}")

# ── SKU selection ────────────────────────────────────────────────────────────
# NOTE: SKUs only appear in demand data when they have demand (all nonzero_pct_of_data=1.0)
# Use nonzero_pct_of_730 (nonzero days / 730) to find active SKUs
sku_stats = daily.groupby("SKU_ID").agg(
    total_rows=("Quantity_Demanded", "count"),
    nonzero_rows=("Quantity_Demanded", lambda x: (x > 0).sum()),
    mean_demand=("Quantity_Demanded", "mean"),
).reset_index()
sku_stats["nonzero_pct"] = sku_stats["nonzero_rows"] / n_days

sku_abc = daily.groupby("SKU_ID")[["ABC_Class", "Category"]].first().reset_index()
sku_stats = sku_stats.merge(sku_abc, on="SKU_ID")

# All SKUs are active (>10% nonzero of 730 days, since min is 15%)
active = sku_stats[sku_stats["nonzero_pct"] > 0.10].copy()
print(f"\nAll active SKUs (>10% nonzero of 730 days): {len(active)}")
print(active["ABC_Class"].value_counts())
print(f"Categories: {active['Category'].nunique()}")

# Select 12: 4 A, 4 B, 4 C — each spanning >=4 categories
def pick_skus(df, cls, n=4):
    """Pick n SKUs from class cls, choosing from different categories."""
    sub = df[df["ABC_Class"] == cls].copy()
    # Sort by category then by mean demand (higher = more stable)
    sub = sub.sort_values(["Category", "mean_demand"], ascending=[True, False])
    cats = sub["Category"].unique()
    chosen = []
    cat_idx = 0
    while len(chosen) < n and cat_idx < len(cats):
        candidates = sub[sub["Category"] == cats[cat_idx]]
        if len(candidates) > 0:
            chosen.append(candidates.iloc[0]["SKU_ID"])
        cat_idx += 1
    # Fill remaining from any category
    remaining = sub[~sub["SKU_ID"].isin(chosen)]
    while len(chosen) < n and len(remaining) > 0:
        chosen.append(remaining.iloc[0]["SKU_ID"])
        remaining = remaining.iloc[1:]
    return chosen

skus_A = pick_skus(active, "A", 4)
skus_B = pick_skus(active, "B", 4)
skus_C = pick_skus(active, "C", 4)
sample_skus = skus_A + skus_B + skus_C
print(f"\nSelected 12 SKUs: {sample_skus}")

sample_df = active[active["SKU_ID"].isin(sample_skus)][
    ["SKU_ID", "ABC_Class", "Category"]
].copy()
sample_df.columns = ["sku_id", "abc_class", "category"]

# Verify category diversity
print("\nSelected SKU details:")
print(sample_df.to_string(index=False))
print(f"Unique categories in selection: {sample_df['category'].nunique()}")

sample_df.to_csv(OUTPUTS_DIR / "sample_skus.csv", index=False)
print(f"\nSaved {len(sample_df)} SKUs to outputs/sample_skus.csv")

# ── Helper functions ─────────────────────────────────────────────────────────
def get_sku_series(sku_id):
    """Return complete daily series reindexed to all 730 dates, zero-filled."""
    s = daily[daily["SKU_ID"] == sku_id].set_index("Date")["Quantity_Demanded"]
    s = s.reindex(all_dates, fill_value=0)
    s.index.name = "Date"
    return s

def run_adf(series):
    """ADF test — return (pval, stationary, recommended_d)."""
    result = adfuller(series, autolag="AIC")
    pval = result[1]
    stationary = pval < 0.05
    d = 0 if stationary else 1
    return pval, stationary, d

def weekly_f_test(series):
    """F-test for joint day-of-week effect using OLS dummy regression."""
    df_tmp = pd.DataFrame({"y": series.values, "dow": series.index.dayofweek})
    dummies = pd.get_dummies(df_tmp["dow"], prefix="dow", drop_first=True, dtype=float)
    X = sm.add_constant(dummies)
    y = df_tmp["y"].astype(float).values
    model = sm.OLS(y, X).fit()
    hypotheses = [f"dow_{i} = 0" for i in range(1, 7) if f"dow_{i}" in model.params]
    if not hypotheses:
        return 1.0
    ftest = model.f_test(hypotheses)
    return float(ftest.pvalue)

# ── 1. Time series panel (3×4) ────────────────────────────────────────────────
print("\nGenerating time series panel...")
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.suptitle("Time Series — 12 Sample SKUs (Aggregated Across Warehouses)", fontsize=14, fontweight="bold")

results = []
for idx, sku_id in enumerate(sample_skus):
    row, col = divmod(idx, 4)
    ax = axes[row][col]
    series = get_sku_series(sku_id)
    info = sample_df[sample_df["sku_id"] == sku_id].iloc[0]

    ax.plot(series.index, series.values, linewidth=0.7, alpha=0.85, color="steelblue")
    ax.set_title(f"{sku_id}\n{info['abc_class']} | {info['category']}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_ylabel("Qty Demanded", fontsize=6)

    adf_pval, stationary, d = run_adf(series)
    f_pval = weekly_f_test(series)
    results.append({
        "SKU_ID": sku_id,
        "ABC_Class": info["abc_class"],
        "Category": info["category"],
        "ADF_pval": round(adf_pval, 4),
        "Stationary": stationary,
        "d": d,
        "Weekly_F_pval": round(f_pval, 4),
        "Seasonal_Period": 7,
    })
    print(f"  {sku_id}: ADF p={adf_pval:.4f} ({'stationary' if stationary else 'non-stat'}), "
          f"d={d}, Weekly F p={f_pval:.4f}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_timeseries_panel.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved figures/eda_timeseries_panel.png")

# ── 2. ACF / PACF panel ───────────────────────────────────────────────────────
print("\nGenerating ACF/PACF panel...")
fig, axes = plt.subplots(12, 2, figsize=(14, 60))
fig.suptitle("ACF / PACF (lag=35) — 12 Sample SKUs", fontsize=14, fontweight="bold")

for idx, sku_id in enumerate(sample_skus):
    series = get_sku_series(sku_id)
    lags = min(35, len(series) // 2 - 1)
    acf_vals = acf(series, nlags=lags, fft=True)
    pacf_vals = pacf(series, nlags=lags)
    conf_int = 1.96 / np.sqrt(len(series))

    ax_acf = axes[idx][0]
    ax_pacf = axes[idx][1]

    ax_acf.bar(range(len(acf_vals)), acf_vals, color="steelblue", alpha=0.7)
    ax_acf.axhline(conf_int, color="red", linestyle="--", linewidth=0.8)
    ax_acf.axhline(-conf_int, color="red", linestyle="--", linewidth=0.8)
    ax_acf.set_title(f"{sku_id} — ACF", fontsize=8)
    ax_acf.set_ylim(-0.5, 1.05)

    ax_pacf.bar(range(len(pacf_vals)), pacf_vals, color="darkorange", alpha=0.7)
    ax_pacf.axhline(conf_int, color="red", linestyle="--", linewidth=0.8)
    ax_pacf.axhline(-conf_int, color="red", linestyle="--", linewidth=0.8)
    ax_pacf.set_title(f"{sku_id} — PACF", fontsize=8)
    ax_pacf.set_ylim(-0.6, 1.05)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_acf_pacf_panel.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved figures/eda_acf_pacf_panel.png")

# ── 3. Seasonal decomposition (per SKU) ──────────────────────────────────────
print("\nGenerating seasonal decomposition figures...")
for sku_id in sample_skus:
    series = get_sku_series(sku_id)
    series_pos = series.astype(float) + 0.01  # avoid exact zeros
    try:
        decomp = seasonal_decompose(series_pos, model="additive", period=7, extrapolate_trend="freq")
        fig, axes2 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Seasonal Decomposition (period=7) — {sku_id}", fontsize=12, fontweight="bold")
        for ax2, component, label, color in zip(
            axes2,
            [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
            ["Observed", "Trend", "Seasonal", "Residual"],
            ["steelblue", "darkorange", "green", "gray"]
        ):
            ax2.plot(component, linewidth=0.7, color=color)
            ax2.set_ylabel(label, fontsize=9)
            ax2.tick_params(labelsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"eda_decomposition_{sku_id}.png", dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Decomp failed for {sku_id}: {e}")
        plt.close("all")

print("Saved decomposition figures (one per SKU).")

# ── Summary table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("EDA SUMMARY TABLE")
print("=" * 100)
print(f"{'SKU_ID':<16} {'ABC':<5} {'Category':<25} {'ADF_p':<8} {'Stat?':<8} {'d':<3} {'Weekly_F_p':<12} {'Period'}")
print("-" * 100)
for _, r in results_df.iterrows():
    print(f"{r['SKU_ID']:<16} {r['ABC_Class']:<5} {r['Category']:<25} "
          f"{r['ADF_pval']:<8.4f} {str(r['Stationary']):<8} {r['d']:<3} "
          f"{r['Weekly_F_pval']:<12.4f} {r['Seasonal_Period']}")
print("=" * 100)

# Save
results_df.to_csv(OUTPUTS_DIR / "eda_results.csv", index=False)
print("\nSaved outputs/eda_results.csv")
print(f"\nStationary SKUs: {results_df['Stationary'].sum()} / {len(results_df)}")
print(f"Weekly seasonality (F p<0.05): {(results_df['Weekly_F_pval'] < 0.05).sum()} / {len(results_df)}")
print("\nDone — 01_eda_diagnostics.py complete.")
