"""
05_model_comparison.py
Unified model comparison across all forecasting approaches.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load all results ───────────────────────────────────────────────────────────
print("Loading results...")
baseline = pd.read_csv(OUTPUTS_DIR / "baseline_results.csv")
arima = pd.read_csv(OUTPUTS_DIR / "arima_results.csv")
ml = pd.read_csv(OUTPUTS_DIR / "ml_results.csv")

print(f"Baseline: {len(baseline)} rows, {baseline['sku_id'].nunique()} SKUs")
print(f"ARIMA: {len(arima)} rows, {arima['sku_id'].nunique()} SKUs")
print(f"ML: {len(ml)} rows, {ml['sku_id'].nunique()} SKUs")

# ── Load demand data to compute demand patterns ───────────────────────────────
demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])
daily = (demand.groupby(["Date", "SKU_ID"])["Quantity_Demanded"].sum().reset_index())
all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
n_days = len(all_dates)
TRAIN_END = pd.Timestamp("2023-09-30")

# Compute zero-demand fraction per SKU in training period
train_daily = daily[daily["Date"] <= TRAIN_END].copy()
sku_zero_frac = train_daily.groupby("SKU_ID").apply(
    lambda g: 1 - g["Quantity_Demanded"].gt(0).sum() / n_days
).reset_index()
sku_zero_frac.columns = ["sku_id", "zero_frac"]
# intermittent: zero_frac >= 0.40
sku_zero_frac["demand_pattern"] = np.where(
    sku_zero_frac["zero_frac"] >= 0.40, "intermittent", "stable"
)
print("\nDemand pattern distribution:")
print(sku_zero_frac["demand_pattern"].value_counts())

# ── Build unified long-form table ─────────────────────────────────────────────
# Baseline
baseline_long = baseline[["sku_id", "abc_class", "model", "mape", "rmse", "mae"]].copy()

# ARIMA — take best model per SKU (lower MAPE)
arima_best = (arima.sort_values("mape")
              .groupby("sku_id").first().reset_index())
arima_best["model"] = "ARIMA/SARIMA"
arima_long = arima_best[["sku_id", "model", "mape", "rmse", "mae"]].copy()
# add abc_class from baseline (12 SKUs are all in baseline)
arima_abc = baseline[["sku_id", "abc_class"]].drop_duplicates()
arima_long = arima_long.merge(arima_abc, on="sku_id", how="left")
arima_long["abc_class"] = arima_long["abc_class"].fillna("B")

# ML
ml_long = ml[["sku_id", "abc_class", "model", "mape", "rmse", "mae"]].copy()

# Combine
all_results = pd.concat([baseline_long, arima_long, ml_long], ignore_index=True)
all_results = all_results.merge(sku_zero_frac[["sku_id", "demand_pattern"]], on="sku_id", how="left")
all_results["demand_pattern"] = all_results["demand_pattern"].fillna("stable")

print(f"\nCombined results: {len(all_results)} rows")
print(f"Models: {all_results['model'].unique()}")

# ── Aggregate by model + abc_class + demand_pattern ──────────────────────────
comparison = (all_results.groupby(["model", "abc_class", "demand_pattern"])
              .agg(mean_mape=("mape", "mean"),
                   mean_rmse=("rmse", "mean"),
                   mean_mae=("mae", "mean"),
                   n_skus=("sku_id", "nunique"))
              .reset_index())

# Also aggregate without demand_pattern for top-level
comparison_top = (all_results.groupby(["model", "abc_class"])
                  .agg(mean_mape=("mape", "mean"),
                       mean_rmse=("rmse", "mean"),
                       mean_mae=("mae", "mean"),
                       n_skus=("sku_id", "nunique"))
                  .reset_index())
comparison_top["demand_pattern"] = "all"

master = pd.concat([comparison, comparison_top], ignore_index=True)
master.to_csv(OUTPUTS_DIR / "model_comparison_master.csv", index=False)
print("Saved outputs/model_comparison_master.csv")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("MODEL COMPARISON — MEAN MAPE BY MODEL AND ABC CLASS")
print("=" * 85)
pivot = comparison_top.pivot_table(
    values="mean_mape", index="model", columns="abc_class"
).round(2)
print(pivot.to_string())

print("\n" + "=" * 85)
print("MODEL COMPARISON — BY DEMAND PATTERN")
print("=" * 85)
pattern_tbl = (all_results.groupby(["model", "demand_pattern"])
               .agg(mean_mape=("mape", "mean"), n_skus=("sku_id", "nunique"))
               .reset_index().round(2))
print(pattern_tbl.to_string(index=False))

# ── Best model per ABC class ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BEST MODEL PER ABC CLASS (by mean MAPE)")
print("=" * 60)
for cls in ["A", "B", "C"]:
    sub = comparison_top[comparison_top["abc_class"] == cls].sort_values("mean_mape")
    if len(sub) > 0:
        best = sub.iloc[0]
        print(f"  {cls}-class: {best['model']:20s} MAPE={best['mean_mape']:.2f}%, n={best['n_skus']}")

# ── Statistical significance testing ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE — ML vs. Best Baseline (per-SKU MAPE)")
print("=" * 70)

# Best ML: lightgbm
# Best baseline: SES (lowest overall MAPE)
baseline_ses = baseline[baseline["model"] == "SES"][["sku_id", "mape"]].rename(columns={"mape": "mape_ses"})
ml_lgb = ml[ml["model"] == "lightgbm"][["sku_id", "mape"]].rename(columns={"mape": "mape_lgb"})

sig_df = baseline_ses.merge(ml_lgb, on="sku_id").dropna()
print(f"SKUs with both SES and LightGBM results: {len(sig_df)}")

if len(sig_df) >= 2:
    ses_mapes = sig_df["mape_ses"].values
    lgb_mapes = sig_df["mape_lgb"].values
    diffs = ses_mapes - lgb_mapes  # positive = ML improves over SES

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(ses_mapes, lgb_mapes)
    # Wilcoxon signed-rank
    try:
        w_stat, w_pval = stats.wilcoxon(diffs)
    except Exception:
        w_stat, w_pval = np.nan, np.nan

    sig_at_05 = t_pval < 0.05 or w_pval < 0.05
    conclusion = "SIGNIFICANT at p<0.05" if sig_at_05 else "NOT significant at p<0.05"

    print(f"\n  Paired t-test:  t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  Wilcoxon:       W={w_stat:.1f}, p={w_pval:.4f}")
    print(f"  Mean MAPE (SES):       {ses_mapes.mean():.2f}%")
    print(f"  Mean MAPE (LightGBM):  {lgb_mapes.mean():.2f}%")
    print(f"  Mean improvement:      {diffs.mean():.2f}pp")
    print(f"  Conclusion: {conclusion}")

# ── Recommendation matrix ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RECOMMENDATION MATRIX — Per ABC Class × Demand Pattern")
print("=" * 70)
print(f"{'ABC':<6} {'Pattern':<15} {'Best Model':<22} {'Mean MAPE%'}")
print("-" * 60)
for cls in ["A", "B", "C"]:
    for pattern in ["stable", "intermittent"]:
        sub = comparison[
            (comparison["abc_class"] == cls) & (comparison["demand_pattern"] == pattern)
        ].sort_values("mean_mape")
        if len(sub) > 0:
            best = sub.iloc[0]
            print(f"  {cls:<4} {pattern:<15} {best['model']:<22} {best['mean_mape']:.2f}%")
        else:
            print(f"  {cls:<4} {pattern:<15} {'N/A':<22} N/A")

# ── Bar chart: MAPE by model and ABC class ────────────────────────────────────
print("\nGenerating model comparison chart...")
# Focus on models that ran on eval set (baseline + ML), plus ARIMA separately noted
plot_df = comparison_top[comparison_top["model"].isin(
    ["Naive", "SeasonalNaive", "SES", "xgboost", "lightgbm"]
)].copy()

fig, ax = plt.subplots(figsize=(12, 6))
models = plot_df["model"].unique()
classes = ["A", "B", "C"]
x = np.arange(len(classes))
width = 0.15
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for i, model in enumerate(models):
    mapes = [
        plot_df[(plot_df["model"] == model) & (plot_df["abc_class"] == cls)]["mean_mape"].values
        for cls in classes
    ]
    mapes = [m[0] if len(m) > 0 else np.nan for m in mapes]
    offset = (i - len(models) / 2 + 0.5) * width
    bars = ax.bar(x + offset, mapes, width, label=model, color=colors[i % len(colors)], alpha=0.85)

ax.set_xlabel("ABC Class", fontsize=12)
ax.set_ylabel("Mean MAPE (%)", fontsize=12)
ax.set_title("Model Comparison — Mean MAPE by ABC Class", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=12)
ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_mape_by_class.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved figures/model_comparison_mape_by_class.png")

print("\nDone — 05_model_comparison.py complete.")
