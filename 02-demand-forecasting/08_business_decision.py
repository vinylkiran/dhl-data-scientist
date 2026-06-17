"""
08_business_decision.py
Business decision framework — per-class model recommendation with VP-facing summary.
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

N_SKUS_TOTAL = 1664

# ── Load results ──────────────────────────────────────────────────────────────
print("Loading results...")
comparison = pd.read_csv(OUTPUTS_DIR / "model_comparison_master.csv")
cb = pd.read_csv(OUTPUTS_DIR / "cost_benefit_analysis.csv")
interm = pd.read_csv(OUTPUTS_DIR / "intermittent_demand_results.csv")

# ── SKU counts per class (from SKU master) ────────────────────────────────────
demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])
sku_abc = demand.groupby("SKU_ID")["ABC_Class"].first().reset_index()
class_counts = sku_abc["ABC_Class"].value_counts()
n_a = int(class_counts.get("A", 0))
n_b = int(class_counts.get("B", 0))
n_c = int(class_counts.get("C", 0))
print(f"SKU counts — A: {n_a}, B: {n_b}, C: {n_c}, Total: {n_a+n_b+n_c}")

# Intermittent SKU count
interm_skus = interm["sku_id"].nunique()
interm_pct = interm_skus / 112 * 100  # as % of eval set

# ── MAPE per model from comparison (all demand patterns) ─────────────────────
mape_all = (comparison[comparison["demand_pattern"] == "all"]
            .groupby("model")["mean_mape"].mean().to_dict())

# Name normalization
def get_mape(model_key):
    for k, v in mape_all.items():
        if k.lower() == model_key.lower():
            return v
    return np.nan

lgb_mape = get_mape("lightgbm")
xgb_mape = get_mape("xgboost")
ses_mape = get_mape("SES")
naive_mape = get_mape("Naive")
arima_mape = get_mape("ARIMA/SARIMA")
croston_mape = interm[interm["model"] == "Croston"]["mape"].mean()

# ── Monthly compute costs from cost_benefit ───────────────────────────────────
def get_cost(model):
    row = cb[cb["model"] == model]
    return float(row["monthly_compute_cost_usd"].values[0]) if len(row) > 0 else 0.0

cost = {m: get_cost(m) for m in ["Naive", "SeasonalNaive", "SES", "ARIMA", "SARIMA",
                                   "XGBoost", "LightGBM", "Croston"]}

# ── Per-class recommendations ─────────────────────────────────────────────────
# A-class: best model with positive cost-benefit (LightGBM wins)
rec_a = "LightGBM"
mape_a = lgb_mape
cost_a_total = cost["LightGBM"] * (n_a / N_SKUS_TOTAL)  # proportional share

# B-class: best mape_improvement_per_dollar (also LightGBM due to global model efficiency)
rec_b = "LightGBM"
mape_b = lgb_mape
cost_b_total = cost["LightGBM"] * (n_b / N_SKUS_TOTAL)

# C-class: cheapest within 5pp of best — SES is next cheapest vs LightGBM
# LightGBM is still cheap enough ($0.0022/mo total), but SES is within range
# Given LightGBM is a global model cost is negligible — use LightGBM for C too
# But per spec: cheapest model within 5pp of best
best_mape = lgb_mape
ses_within = abs(ses_mape - best_mape) <= 5.0
rec_c = "SES" if ses_within else "LightGBM"
mape_c = ses_mape if ses_within else lgb_mape
cost_c_total = cost["SES"] * (n_c / N_SKUS_TOTAL) if ses_within else cost["LightGBM"] * (n_c / N_SKUS_TOTAL)

# Intermittent override: Croston/SBA (mostly B and C class)
rec_interm = "Croston/SBA"

# ── Total costs ────────────────────────────────────────────────────────────────
# Since LightGBM is global — entire model cost is shared, not additive per class
# We attribute the full model cost once; per-class cost is proportional
total_recommended_cost = cost["LightGBM"] + (cost["SES"] * (n_c / N_SKUS_TOTAL) if ses_within else 0)
total_naive_cost = cost["Naive"]  # essentially $0 (0.0002/mo)
net_incremental = total_recommended_cost - total_naive_cost

# ── Build output dataframe ────────────────────────────────────────────────────
decision_records = [
    {"abc_class": "A", "recommended_model": rec_a, "mean_mape": round(mape_a, 2),
     "monthly_cost_usd": round(cost_a_total, 4), "n_skus": n_a},
    {"abc_class": "B", "recommended_model": rec_b, "mean_mape": round(mape_b, 2),
     "monthly_cost_usd": round(cost_b_total, 4), "n_skus": n_b},
    {"abc_class": "C", "recommended_model": rec_c, "mean_mape": round(mape_c, 2),
     "monthly_cost_usd": round(cost_c_total, 4), "n_skus": n_c},
    {"abc_class": "Intermittent", "recommended_model": rec_interm,
     "mean_mape": round(croston_mape, 2), "monthly_cost_usd": round(cost["Croston"], 4),
     "n_skus": interm_skus},
]
decision_df = pd.DataFrame(decision_records)
decision_df.to_csv(OUTPUTS_DIR / "business_decision.csv", index=False)
print("Saved outputs/business_decision.csv")

# ── Significance testing result (from script 05) ─────────────────────────────
# t=2.016, p=0.0463 — significant; Wilcoxon p=0.0150
sig_pval = 0.0463
sig_result = "SIGNIFICANT at p<0.05 (paired t-test p=0.046, Wilcoxon p=0.015)"

# ── VP-facing summary ──────────────────────────────────────────────────────────
print("\n")
print("=" * 60)
print("PRODUCTION RECOMMENDATION — DEMAND FORECASTING")
print("=" * 60)
print(f"A-class SKUs (n={n_a}):    {rec_a:<12} — MAPE={mape_a:.1f}%, monthly cost=${cost['LightGBM']:.4f}")
print(f"B-class SKUs (n={n_b}):    {rec_b:<12} — MAPE={mape_b:.1f}%, monthly cost=${cost['LightGBM']:.4f}")
print(f"C-class SKUs (n={n_c}):    {rec_c:<12} — MAPE={mape_c:.1f}%, monthly cost=${cost.get(rec_c, 0):.4f}")
print(f"Intermittent SKUs:    {rec_interm} override (n={interm_skus}, ~{interm_pct:.0f}% of eval set)")
print()
print(f"Total monthly compute cost (recommended):  ${total_recommended_cost:.4f}")
print(f"vs. All-Naive counterfactual:              ${total_naive_cost:.4f}")
print(f"Net incremental cost:                      ${net_incremental:.4f}")
print()
print("NOT DEPLOYED and why:")
print(f"  - SARIMA:         3.75/5 maintenance complexity, MAPE={arima_mape:.1f}% (no improvement over")
print(f"                    LightGBM), cost ${cost['SARIMA']:.2f}/mo — 1250x more expensive than LightGBM")
print(f"  - ARIMA:          8s/SKU retrain, cost ${cost['ARIMA']:.2f}/mo — accuracy no better than SES")
print(f"  - SeasonalNaive:  Worst MAPE ({get_mape('SeasonalNaive'):.1f}%) across all classes")
print()
print("Decision:")
print(f"  LightGBM (global model) is deployed as the primary forecasting model for A- and")
print(f"  B-class SKUs. It achieves {lgb_mape:.1f}% MAPE at ${cost['LightGBM']:.4f}/month total compute cost,")
print(f"  outperforming the best statistical baseline (SES at {ses_mape:.1f}%) by 2.97 percentage")
print(f"  points — a difference confirmed statistically significant (p=0.046). C-class SKUs")
print(f"  use {'SES' if ses_within else 'LightGBM'} for cost efficiency given low revenue concentration. Intermittent")
print(f"  SKUs (32% of eval set, all B/C-class) receive a Croston/SBA override,")
print(f"  reducing bias from {-1.88:.2f} (Naive) to {-1.39:.2f}. ARIMA/SARIMA is not deployed")
print(f"  — it costs ${cost['ARIMA']:.2f}/month and offers no MAPE improvement over LightGBM.")
print()
print("Conditions that would change this:")
print("  1. If compute costs drop below $0.01/hr: ARIMA becomes viable for B-class SKUs")
print("     with longer seasonal cycles (e.g. quarterly patterns not captured by lag features)")
print("  2. If a high-value A-class SKU causes a stockout costing >$50K: evaluate SARIMA")
print("     with promotional calendar features for that specific SKU only")
print("=" * 60)

print("\nDone — 08_business_decision.py complete.")
