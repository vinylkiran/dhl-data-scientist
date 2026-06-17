"""
07_cost_benefit_analysis.py
Cost-benefit analysis for model selection.
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

N_SKUS_TOTAL = 1664  # total SKU catalogue

# ── Training time estimates (seconds/SKU) ─────────────────────────────────────
timing = {
    "Naive":         {"train_s": 0.001, "inference_s": 0.001},
    "SeasonalNaive": {"train_s": 0.001, "inference_s": 0.001},
    "SES":           {"train_s": 0.05,  "inference_s": 0.001},
    "ARIMA":         {"train_s": 8.0,   "inference_s": 0.500},
    "SARIMA":        {"train_s": 15.0,  "inference_s": 0.500},
    "XGBoost":       {"train_s": 30.0 / N_SKUS_TOTAL, "inference_s": 0.001},
    "LightGBM":      {"train_s": 20.0 / N_SKUS_TOTAL, "inference_s": 0.001},
    "Croston":       {"train_s": 0.01,  "inference_s": 0.001},
}

# ── Maintenance complexity ────────────────────────────────────────────────────
complexity = {
    "Naive":         {"tuning": 1, "monitoring": 1, "explainability": 5, "deployment": 1},
    "SeasonalNaive": {"tuning": 1, "monitoring": 1, "explainability": 5, "deployment": 1},
    "SES":           {"tuning": 2, "monitoring": 2, "explainability": 4, "deployment": 2},
    "ARIMA":         {"tuning": 4, "monitoring": 3, "explainability": 2, "deployment": 3},
    "SARIMA":        {"tuning": 5, "monitoring": 4, "explainability": 2, "deployment": 4},
    "XGBoost":       {"tuning": 3, "monitoring": 3, "explainability": 2, "deployment": 4},
    "LightGBM":      {"tuning": 3, "monitoring": 3, "explainability": 2, "deployment": 4},
    "Croston":       {"tuning": 2, "monitoring": 2, "explainability": 3, "deployment": 2},
}

# ── Load MAPE from results ────────────────────────────────────────────────────
print("Loading model comparison results...")
comparison = pd.read_csv(OUTPUTS_DIR / "model_comparison_master.csv")

# Overall mean MAPE per model (demand_pattern='all')
model_mape = (comparison[comparison["demand_pattern"] == "all"]
              .groupby("model")["mean_mape"].mean()
              .to_dict())

# Map model names to cost-benefit model names
name_map = {
    "Naive": "Naive",
    "SeasonalNaive": "SeasonalNaive",
    "SES": "SES",
    "ARIMA/SARIMA": "ARIMA",  # treat as ARIMA for cost
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

model_mape_mapped = {name_map.get(k, k): v for k, v in model_mape.items()}
# Add SARIMA and Croston (SARIMA ~ same MAPE as ARIMA, Croston from intermittent results)
model_mape_mapped["SARIMA"] = model_mape_mapped.get("ARIMA", 80.0)

try:
    interm = pd.read_csv(OUTPUTS_DIR / "intermittent_demand_results.csv")
    croston_mape = interm[interm["model"] == "Croston"]["mape"].mean()
except Exception:
    croston_mape = 70.0
model_mape_mapped["Croston"] = croston_mape

print("MAPE by model:")
for m, v in sorted(model_mape_mapped.items(), key=lambda x: x[1]):
    print(f"  {m:<20}: {v:.2f}%")

# ── Compute cost metrics ───────────────────────────────────────────────────────
records = []
HOURLY_RATE = 0.10  # $/hr compute
WEEKS_PER_YEAR = 52

for model, t in timing.items():
    train_s_per_sku = t["train_s"]
    total_train_s = train_s_per_sku * N_SKUS_TOTAL
    # Weekly retraining compute hours
    weekly_retrain_hrs = total_train_s / 3600
    # Monthly compute cost (4 retrains per month)
    monthly_cost = weekly_retrain_hrs * 4 * HOURLY_RATE

    comp = complexity[model]
    complexity_avg = np.mean(list(comp.values()))

    mape = model_mape_mapped.get(model, np.nan)

    records.append({
        "model": model,
        "train_s_per_sku": round(train_s_per_sku, 6),
        "total_train_s_1664": round(total_train_s, 2),
        "weekly_retrain_hrs": round(weekly_retrain_hrs, 4),
        "monthly_compute_cost_usd": round(monthly_cost, 4),
        "maintenance_complexity": round(complexity_avg, 2),
        "mean_mape": round(mape, 2) if not np.isnan(mape) else np.nan,
        "tuning": comp["tuning"],
        "monitoring": comp["monitoring"],
        "explainability": comp["explainability"],
        "deployment": comp["deployment"],
    })

cb_df = pd.DataFrame(records)

# ── MAPE improvement per dollar ───────────────────────────────────────────────
# Baseline = Naive
baseline_mape = cb_df[cb_df["model"] == "Naive"]["mean_mape"].values[0]
baseline_cost = cb_df[cb_df["model"] == "Naive"]["monthly_compute_cost_usd"].values[0]

cb_df["delta_mape"] = baseline_mape - cb_df["mean_mape"]  # positive = improvement
cb_df["delta_monthly_cost"] = cb_df["monthly_compute_cost_usd"] - baseline_cost
cb_df["mape_improvement_per_dollar"] = (
    cb_df["delta_mape"] / np.maximum(cb_df["delta_monthly_cost"], 0.001)
).round(2)

cb_df.to_csv(OUTPUTS_DIR / "cost_benefit_analysis.csv", index=False)
print(f"\nSaved outputs/cost_benefit_analysis.csv")

# ── Print full table ───────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("COST-BENEFIT ANALYSIS — ALL MODELS")
print("=" * 100)
cols = ["model", "mean_mape", "monthly_compute_cost_usd", "maintenance_complexity",
        "delta_mape", "mape_improvement_per_dollar"]
print(cb_df[cols].sort_values("mean_mape").to_string(index=False))

# ── Decision logic per ABC class ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("DECISION LOGIC PER ABC CLASS")
print("=" * 70)

# A-class: best MAPE where mape_improvement_per_dollar > 0
a_candidates = cb_df[cb_df["mape_improvement_per_dollar"] > 0].sort_values("mean_mape")
if len(a_candidates) > 0:
    a_rec = a_candidates.iloc[0]["model"]
    a_mape = a_candidates.iloc[0]["mean_mape"]
    a_cost = a_candidates.iloc[0]["monthly_compute_cost_usd"]
else:
    a_rec = "SES"; a_mape = model_mape_mapped.get("SES", 62); a_cost = 0

# B-class: best mape_improvement_per_dollar ratio
b_candidates = cb_df[cb_df["mape_improvement_per_dollar"] > 0].sort_values(
    "mape_improvement_per_dollar", ascending=False
)
if len(b_candidates) > 0:
    b_rec = b_candidates.iloc[0]["model"]
    b_mape = b_candidates.iloc[0]["mean_mape"]
    b_cost = b_candidates.iloc[0]["monthly_compute_cost_usd"]
else:
    b_rec = "SES"; b_mape = 62; b_cost = 0

# C-class: cheapest with MAPE within 5pp of best model
best_mape = cb_df["mean_mape"].min()
c_candidates = cb_df[cb_df["mean_mape"] <= best_mape + 5.0].sort_values("monthly_compute_cost_usd")
if len(c_candidates) > 0:
    c_rec = c_candidates.iloc[0]["model"]
    c_mape = c_candidates.iloc[0]["mean_mape"]
    c_cost = c_candidates.iloc[0]["monthly_compute_cost_usd"]
else:
    c_rec = "Naive"; c_mape = baseline_mape; c_cost = baseline_cost

print(f"  A-class recommendation: {a_rec} (MAPE={a_mape:.1f}%, cost=${a_cost:.4f}/mo)")
print(f"  B-class recommendation: {b_rec} (MAPE={b_mape:.1f}%, cost=${b_cost:.4f}/mo)")
print(f"  C-class recommendation: {c_rec} (MAPE={c_mape:.1f}%, cost=${c_cost:.4f}/mo)")

# ── Pareto frontier plot ───────────────────────────────────────────────────────
print("\nGenerating cost-benefit frontier plot...")
fig, ax = plt.subplots(figsize=(10, 7))

colors_map = {
    "Naive": "#4C72B0", "SeasonalNaive": "#55A868", "SES": "#DD8452",
    "ARIMA": "#C44E52", "SARIMA": "#8172B2",
    "XGBoost": "#CCB974", "LightGBM": "#64B5CD", "Croston": "#E377C2"
}

for _, row in cb_df.iterrows():
    color = colors_map.get(row["model"], "gray")
    ax.scatter(row["monthly_compute_cost_usd"], row["mean_mape"],
               s=120, color=color, zorder=3, alpha=0.85)
    ax.annotate(row["model"], (row["monthly_compute_cost_usd"], row["mean_mape"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)

# Pareto frontier: lower-left is better (lower cost AND lower MAPE)
pareto = []
sorted_by_cost = cb_df.sort_values("monthly_compute_cost_usd")
min_mape_so_far = np.inf
for _, row in sorted_by_cost.iterrows():
    if row["mean_mape"] < min_mape_so_far:
        pareto.append(row)
        min_mape_so_far = row["mean_mape"]

if len(pareto) >= 2:
    px = [r["monthly_compute_cost_usd"] for r in pareto]
    py = [r["mean_mape"] for r in pareto]
    ax.plot(px, py, "r--", linewidth=1.5, label="Pareto Frontier", zorder=2)

ax.set_xlabel("Monthly Compute Cost (USD)", fontsize=12)
ax.set_ylabel("Mean MAPE (%)", fontsize=12)
ax.set_title("Cost-Benefit Frontier — All Models\n(Lower-left = better)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cost_benefit_frontier.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved figures/cost_benefit_frontier.png")

print("\nDone — 07_cost_benefit_analysis.py complete.")
