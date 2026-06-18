"""
06_cost_benefit_analysis.py
WMS Anomaly Detection — Cost-Benefit Analysis
Compute and compare economic cost of each detection method.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load FP rate estimates from method comparison ─────────────────────────────
fp_df = pd.read_csv(OUTPUT_DIR / "_fp_rates.csv")
comp_df = pd.read_csv(OUTPUT_DIR / "method_comparison.csv", parse_dates=["date"])

spc_fp_rate = fp_df[fp_df["method"] == "SPC"]["fp_rate_estimate"].values[0]
if_fp_rate = fp_df[fp_df["method"] == "IF"]["fp_rate_estimate"].values[0]
lof_fp_rate = fp_df[fp_df["method"] == "LOF"]["fp_rate_estimate"].values[0]

# Use a capped SPC FP rate: qualitative sample shows Rule3 (8 consecutive) fires
# frequently on normal data with synthetic stable distributions. Cap at 60% as
# a conservative upper bound (SPC on real data typically 10-30% FP for run rules).
# The raw sample was drawn from low-variance period, inflating the estimate.
SPC_FP_RATE_ADJUSTED = 0.60
IF_FP_RATE = if_fp_rate    # 0.20
LOF_FP_RATE = lof_fp_rate  # 0.20

print(f"FP rates:  SPC={SPC_FP_RATE_ADJUSTED:.0%} (adjusted from qualitative sample)  "
      f"IF={IF_FP_RATE:.0%}  LOF={LOF_FP_RATE:.0%}")

N_WAREHOUSES = 3
FP_COST = 10.0          # $10 per false positive (20 min × $30/hr)
SUPERVISOR_RATE = 30.0  # $/hr

# ── COMPUTE COST ─────────────────────────────────────────────────────────────
# Daily compute time estimates (seconds)
daily_compute_time = {
    "SPC (control charts + CUSUM)": 0.15,        # 0.1s charts + 0.05s run rules
    "Isolation Forest (warehouse)": 2.10,         # 2s fit + 0.1s score
    "LOF (warehouse)":              3.00,         # 1s × 3 warehouses
    "IF (operator level)":          15.00,        # 5s × 3 warehouses
    "Hybrid (SPC daily + IF weekly)": 0.15 + 2.10/7,  # daily SPC + IF once/week
}

records = []
for method, secs in daily_compute_time.items():
    monthly_compute_hours = secs * 30 / 3600
    monthly_compute_cost = monthly_compute_hours * 0.10  # $0.10/hr cloud compute

    # Determine FP rate for this method
    if "SPC" in method and "Hybrid" not in method:
        fp_rate = SPC_FP_RATE_ADJUSTED
        total_flags_annual = comp_df["spc_flag"].sum() / 2 * 12  # 2 years → monthly → annual
    elif "Isolation Forest" in method or ("IF" in method and "operator" not in method):
        fp_rate = IF_FP_RATE
        total_flags_annual = comp_df["if_flag"].sum() / 2 * 12
    elif "LOF" in method:
        fp_rate = LOF_FP_RATE
        total_flags_annual = comp_df["lof_flag"].sum() / 2 * 12
    elif "operator" in method.lower():
        fp_rate = IF_FP_RATE
        total_flags_annual = 182 * 12 / 2  # from step 3
    elif "Hybrid" in method:
        # SPC daily + IF weekly: FP = weighted blend
        fp_rate = (SPC_FP_RATE_ADJUSTED + IF_FP_RATE) / 2
        total_flags_annual = (comp_df["spc_flag"].sum() + comp_df["if_flag"].sum()) / 2 / 2 * 12

    annual_fp_investigations = total_flags_annual * fp_rate
    annual_fp_cost = annual_fp_investigations * FP_COST
    monthly_fp_cost = annual_fp_cost / 12

    annual_compute_cost = monthly_compute_cost * 12

    records.append({
        "method": method,
        "daily_compute_seconds": round(secs, 3),
        "monthly_compute_hours": round(monthly_compute_hours, 6),
        "monthly_compute_cost_usd": round(monthly_compute_cost, 4),
        "annual_compute_cost_usd": round(annual_compute_cost, 2),
        "fp_rate_estimate": fp_rate,
        "total_flags_annual": round(total_flags_annual, 0),
        "annual_fp_investigations": round(annual_fp_investigations, 0),
        "annual_fp_cost_usd": round(annual_fp_cost, 2),
        "monthly_fp_cost_usd": round(monthly_fp_cost, 2),
        "monthly_total_cost_usd": round(monthly_compute_cost + monthly_fp_cost, 2),
        "annual_total_cost_usd": round(annual_compute_cost + annual_fp_cost, 2),
    })

cost_df = pd.DataFrame(records)

# ── MAINTENANCE COMPLEXITY SCORES ─────────────────────────────────────────────
maintenance = {
    "SPC (control charts + CUSUM)":      {"retuning": 2, "explainability": 5, "fp_cost": 3, "avg": 3.3},
    "Isolation Forest (warehouse)":       {"retuning": 3, "explainability": 2, "fp_cost": 3, "avg": 2.7},
    "LOF (warehouse)":                    {"retuning": 3, "explainability": 2, "fp_cost": 3, "avg": 2.7},
    "IF (operator level)":                {"retuning": 3, "explainability": 2, "fp_cost": 3, "avg": 2.7},
    "Hybrid (SPC daily + IF weekly)":     {"retuning": 2, "explainability": 4, "fp_cost": 3, "avg": 3.0},
}

cost_df["maintenance_score"] = cost_df["method"].map(lambda m: maintenance.get(m, {}).get("avg", 0))

# ── VALUE OF CATCHING TRUE ANOMALY ───────────────────────────────────────────
BASELINE_ACCURACY = 0.98
TASKS_PER_DAY_PER_WH = 500
ACCURACY_DROP_PP = 0.02
DAYS_UNDETECTED = 7
COST_PER_ERROR = 25.0
INCIDENTS_PER_QUARTER_PER_WH = 1

extra_errors_per_incident = ACCURACY_DROP_PP * TASKS_PER_DAY_PER_WH * DAYS_UNDETECTED
cost_per_incident = extra_errors_per_incident * COST_PER_ERROR
annual_value_per_wh = cost_per_incident * INCIDENTS_PER_QUARTER_PER_WH * 4
total_annual_value = annual_value_per_wh * N_WAREHOUSES

print(f"\nValue of early detection:")
print(f"  Extra errors per undetected incident: {extra_errors_per_incident:.0f}")
print(f"  Cost per incident: ${cost_per_incident:,.0f}")
print(f"  Annual early-detection value (all warehouses): ${total_annual_value:,.0f}")

# ── IS ML WORTH IT? ───────────────────────────────────────────────────────────
spc_row = cost_df[cost_df["method"] == "SPC (control charts + CUSUM)"].iloc[0]
if_row  = cost_df[cost_df["method"] == "Isolation Forest (warehouse)"].iloc[0]

extra_ml_annual_cost = (if_row["annual_total_cost_usd"] - spc_row["annual_total_cost_usd"])
threshold = annual_value_per_wh  # per warehouse

print(f"\n── Is ML worth it? ──")
print(f"  SPC annual total cost:      ${spc_row['annual_total_cost_usd']:,.2f}")
print(f"  IF annual total cost:       ${if_row['annual_total_cost_usd']:,.2f}")
print(f"  Extra ML cost vs SPC:       ${extra_ml_annual_cost:,.2f}")
print(f"  Annual early-detection value (per warehouse): ${annual_value_per_wh:,.0f}")
if extra_ml_annual_cost < annual_value_per_wh:
    print(f"  VERDICT: YES — extra ML cost (${extra_ml_annual_cost:,.0f}) < detection value (${annual_value_per_wh:,.0f}) → ML adds value")
else:
    print(f"  VERDICT: MARGINAL — extra ML cost (${extra_ml_annual_cost:,.0f}) ≥ detection value (${annual_value_per_wh:,.0f})")
    print(f"  However, ML adds coverage for multi-feature anomalies — recommend hybrid as weekly diagnostic")

cost_df["early_detection_value_annual"] = annual_value_per_wh * N_WAREHOUSES
cost_df.to_csv(OUTPUT_DIR / "cost_benefit_analysis.csv", index=False)
print(f"\nExported cost_benefit_analysis.csv")

# ── FULL TABLE ────────────────────────────────────────────────────────────────
print("\n── Full Cost-Benefit Table ──")
display_cols = ["method", "daily_compute_seconds", "monthly_compute_cost_usd",
                "annual_compute_cost_usd", "fp_rate_estimate", "annual_fp_cost_usd",
                "annual_total_cost_usd", "maintenance_score"]
print(cost_df[display_cols].to_string(index=False))

# ── FIGURE ───────────────────────────────────────────────────────────────────
methods_short = ["SPC", "IF (WH)", "LOF (WH)", "IF (Ops)", "Hybrid"]
monthly_compute = cost_df["monthly_compute_cost_usd"].values
monthly_fp = cost_df["monthly_fp_cost_usd"].values

x = np.arange(len(methods_short))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, monthly_compute, width, label="Monthly Compute Cost ($)", color="steelblue", alpha=0.85)
bars2 = ax.bar(x + width/2, monthly_fp, width, label="Monthly FP Investigation Cost ($)", color="tomato", alpha=0.85)

ax.set_xlabel("Method")
ax.set_ylabel("Monthly Cost (USD)")
ax.set_title("WMS Anomaly Detection — Monthly Cost Comparison\n(Compute + False Positive Investigation)")
ax.set_xticks(x)
ax.set_xticklabels(methods_short, rotation=15, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar in bars1:
    h = bar.get_height()
    if h > 0.001:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'${h:.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
            f'${h:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "cost_benefit_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved cost_benefit_comparison.png")
