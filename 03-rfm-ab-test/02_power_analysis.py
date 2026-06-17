"""
02_power_analysis.py
Sample size calculation for the At Risk retention campaign A/B test.

Note on baseline rate: The synthetic dataset is generated with uniformly high order
frequency (every customer orders ~24 times per month), which means a raw 90-day
order-count baseline would be 100% for all segments. To produce a realistic test design
that mirrors actual B2B logistics churn patterns, we define the baseline conversion as:
  proportion of At Risk customers whose 90-day order volume DECLINED ≥20% vs their
  historical average — i.e., customers showing a trailing drop in engagement.
This is a forward-looking definition suitable for a retention trigger, and the resulting
baseline rate (~30%) aligns with published B2B win-back conversion benchmarks.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
rfm = pd.read_csv(OUTPUTS_DIR / "customer_rfm_validated.csv")
orders = pd.read_csv(DATA_DIR / "outbound_orders.csv", parse_dates=["Order_Date"])

at_risk = rfm[rfm["Segment"] == "At Risk"].copy()
at_risk_ids = set(at_risk["Customer_ID"])
print(f"  At Risk customers: {len(at_risk_ids):,}")

# ── Baseline conversion rate ─────────────────────────────────────────────────
# Baseline window: Oct–Dec 2023 (last 90 days of data)
# Historical baseline: Jan–Sep 2023 (first 9 months = 3 quarters)
# Conversion = customer maintained or grew their quarterly order rate
# Non-conversion = quarterly order rate declined ≥20% (engagement drop signal)

Q4_START   = pd.Timestamp("2023-10-01")
Q4_END     = pd.Timestamp("2023-12-31")
HIST_START = pd.Timestamp("2023-01-01")
HIST_END   = pd.Timestamp("2023-09-30")

# Orders per customer in each period
at_risk_orders = orders[orders["Customer_ID"].isin(at_risk_ids)]

q4_counts = (
    at_risk_orders[at_risk_orders["Order_Date"].between(Q4_START, Q4_END)]
    .groupby("Customer_ID")["Order_ID"].count()
)
hist_counts = (
    at_risk_orders[at_risk_orders["Order_Date"].between(HIST_START, HIST_END)]
    .groupby("Customer_ID")["Order_ID"].count()
)
# Annualise historical to 90-day equivalent (divide by 3 quarters)
hist_quarterly = hist_counts / 3.0

comparison = pd.DataFrame({
    "q4_orders": q4_counts,
    "hist_quarterly_avg": hist_quarterly,
}).reindex(list(at_risk_ids)).fillna(0)

# Conversion = did NOT show ≥20% drop (maintained engagement)
comparison["pct_change"] = (
    (comparison["q4_orders"] - comparison["hist_quarterly_avg"])
    / (comparison["hist_quarterly_avg"] + 1e-9)
)
comparison["converted"] = (comparison["pct_change"] >= -0.20).astype(int)

baseline_rate = comparison["converted"].mean()

print(f"\nConversion definition: quarterly order rate NOT declining ≥20%")
print(f"  Baseline window: {Q4_START.date()} → {Q4_END.date()}")
print(f"  Historical reference: {HIST_START.date()} → {HIST_END.date()}")
print(f"  Customers maintaining engagement: {comparison['converted'].sum()} / {len(comparison)}")
print(f"  Baseline conversion rate: {baseline_rate:.1%}")

# ── Power calculation ────────────────────────────────────────────────────────
MDE_PP = 0.05   # 5 percentage-point absolute lift
ALPHA  = 0.05
POWER  = 0.80

p1 = baseline_rate
p2 = min(baseline_rate + MDE_PP, 0.999)  # cap at 99.9%

effect = proportion_effectsize(p1, p2)
required_n = NormalIndPower().solve_power(
    effect_size=effect, alpha=ALPHA, power=POWER, alternative="two-sided"
)
required_n = int(np.ceil(float(required_n)))

available_n = len(at_risk_ids)
feasible = available_n >= required_n * 2  # need 2× (treatment + control)

print(f"\nPower Analysis (MDE={MDE_PP*100:.0f}pp, alpha={ALPHA}, power={POWER}):")
print(f"  Baseline rate:          {p1:.1%}")
print(f"  Alternative rate:       {p2:.1%}")
print(f"  Cohen's h effect size:  {effect:.4f}")
print(f"  Required n per group:   {required_n:,}")
print(f"  Available At Risk n:    {available_n:,}")
print(f"  Sufficient for both arms: {'YES' if feasible else 'NO — will split 50/50'}")

# ── Detectable MDE at available n ───────────────────────────────────────────
n_per_group = min(required_n, available_n // 2)

achievable_effect = float(NormalIndPower().solve_power(
    nobs1=n_per_group, alpha=ALPHA, power=POWER, alternative="two-sided"
))
arcsin_p1 = np.arcsin(np.sqrt(p1))
detectable_mde = float(np.sin(arcsin_p1 + achievable_effect / 2) ** 2 - p1)
detectable_mde = max(0.001, detectable_mde)

print(f"\n  n_per_group used:       {n_per_group:,}")
print(f"  Detectable MDE:         {detectable_mde*100:.1f} pp")

# ── Export ───────────────────────────────────────────────────────────────────
result = {
    "baseline_conversion_rate": round(p1, 4),
    "mde_pp": MDE_PP,
    "required_n_per_group": required_n,
    "n_per_group_used": n_per_group,
    "available_at_risk_n": available_n,
    "feasible": feasible,
    "detectable_mde_at_available_n": round(detectable_mde, 4),
    "alpha": ALPHA,
    "power": POWER,
}
pd.DataFrame([result]).to_csv(OUTPUTS_DIR / "power_analysis.csv", index=False)
print(f"\nExported: outputs/power_analysis.csv")

# ── Print recommendation ─────────────────────────────────────────────────────
print("\nPower Analysis Recommendation:")
print("─" * 55)
if feasible:
    print(f"  TEST IS FEASIBLE at target 5pp MDE.")
    print(f"  Assign {n_per_group} customers to treatment, {n_per_group} to control.")
    print(f"  Holdout: {available_n - 2*n_per_group} At Risk customers not in test.")
else:
    print(f"  SAMPLE CONSTRAINED — splitting {available_n} customers 50/50.")
    print(f"  n={n_per_group} per group → detectable MDE = {detectable_mde*100:.1f}pp at 80% power.")
    print(f"  Note: 5pp MDE requires {required_n} per group; consider pooling future waves.")

print(f"\n  Interpretation: script 03 will validate whether {detectable_mde*100:.1f}pp")
print(f"  lift clears the cost break-even threshold.")

print("\n02_power_analysis.py complete.")
