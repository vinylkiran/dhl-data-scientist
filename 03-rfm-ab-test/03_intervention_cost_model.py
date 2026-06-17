"""
03_intervention_cost_model.py
Campaign cost model for the At Risk retention A/B test.
Determines economic viability and break-even lift threshold.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
rfm       = pd.read_csv(OUTPUTS_DIR / "customer_rfm_validated.csv")
power     = pd.read_csv(OUTPUTS_DIR / "power_analysis.csv")
orders    = pd.read_csv(DATA_DIR / "outbound_orders.csv", parse_dates=["Order_Date"])

at_risk = rfm[rfm["Segment"] == "At Risk"].copy()
at_risk_ids = set(at_risk["Customer_ID"])

n_treatment  = int(power["n_per_group_used"].iloc[0])
total_at_risk = int(power["available_at_risk_n"].iloc[0])
baseline_rate = float(power["baseline_conversion_rate"].iloc[0])
detectable_mde = float(power["detectable_mde_at_available_n"].iloc[0])

print(f"  At Risk population: {total_at_risk}")
print(f"  n per group (A/B split): {n_treatment}")
print(f"  Baseline conversion rate: {baseline_rate:.1%}")
print(f"  Detectable MDE: {detectable_mde*100:.1f}pp")

# ── Campaign cost parameters ─────────────────────────────────────────────────
COST_PER_CONTACT = 8.00    # $8 per customer (email + account manager time)
CAC_BENCHMARK    = 150.00  # $150 B2B logistics industry CAC (context only)

cost_test_treatment  = COST_PER_CONTACT * n_treatment
cost_full_rollout    = COST_PER_CONTACT * total_at_risk

print(f"\nCampaign Cost Parameters:")
print(f"  Cost per contact:         ${COST_PER_CONTACT:.2f}")
print(f"  B2B CAC benchmark:        ${CAC_BENCHMARK:.0f} (industry reference)")
print(f"  Test campaign cost:       ${cost_test_treatment:,.0f}  ({n_treatment} treatment)")
print(f"  Full rollout cost:        ${cost_full_rollout:,.0f}  ({total_at_risk} customers)")

# ── Expected Average Order Value ─────────────────────────────────────────────
# AOV = mean Revenue per order for At Risk customers in historical data
at_risk_orders = orders[orders["Customer_ID"].isin(at_risk_ids)]
expected_aov = at_risk_orders["Revenue"].mean()
print(f"\nExpected AOV (At Risk historical): ${expected_aov:,.2f} per order")

# ── Break-even lift ──────────────────────────────────────────────────────────
# Break-even lift = how many pp improvement covers full rollout cost?
# incremental_orders = total_at_risk × lift
# incremental_revenue = incremental_orders × expected_aov
# Set incremental_revenue = full_rollout_cost → lift = cost / (expected_aov × total_at_risk)
break_even_lift = cost_full_rollout / (expected_aov * total_at_risk)

print(f"\nBreak-Even Lift Calculation:")
print(f"  Full rollout cost:        ${cost_full_rollout:,.0f}")
print(f"  Expected AOV:             ${expected_aov:,.2f}")
print(f"  Break-even lift:          {break_even_lift*100:.3f}pp")
print(f"  Break-even vs detectable: {'WELL BELOW detectable MDE' if break_even_lift < detectable_mde else 'EXCEEDS detectable MDE — check ROI'}")

# ── Three-scenario analysis ──────────────────────────────────────────────────
mde = detectable_mde
scenarios = {
    "Pessimistic (0.5× MDE)": 0.5 * mde,
    "Realistic (MDE)":          mde,
    "Optimistic (2× MDE)":     2.0 * mde,
}

rows = []
print(f"\nScenario Analysis (full rollout of {total_at_risk} At Risk customers):")
print(f"{'Scenario':<30} {'Lift':>7} {'Incr.Orders':>12} {'Incr.Rev':>14} {'Net Rev':>14} {'ROI':>8}")
print("─" * 90)

for name, lift in scenarios.items():
    incremental_orders  = total_at_risk * lift
    incremental_revenue = incremental_orders * expected_aov
    net_revenue         = incremental_revenue - cost_full_rollout
    roi                 = net_revenue / cost_full_rollout if cost_full_rollout > 0 else 0

    print(f"{name:<30} {lift*100:>6.1f}pp {incremental_orders:>12.1f} "
          f"${incremental_revenue:>13,.0f} ${net_revenue:>13,.0f} {roi*100:>7.0f}%")
    rows.append({
        "scenario":             name,
        "lift_pp":              round(lift, 4),
        "n_at_risk":            total_at_risk,
        "incremental_orders":   round(incremental_orders, 1),
        "expected_aov":         round(expected_aov, 2),
        "incremental_revenue":  round(incremental_revenue, 2),
        "campaign_cost":        round(cost_full_rollout, 2),
        "net_revenue":          round(net_revenue, 2),
        "roi_pct":              round(roi * 100, 1),
    })

print("─" * 90)
print(f"\nBreak-even lift: {break_even_lift*100:.3f}pp  |  "
      f"Detectable MDE: {mde*100:.1f}pp  |  "
      f"MDE clears break-even: {'YES' if mde > break_even_lift else 'NO'}")

# ── Export ───────────────────────────────────────────────────────────────────
cost_df = pd.DataFrame(rows)
# Add summary fields
cost_df["cost_per_contact"]  = COST_PER_CONTACT
cost_df["break_even_lift_pp"] = round(break_even_lift, 4)
cost_df["baseline_rate"]      = baseline_rate
cost_df["cac_benchmark"]      = CAC_BENCHMARK

cost_df.to_csv(OUTPUTS_DIR / "cost_model.csv", index=False)
print(f"\nExported: outputs/cost_model.csv ({len(cost_df)} scenarios)")

print(f"""
Cost Model Summary
──────────────────────────────────────────────────────
  Campaign investment (full rollout): ${cost_full_rollout:,.0f}
  Cost per customer:                  ${COST_PER_CONTACT:.2f}
  Expected AOV (At Risk):             ${expected_aov:,.0f}
  Break-even lift required:           {break_even_lift*100:.3f}pp (negligible)
  Detectable MDE at n={n_treatment}:         {mde*100:.1f}pp
  At realistic lift ({mde*100:.1f}pp), ROI:   {rows[1]['roi_pct']:.0f}%
  Conclusion: Campaign is highly cost-efficient — low cost-per-contact
  ($8) relative to high AOV (${expected_aov:,.0f}) means break-even is trivial.
""")

print("03_intervention_cost_model.py complete.")
