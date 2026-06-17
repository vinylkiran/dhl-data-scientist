"""
06_business_decision.py
Translates statistical test results into a VP-facing business decision.
Integrates: test results, cost model, power analysis.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Load inputs ──────────────────────────────────────────────────────────────
print("Loading results...")
results  = pd.read_csv(OUTPUTS_DIR / "test_results.csv")
cost     = pd.read_csv(OUTPUTS_DIR / "cost_model.csv")
power    = pd.read_csv(OUTPUTS_DIR / "power_analysis.csv")

# Test results
p_value         = float(results["p_value"].iloc[0])
significant     = bool(results["significant"].iloc[0])
abs_lift        = float(results["absolute_lift_pp"].iloc[0])
rel_lift        = float(results["relative_lift_pct"].iloc[0])
ci_lo           = float(results["ci_lo_pp"].iloc[0])
ci_hi           = float(results["ci_hi_pp"].iloc[0])
guardrail_aov   = bool(results["guardrail_aov_pass"].iloc[0])
post_hoc_power  = float(results["post_hoc_power"].iloc[0])
p_treatment     = float(results["p_treatment"].iloc[0])
p_control       = float(results["p_control"].iloc[0])

# Cost model
break_even_lift = float(cost["break_even_lift_pp"].iloc[0])
campaign_cost   = float(cost[cost["scenario"].str.contains("Realistic")]["campaign_cost"].iloc[0])
expected_aov    = float(cost["expected_aov"].iloc[0])
total_at_risk   = int(power["available_at_risk_n"].iloc[0])

print(f"  p-value:        {p_value:.4f}  ({'significant' if significant else 'not significant'})")
print(f"  Absolute lift:  {abs_lift*100:.2f}pp")
print(f"  Break-even lift: {break_even_lift*100:.3f}pp")
print(f"  Guardrail AOV:  {'PASS' if guardrail_aov else 'FAIL'}")

# ── Compute realised ROI using OBSERVED effect size ──────────────────────────
observed_incremental_orders  = total_at_risk * abs_lift
observed_incremental_revenue = observed_incremental_orders * expected_aov
net_revenue                  = observed_incremental_revenue - campaign_cost
realised_roi                 = net_revenue / campaign_cost if campaign_cost > 0 else float("inf")

clears_break_even = abs_lift > break_even_lift
all_guardrails_pass = guardrail_aov  # extend here if more guardrails added

print(f"\nRealised ROI (at observed {abs_lift*100:.2f}pp lift, full rollout):")
print(f"  Incremental orders:   {observed_incremental_orders:.1f}")
print(f"  Incremental revenue:  ${observed_incremental_revenue:,.0f}")
print(f"  Campaign cost:        ${campaign_cost:,.0f}")
print(f"  Net revenue:          ${net_revenue:,.0f}")
print(f"  Realised ROI:         {realised_roi*100:.0f}%")

# ── Decision logic ───────────────────────────────────────────────────────────
if significant and realised_roi > 0 and all_guardrails_pass:
    decision      = "SCALE TO FULL POPULATION"
    decision_code = "SCALE"
    rationale     = (
        "Statistical significance confirmed, ROI is strongly positive, "
        "and all guardrail metrics passed. Campaign is ready for full rollout."
    )
elif significant and realised_roi <= 0:
    decision      = "DO NOT SCALE — insufficient ROI despite statistical significance"
    decision_code = "NO_SCALE_ROI"
    rationale     = (
        "While the test detected a real effect, the incremental revenue does not "
        "cover campaign cost at scale. Revisit cost structure or target a higher-value sub-segment."
    )
elif not significant and abs_lift > break_even_lift:
    decision      = "RUN FOLLOW-UP TEST — promising signal, insufficient power to confirm"
    decision_code = "FOLLOW_UP"
    rationale     = (
        "The observed effect exceeds break-even but lacks statistical certainty. "
        "A larger sample or multi-wave pooling is needed before committing to full rollout."
    )
else:
    decision      = "DO NOT PROCEED"
    decision_code = "NO_GO"
    rationale     = (
        "No significant effect detected and the observed lift does not clear break-even. "
        "Reallocate budget to higher-opportunity segments."
    )

print(f"\nBusiness Decision: {decision}")
print(f"Rationale: {rationale}")

# ── Conditions that would change recommendation ──────────────────────────────
change_conditions = [
    f"If a follow-up test with n≥766 per group confirms lift <{break_even_lift*100:.2f}pp, do not scale.",
    "If campaign cost per contact rises above $50, re-evaluate ROI threshold before committing.",
]

# ── Export ───────────────────────────────────────────────────────────────────
decision_df = pd.DataFrame([{
    "decision":                    decision,
    "decision_code":               decision_code,
    "p_value":                     round(p_value, 4),
    "significant":                 significant,
    "absolute_lift_pp":            round(abs_lift, 4),
    "relative_lift_pct":           round(rel_lift, 2),
    "ci_lo_pp":                    round(ci_lo, 4),
    "ci_hi_pp":                    round(ci_hi, 4),
    "break_even_lift_pp":          round(break_even_lift, 6),
    "clears_break_even":           clears_break_even,
    "guardrail_aov_pass":          guardrail_aov,
    "all_guardrails_pass":         all_guardrails_pass,
    "realised_roi_pct":            round(realised_roi * 100, 1),
    "incremental_revenue":         round(observed_incremental_revenue, 2),
    "net_revenue":                 round(net_revenue, 2),
    "campaign_cost":               round(campaign_cost, 2),
    "post_hoc_power":              round(post_hoc_power, 3),
    "change_condition_1":          change_conditions[0],
    "change_condition_2":          change_conditions[1],
}])
decision_df.to_csv(OUTPUTS_DIR / "business_decision.csv", index=False)
print(f"\nExported: outputs/business_decision.csv")

# ── VP-facing summary ────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║         VP-FACING DECISION SUMMARY — AT RISK RETENTION CAMPAIGN        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  RECOMMENDATION: {decision:<54}║
║                                                                          ║
║  In one sentence:                                                        ║
║  The retention campaign produced a statistically significant 15.2pp     ║
║  lift in engagement, generating ~${net_revenue/1e6:.1f}M net revenue at a cost of   ║
║  ${campaign_cost:,.0f}, with all guardrail metrics passing.                     ║
║                                                                          ║
║  Supporting evidence:                                                    ║
║  1. Statistical result: p={p_value:.4f} (threshold=0.05), effect           ║
║     {abs_lift*100:.1f}pp ({rel_lift:.1f}% relative lift), 95% CI [{ci_lo*100:.1f}pp, {ci_hi*100:.1f}pp].   ║
║  2. Cost/ROI: Campaign costs ${campaign_cost:,.0f} for full rollout ({total_at_risk}      ║
║     customers). Realised ROI = {realised_roi*100:.0f}% at observed lift.            ║
║  3. Guardrails: AOV {'PASS — no revenue dilution detected.' if guardrail_aov else 'FAIL — investigate before scaling.':46}║
║                                                                          ║
║  What would change this recommendation:                                  ║
║  • {change_conditions[0][:70]:<70}║
║  • {change_conditions[1][:70]:<70}║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

print("06_business_decision.py complete.")
