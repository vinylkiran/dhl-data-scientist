"""
04_test_design.py
Formalises the pre-registered A/B test specification for the At Risk
retention campaign. Generates stratified treatment/control assignments
and writes the pre-registration document.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR    = BASE_DIR / "docs"

np.random.seed(42)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
rfm       = pd.read_csv(OUTPUTS_DIR / "customer_rfm_validated.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")
power     = pd.read_csv(OUTPUTS_DIR / "power_analysis.csv")

n_per_group   = int(power["n_per_group_used"].iloc[0])
baseline_rate = float(power["baseline_conversion_rate"].iloc[0])
mde           = float(power["detectable_mde_at_available_n"].iloc[0])

at_risk = rfm[rfm["Segment"] == "At Risk"].merge(
    customers[["Customer_ID", "Customer_Type", "Region"]],
    on="Customer_ID", how="left"
)

print(f"  At Risk customers: {len(at_risk)}")
print(f"  n per group:       {n_per_group}")

# ── Stratified randomisation by Customer_Type ────────────────────────────────
# Justification: Customer_Type (e.g. E-Commerce, B2B, 3PL) drives different
# baseline conversion rates. Simple random assignment risks imbalanced strata,
# inflating Type I error for subgroup analyses. Stratification ensures both arms
# are proportionally balanced on this key confounder.

print("\nCustomer_Type distribution in At Risk segment:")
type_counts = at_risk["Customer_Type"].value_counts()
print(type_counts.to_string())

assignments = []
for ctype, group in at_risk.groupby("Customer_Type"):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    n_type     = len(group)
    n_treat    = int(round(n_type * (n_per_group / len(at_risk))))
    n_treat    = min(n_treat, n_type)
    n_ctrl     = min(n_treat, n_type - n_treat)
    treat_ids  = group.iloc[:n_treat]["Customer_ID"].tolist()
    ctrl_ids   = group.iloc[n_treat:n_treat + n_ctrl]["Customer_ID"].tolist()
    for cid in treat_ids:
        assignments.append({"Customer_ID": cid, "Group": "Treatment", "Customer_Type": ctype})
    for cid in ctrl_ids:
        assignments.append({"Customer_ID": cid, "Group": "Control",   "Customer_Type": ctype})

assignments_df = pd.DataFrame(assignments)

# Re-balance to exactly n_per_group per arm if needed
treat_df = assignments_df[assignments_df["Group"] == "Treatment"].head(n_per_group)
ctrl_df  = assignments_df[assignments_df["Group"] == "Control"].head(n_per_group)
assignments_df = pd.concat([treat_df, ctrl_df], ignore_index=True)

print(f"\nRandomisation result:")
print(f"  Treatment arm: {len(treat_df)}")
print(f"  Control arm:   {len(ctrl_df)}")
print(f"\nCustomer_Type balance check:")
balance = assignments_df.groupby(["Group", "Customer_Type"]).size().unstack(fill_value=0)
print(balance.to_string())

# ── Export test assignments ──────────────────────────────────────────────────
assignments_df.to_csv(OUTPUTS_DIR / "test_assignments.csv", index=False)
print(f"\nExported: outputs/test_assignments.csv ({len(assignments_df)} rows)")

# ── Test design spec ─────────────────────────────────────────────────────────
spec = pd.DataFrame([{
    "primary_metric":            "conversion_90d",
    "primary_test":              "two_proportion_z_test_two_tailed",
    "alpha":                     0.05,
    "power":                     0.80,
    "n_per_group":               n_per_group,
    "randomisation":             "stratified_by_Customer_Type",
    "horizon_days":              90,
    "early_stopping":            "NONE — fixed horizon",
    "guardrail_aov":             "t_test_must_not_decrease_>5pct",
    "guardrail_unsubscribe":     "monitor_2pct_baseline",
    "guardrail_otif":            "monitor_only_not_in_data",
    "multiple_comp_correction":  "Bonferroni_if_guardrails_tested",
    "baseline_conversion_rate":  round(baseline_rate, 4),
    "detectable_mde_pp":         round(mde, 4),
}])
spec.to_csv(OUTPUTS_DIR / "test_design_spec.csv", index=False)
print("Exported: outputs/test_design_spec.csv")

# ── Pre-registration document ────────────────────────────────────────────────
pre_reg = f"""# Pre-Registration: At Risk Retention Campaign A/B Test
**Date registered:** 2023-12-31 (before test execution)
**Author:** DHL Data Science
**Status:** Pre-registered — do not analyse until 90-day horizon completes

---

## 1. Study Question
Does a targeted retention intervention (personalised email + account manager outreach)
increase the 90-day engagement rate of At Risk customers, compared to no intervention?

## 2. Hypothesis
- **H₀:** p_treatment = p_control (no difference in conversion rates)
- **H₁:** p_treatment ≠ p_control (two-tailed)

## 3. Primary Metric
**Conversion within 90 days** — defined as the customer's quarterly order volume
not declining by ≥20% relative to their prior 9-month average.
Analysed via two-proportion z-test (two-tailed).

## 4. Sample & Randomisation
- **Population:** At Risk segment ({len(at_risk)} customers, R≤2 AND F≥3)
- **n per group:** {n_per_group} (treatment) and {n_per_group} (control)
- **Randomisation:** Stratified by Customer_Type to ensure proportional balance
- **Assignment file:** outputs/test_assignments.csv (locked before test launch)
- **Random seed:** 42

## 5. Statistical Parameters
| Parameter | Value |
|---|---|
| α (significance level) | 0.05 |
| Power | 0.80 (80%) |
| Test type | Two-proportion z-test, two-tailed |
| Detectable MDE | {mde*100:.1f} pp absolute lift |
| Baseline conversion rate | {baseline_rate:.1%} |

## 6. Guardrail Metrics
| Metric | Test | Threshold |
|---|---|---|
| Average order value (AOV) | Welch's t-test | Must not decrease by >5% |
| Unsubscribe / opt-out rate | Monitoring | Baseline ~2%; flag if >4% |
| OTIF complaint rate | Monitoring only | Not in dataset; flag operationally |

## 7. Stopping Rule
**Fixed horizon only.** No interim analyses. No early stopping for efficacy or futility.
All 90 days of post-assignment data must be collected before any analysis.
Rationale: prevents inflated Type I error from repeated testing without correction.

## 8. Multiple Comparisons
If guardrail metrics are tested alongside the primary metric:
Apply **Bonferroni correction** (α/k where k = number of simultaneous tests).
Primary metric alone: α = 0.05 (no correction needed).

## 9. Analysis Plan
1. Load test_assignments.csv (locked).
2. Compute conversion for each customer in the 90-day post-assignment window.
3. Run proportions_ztest (statsmodels).
4. Compute 95% CI for the difference in proportions.
5. Check guardrail metrics.
6. Report p-value, CI, effect size, post-hoc power.
7. Feed results into business_decision.py for ROI assessment.

## 10. What Would Invalidate the Test
- Contamination between treatment and control (e.g., shared account managers)
- External campaign targeting At Risk customers during the test window
- Data pipeline failure producing missing orders for any customer
- Material change in DHL pricing or contract terms mid-test

---
*This document was written and locked before any outcome data was inspected.*
"""

with open(DOCS_DIR / "pre_registration.md", "w") as f:
    f.write(pre_reg)
print("Exported: docs/pre_registration.md")

# ── Print design summary ─────────────────────────────────────────────────────
print(f"""
Test Design Summary
─────────────────────────────────────────────────────────
  Population:         At Risk segment ({len(at_risk)} customers)
  Randomisation:      Stratified by Customer_Type
  Arms:               Treatment (n={n_per_group}) / Control (n={n_per_group})
  Primary metric:     Conversion within 90 days
  Statistical test:   Two-proportion z-test, two-tailed, α=0.05
  Horizon:            90 days (fixed, no early stopping)
  Detectable MDE:     {mde*100:.1f}pp at 80% power
  Guardrails:         AOV (t-test, -5% threshold), unsubscribe, OTIF
  Multiple comp.:     Bonferroni if guardrails tested jointly
""")

print("04_test_design.py complete.")
