"""
05_run_analysis.py
Execute the pre-registered A/B test analysis using simulated outcome data.
Treatment group converts at (baseline + MDE + noise), control at baseline.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

np.random.seed(42)

# ── Load inputs ──────────────────────────────────────────────────────────────
print("Loading data...")
assignments = pd.read_csv(OUTPUTS_DIR / "test_assignments.csv")
power       = pd.read_csv(OUTPUTS_DIR / "power_analysis.csv")
rfm         = pd.read_csv(OUTPUTS_DIR / "customer_rfm_validated.csv")
orders      = pd.read_csv(DATA_DIR / "outbound_orders.csv", parse_dates=["Order_Date"])

baseline_rate = float(power["baseline_conversion_rate"].iloc[0])
mde           = float(power["detectable_mde_at_available_n"].iloc[0])

treatment = assignments[assignments["Group"] == "Treatment"].copy()
control   = assignments[assignments["Group"] == "Control"].copy()
n_treatment = len(treatment)
n_control   = len(control)

print(f"  Treatment: n={n_treatment}, Control: n={n_control}")
print(f"  Baseline conversion rate: {baseline_rate:.1%}")
print(f"  Simulated lift: {mde:.4f} ({mde*100:.1f}pp)")

# ── Simulate outcomes ────────────────────────────────────────────────────────
# Treatment: converts at baseline + MDE + N(0, 0.01) noise (clipped)
treat_rate = np.clip(baseline_rate + mde + np.random.normal(0, 0.01), 0.01, 0.999)
ctrl_rate  = baseline_rate

treat_conversions = int(round(treat_rate * n_treatment))
ctrl_conversions  = int(round(ctrl_rate  * n_control))

print(f"\nSimulated outcomes:")
print(f"  Treatment: {treat_conversions}/{n_treatment} converted ({treat_conversions/n_treatment:.1%})")
print(f"  Control:   {ctrl_conversions}/{n_control} converted ({ctrl_conversions/n_control:.1%})")
print(f"  True simulated treatment rate: {treat_rate:.4f}")

# ── Primary test: two-proportion z-test ─────────────────────────────────────
count = np.array([treat_conversions, ctrl_conversions])
nobs  = np.array([n_treatment, n_control])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")

p_treat = treat_conversions / n_treatment
p_ctrl  = ctrl_conversions  / n_control
diff    = p_treat - p_ctrl

# 95% CI for difference in proportions (manual)
se_diff = np.sqrt(
    p_treat * (1 - p_treat) / n_treatment +
    p_ctrl  * (1 - p_ctrl)  / n_control
)
ci_lo = diff - 1.96 * se_diff
ci_hi = diff + 1.96 * se_diff

significant = pval < 0.05

print(f"\nPrimary Test Results:")
print(f"  Z-statistic:          {stat:.4f}")
print(f"  p-value:              {pval:.4f}")
print(f"  Significant (α=0.05): {'YES' if significant else 'NO'}")
print(f"  Absolute lift:        {diff*100:.2f}pp")
print(f"  Relative lift:        {diff/p_ctrl*100:.1f}%")
print(f"  95% CI:               [{ci_lo*100:.2f}pp, {ci_hi*100:.2f}pp]")

# ── Guardrail: AOV check ─────────────────────────────────────────────────────
# Simulate AOV for converters: sample from historical At Risk AOV + N(0, 5) noise
at_risk_ids = set(rfm[rfm["Segment"] == "At Risk"]["Customer_ID"])
at_risk_orders = orders[orders["Customer_ID"].isin(at_risk_ids)]
historical_aov = at_risk_orders["Revenue"].values  # per-order revenue

n_treat_conv = treat_conversions
n_ctrl_conv  = ctrl_conversions

aov_treatment = np.random.choice(historical_aov, size=n_treat_conv, replace=True) + np.random.normal(0, 5, n_treat_conv)
aov_control   = np.random.choice(historical_aov, size=n_ctrl_conv,  replace=True) + np.random.normal(0, 5, n_ctrl_conv)

mean_aov_treat = aov_treatment.mean()
mean_aov_ctrl  = aov_control.mean()
aov_pct_change = (mean_aov_treat - mean_aov_ctrl) / mean_aov_ctrl

t_stat_aov, pval_aov = stats.ttest_ind(aov_treatment, aov_control, equal_var=False)
guardrail_aov_pass = aov_pct_change >= -0.05  # must not decrease >5%

print(f"\nGuardrail Check — Average Order Value:")
print(f"  Treatment AOV: ${mean_aov_treat:,.0f}")
print(f"  Control AOV:   ${mean_aov_ctrl:,.0f}")
print(f"  Change:        {aov_pct_change*100:.2f}%  (threshold: must not decrease >5%)")
print(f"  t-statistic:   {t_stat_aov:.4f}, p={pval_aov:.4f}")
print(f"  Guardrail:     {'PASS' if guardrail_aov_pass else 'FAIL'}")

# ── Post-hoc power ───────────────────────────────────────────────────────────
observed_effect = proportion_effectsize(p_treat, p_ctrl)
if abs(observed_effect) > 1e-6:
    post_hoc_power = NormalIndPower().solve_power(
        effect_size=abs(observed_effect),
        nobs1=n_treatment,
        alpha=0.05,
        alternative="two-sided"
    )
    post_hoc_power = float(post_hoc_power)
else:
    post_hoc_power = 0.0

print(f"\nPost-Hoc Power:")
print(f"  Observed effect (Cohen's h): {observed_effect:.4f}")
print(f"  Post-hoc power:              {post_hoc_power:.3f}")
if post_hoc_power < 0.80:
    print(f"  Note: Power below 0.80 — treat results with caution.")

# ── Export results ───────────────────────────────────────────────────────────
results = pd.DataFrame([{
    "n_treatment":             n_treatment,
    "n_control":               n_control,
    "treatment_conversions":   treat_conversions,
    "control_conversions":     ctrl_conversions,
    "p_treatment":             round(p_treat, 4),
    "p_control":               round(p_ctrl,  4),
    "absolute_lift_pp":        round(diff, 4),
    "relative_lift_pct":       round(diff / p_ctrl * 100, 2),
    "ci_lo_pp":                round(ci_lo, 4),
    "ci_hi_pp":                round(ci_hi, 4),
    "z_stat":                  round(stat, 4),
    "p_value":                 round(pval, 4),
    "significant":             significant,
    "guardrail_aov_pass":      guardrail_aov_pass,
    "aov_pct_change":          round(aov_pct_change, 4),
    "post_hoc_power":          round(post_hoc_power, 3),
    "baseline_rate":           round(baseline_rate, 4),
    "simulated_mde":           round(mde, 4),
}])
results.to_csv(OUTPUTS_DIR / "test_results.csv", index=False)
print(f"\nExported: outputs/test_results.csv")

# ── Figure: conversion rates with CI error bars ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

groups    = ["Control", "Treatment"]
rates     = [p_ctrl, p_treat]
ci_errors = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in [(p_ctrl, n_control), (p_treat, n_treatment)]]
colors    = ["#90A4AE", "#1565C0"]

bars = ax.bar(groups, [r * 100 for r in rates], color=colors,
              yerr=[e * 100 for e in ci_errors],
              capsize=8, error_kw={"elinewidth": 2, "ecolor": "#333"},
              width=0.5, edgecolor="white")

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.5,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title(
    f"A/B Test: At Risk Retention Campaign\n"
    f"p={pval:.4f}  {'★ Significant' if significant else '✗ Not significant'}  |  "
    f"Lift: {diff*100:.1f}pp  95%CI [{ci_lo*100:.1f}, {ci_hi*100:.1f}]",
    fontsize=11, pad=14
)
ax.set_ylabel("Conversion Rate (%)")
ax.set_ylim(0, max(p_treat, p_ctrl) * 100 + 15)
ax.axhline(baseline_rate * 100, color="gray", linestyle="--", linewidth=1, label=f"Baseline {baseline_rate:.1%}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "test_results_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/test_results_summary.png")

# ── Print full summary ───────────────────────────────────────────────────────
print(f"""
Full Test Result Table
─────────────────────────────────────────────────────────
  Test:          Two-proportion z-test (two-tailed)
  n:             {n_treatment} treatment, {n_control} control
  Conversions:   {treat_conversions} treatment ({p_treat:.1%}), {ctrl_conversions} control ({p_ctrl:.1%})
  Absolute lift: {diff*100:.2f}pp
  Relative lift: {diff/p_ctrl*100:.1f}%
  95% CI:        [{ci_lo*100:.2f}pp, {ci_hi*100:.2f}pp]
  Z-statistic:   {stat:.4f}
  p-value:       {pval:.4f}
  Significant:   {'YES (p < 0.05)' if significant else 'NO (p ≥ 0.05)'}

  Guardrail — AOV: {'PASS' if guardrail_aov_pass else 'FAIL'} ({aov_pct_change*100:.2f}% change, threshold -5%)
  Post-hoc power:  {post_hoc_power:.3f}

  Conclusion: {'The retention campaign produced a statistically significant lift.' if significant else 'No statistically significant lift detected.'}
  {'CI excludes zero, confirming a real effect at α=0.05.' if significant and ci_lo > 0 else ''}
""")

print("05_run_analysis.py complete.")
