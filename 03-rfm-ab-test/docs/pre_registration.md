# Pre-Registration: At Risk Retention Campaign A/B Test
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
- **Population:** At Risk segment (95 customers, R≤2 AND F≥3)
- **n per group:** 47 (treatment) and 47 (control)
- **Randomisation:** Stratified by Customer_Type to ensure proportional balance
- **Assignment file:** outputs/test_assignments.csv (locked before test launch)
- **Random seed:** 42

## 5. Statistical Parameters
| Parameter | Value |
|---|---|
| α (significance level) | 0.05 |
| Power | 0.80 (80%) |
| Test type | Two-proportion z-test, two-tailed |
| Detectable MDE | 15.1 pp absolute lift |
| Baseline conversion rate | 83.2% |

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
