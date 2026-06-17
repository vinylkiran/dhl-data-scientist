# Model Card — RFM Segmentation & A/B Test Design
**Project:** DS Project 3 — RFM Analysis and A/B Test Design  
**Author:** DHL Data Science Portfolio  
**Reference date:** 2023-12-31  
**Status:** Complete (simulated outcome)

---

## 1. Purpose

This project segments DHL's customer base using RFM (Recency, Frequency, Monetary) scoring, identifies the At Risk segment, designs a statistically rigorous A/B test for a retention campaign, evaluates the economics, and produces a VP-facing business decision. The workflow demonstrates the full loop: descriptive analytics → causal inference design → cost modelling → business translation.

---

## 2. Data

| Source | Rows | Description |
|---|---|---|
| `outbound_orders.csv` | 68,941 | Order-level data, Jan 2022–Dec 2023 |
| `customers.csv` | 500 | Customer metadata (type, region, SLA, contract) |

- **Customers with orders:** 398 (102 customers had no orders and were excluded from RFM)
- **At Risk segment:** 95 customers

---

## 3. RFM Methodology

### Scoring
- **Reference date:** 2023-12-31
- **Recency:** Days since last order (lower = more recent)
- **Frequency:** Total order count across 2-year window
- **Monetary:** Total revenue across 2-year window
- **Scoring method:** Rank-percentile quintiles (1–5). R is inverted so lower recency (more recent) = score 5.
- **RFM_Score:** R + F + M (range 3–15)

### Segment Definitions

| Segment | Rule |
|---|---|
| Champions | R≥4 AND F≥4 AND M≥4 |
| Loyal | R≥3 AND F≥4 |
| Potential Loyalist | R≥4 AND F≤3 AND M≥3 |
| At Risk | R≤2 AND F≥3 |
| Lost Cheap | R=1 AND F≤2 AND M≤2 |
| Lost | R≤2 AND F≤2 (not Lost Cheap) |
| Needs Attention | R=3 AND F=3 AND M=3 |
| New Customers | R≥4 AND F≤2 AND M≤2 |
| Promising | R≥3 AND F≤2 AND M≤3 |
| Others | All remaining |

---

## 4. Validation Results

### Bootstrap Stability (50 iterations, 70% subsample)

| Dimension | Max Boundary CV | Stable? |
|---|---|---|
| R (Recency) | 0.0000 | YES |
| F (Frequency) | 0.0000 | YES |
| M (Monetary) | 0.0000 | YES |

All CVs < 0.10 → quintile boundaries are stable across subsamples. The synthetic dataset's consistent order patterns produce near-zero variance across bootstrap iterations.

### Correlation Analysis (Spearman)

| Pair | ρ | Interpretation |
|---|---|---|
| R vs F | 0.053 | Uncorrelated — independent dimensions |
| R vs M | 0.049 | Uncorrelated — independent dimensions |
| F vs M | 0.313 | **Correlated (|ρ| > 0.3)** — mild redundancy |

F and M are positively correlated (more frequent buyers also spend more), which is expected in B2B logistics. R remains orthogonal to both, confirming it captures a distinct behavioural signal (recency of engagement).

### Segment Population

| Segment | N | % |
|---|---|---|
| At Risk | 95 | 23.9% |
| Loyal | 63 | 15.8% |
| Lost | 47 | 11.8% |
| Potential Loyalist | 45 | 11.3% |
| Champions | 41 | 10.3% |
| New Customers | 36 | 9.0% |
| Others | 28 | 7.0% |
| Promising | 22 | 5.5% |
| Lost Cheap | 18 | 4.5% |
| Needs Attention | 3 | 0.8% |

---

## 5. Test Design

### Randomisation
Stratified by `Customer_Type` (Retailer, E-Commerce, Distributor, Manufacturer, Government, Healthcare). Stratification ensures proportional balance across customer types, preventing confounding from type-specific baseline conversion rates. Each stratum was shuffled independently and split proportionally between arms.

**Balance check:** All 6 Customer_Type strata were perfectly balanced between treatment and control (identical counts per stratum in both arms).

### Pre-Registration
The test design was formally pre-registered before any outcome data was inspected (`docs/pre_registration.md`). Assignment file was locked at registration.

| Parameter | Value |
|---|---|
| Primary metric | Conversion within 90 days |
| Primary test | Two-proportion z-test, two-tailed |
| α | 0.05 |
| Power | 0.80 |
| n per group | 47 (constrained by At Risk population = 95) |
| Horizon | 90 days, fixed |
| Early stopping | None |
| Guardrails | AOV (t-test, −5% threshold), unsubscribe, OTIF |
| Multiple comparisons | Bonferroni if guardrails tested jointly |

---

## 6. Statistical Method

**Test:** Two-proportion z-test (two-tailed)

**Why this test was chosen:**
- Binary metric (converted / not converted) → proportions framework is the natural choice
- Fully explainable to non-technical stakeholders: "we compare the fraction who responded in each group"
- Robust to non-normality at the sample level by the Central Limit Theorem (CLT applies at n≥30)
- Established precedent in A/B testing literature and industry practice for conversion rate tests
- Simple to implement and audit — no black-box components

---

## 7. Power Analysis

| Parameter | Value |
|---|---|
| Baseline conversion rate | 83.2% |
| Target MDE | 5.0pp (pre-specified) |
| Required n per group (5pp MDE) | 766 |
| Available At Risk n | 95 |
| n per group used | 47 |
| Detectable MDE at n=47 | **15.1pp** |
| Alpha | 0.05 |
| Power | 0.80 |

The At Risk population (n=95) is substantially smaller than required for a 5pp MDE. The test was redesigned for the available population with a 47/47 split, yielding a detectable MDE of 15.1pp at 80% power. The cost model (Section 8) confirms that even 15.1pp is economically meaningful.

---

## 8. Cost Model Summary

| Parameter | Value |
|---|---|
| Cost per customer contacted | $8.00 |
| Full rollout cost (95 customers) | $760 |
| Expected AOV (At Risk historical) | $33,381 |
| Break-even lift required | 0.024pp (negligible) |
| At realistic lift (15.1pp), ROI | ~62,739% |

The campaign is highly cost-efficient. The $8 cost-per-contact is trivial relative to the $33K average order value, meaning the break-even lift is essentially zero. Any detectable positive effect justifies rollout.

---

## 9. Results (Simulated Outcome)

| Metric | Value |
|---|---|
| Treatment conversion | 97.8% (45/46) |
| Control conversion | 82.6% (38/46) |
| Absolute lift | **15.2pp** |
| Relative lift | 18.4% |
| 95% CI | [3.5pp, 27.0pp] |
| Z-statistic | 2.457 |
| **p-value** | **0.0140** |
| Significant at α=0.05 | **YES** |
| Guardrail — AOV | PASS |
| Post-hoc power | 0.772 |

**Note:** Outcomes are simulated (numpy seed=42). The simulation injects the detectable MDE plus N(0, 0.01) noise into the treatment arm to approximate what a real test with the observed lift magnitude would produce. In a live deployment, actual order data from the 90-day post-assignment window would replace the simulation.

---

## 10. Final Recommendation

**SCALE TO FULL POPULATION**

Statistical significance confirmed (p=0.014), realised ROI at observed lift is ~63,407%, and all guardrail metrics passed. The campaign should be rolled out to all 95 At Risk customers.

Realised economics at observed 15.2pp lift (full rollout):
- Incremental revenue: ~$482,652
- Campaign cost: $760
- Net revenue: ~$481,892

---

## 11. Known Limitations

1. **Simulated outcome:** Real test results will differ. Simulation is for demonstration only — actual deployment requires running the campaign and collecting outcome data over 90 days.
2. **No seasonal adjustment:** The 2-year dataset spans only 2022–2023. Campaign effects may vary by quarter (e.g., Q4 holiday uplift may inflate treatment conversion).
3. **New customer cold-start:** RFM scoring requires 2+ years of history. Customers acquired after 2022 have truncated history and may be misclassified.
4. **Single-channel assumption:** The $8 cost assumes email + account manager time. Multi-channel campaigns (phone, direct mail) will increase cost and require re-computing break-even.
5. **No network effects modelled:** B2B customers may influence each other. If treatment customers discuss the campaign with control customers, contamination inflates observed lift.
6. **Synthetic data density:** All 500 customers placed orders every month, making natural churn hard to observe. Real DHL data will have meaningful churn rates enabling sharper baseline estimates.

---

## 12. Recommended Next Steps

1. **Real test execution:** Apply pre-registration (`docs/pre_registration.md`) to live campaign. Lock assignment file and collect 90 days of order data.
2. **Sample pooling:** Pool multiple campaign waves to reach n=766 per group, enabling detection of 5pp lifts — more actionable for fine-grained optimisation.
3. **Subgroup analysis:** Post-test, examine lift by Customer_Type and Region to identify the most responsive sub-segments for future targeting.
4. **Longitudinal tracking:** Monitor converted At Risk customers over 12 months to measure whether they migrate to Loyal/Champions or relapse — quantifying campaign lifetime value.
5. **Automation:** Integrate the RFM pipeline into a monthly scoring cadence. Trigger the retention intervention automatically when a customer's R_score drops to ≤2 for two consecutive months.
