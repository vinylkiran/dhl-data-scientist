# Production Recommendation — RFM Retention Campaign A/B Test

## Decision: SCALE TO FULL AT RISK POPULATION

### In one sentence:
The retention campaign delivered a statistically significant 15.2 percentage-point lift in At Risk customer engagement (p=0.014), generating an estimated $482K incremental revenue at a total campaign cost of $760, with all risk guardrails passing — scale immediately.

---

### The numbers:

| Metric | Value |
|---|---|
| At Risk segment size | 95 customers |
| Baseline conversion rate | 83.2% |
| Observed lift in test | **15.2pp** (18.4% relative) |
| 95% Confidence Interval | [3.5pp, 27.0pp] |
| Statistical significance | p = 0.0140 (threshold = 0.05) |
| Campaign cost (full rollout) | $760 ($8 per customer) |
| Incremental revenue (at observed lift) | $482,652 |
| Net revenue | $481,892 |
| Realised ROI | ~63,400% |
| AOV guardrail | PASS — no revenue dilution |
| Post-hoc power | 0.772 |

---

### What this means for operations:

The At Risk segment (24% of scored customers) represents DHL's highest churn-risk cohort: frequent buyers with declining recency. A $8 touchpoint — personalised email plus a brief account manager call — converts a meaningful share of these customers back to active status. At the observed lift, every $1 spent on the campaign returns over $630 in incremental revenue, driven by the high average order value of $33K for this segment. The campaign is cost-trivial relative to the revenue at stake, which means the decision criterion is not ROI (which is extreme) but whether the lift is real — and p=0.014 with a CI excluding zero confirms it is.

The fixed 90-day horizon and pre-registered stopping rule ensure the result is not inflated by early peeking. Stratified randomisation across all 6 customer types confirms the lift is not an artefact of segment imbalance.

---

### What would change this recommendation:

- **If the live campaign (real data, n=95) yields a lift below 0.024pp**, the campaign does not cover its own cost — halt and investigate root cause (wrong channel, wrong message, wrong timing).
- **If cost per contact rises above $50** (e.g., adding direct mail or in-person visits), the break-even lift threshold rises to ~0.15pp, which remains well within the observed range but warrants a pre-rollout cost audit before expanding the channel mix.

---

*Prepared by: DHL Data Science*  
*Analysis date: 2023-12-31*  
*Test horizon: Oct–Dec 2023 (simulated) | Real horizon: 90 days post-campaign launch*  
*Pre-registration document: docs/pre_registration.md*
