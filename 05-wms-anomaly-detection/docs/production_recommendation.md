# Production Recommendation — WMS Anomaly Detection

**Project:** DS Project 5 | **Prepared for:** DHL Supply Chain — Operations Analytics  
**Warehouses:** DHL-WH-IL02, DHL-WH-NJ01, DHL-WH-TX03 | **Date:** June 2026  

---

## Executive Summary

We recommend deploying a **hybrid two-layer anomaly detection system** for daily warehouse monitoring across all three DHL distribution centres. The system combines Statistical Process Control (SPC) as an immediately explainable daily operational layer with Isolation Forest as a weekly diagnostic for multi-feature anomalies invisible to single-metric monitoring. Operator-level LOF monitoring provides a third layer for workforce analytics.

**Total monthly monitoring cost: $110** — driven almost entirely by false-positive investigation time, not compute infrastructure (which costs less than $0.02/month).

---

## The Problem

Warehouse accuracy degradation and volume anomalies are routinely detected late — after errors have compounded across multiple days. A single week of undetected accuracy degradation at 2pp below baseline generates approximately 70 additional pick errors at $25 each: **$1,750 in recoverable downstream cost per incident**. Across three warehouses and four incidents per year per warehouse, the addressable loss is **$21,000/year**.

The current detection approach relies on end-of-shift supervisor reviews and weekly performance reports. The gap between when an anomaly begins and when it is detected is typically 3–7 days.

---

## Recommended System

### Layer 1 — SPC (Deploy Immediately, Daily)

**What it does:** Monitors pick accuracy rate, task volume, and error count per warehouse using four Western Electric run rules plus a CUSUM drift detector.

**Why it's the right starting point:** Control charts are universally understood by ops teams. Every supervisor who has worked in a manufacturing or logistics environment will be familiar with the concept. Zero explanation overhead. Actionable outputs with zero data science involvement in day-to-day use.

**Results over 2-year test window:** 286 anomaly records flagged across 3 warehouses. CUSUM detected 15 gradual drift events that would have taken an additional 2–3 weeks to surface through threshold-only monitoring.

**Key parameter:** 3σ threshold. The robustness test shows that dropping to 2σ increases daily flags by 600–673% — creating unmanageable alert volumes. 3σ is the right production threshold.

### Layer 2 — Isolation Forest (Deploy Week 1, Weekly)

**What it does:** Scores each warehouse-day as anomalous based on all 6 KPI features jointly. Catches days where no single metric is extreme but the combination of accuracy, volume, duration, and error counts is statistically unusual for that warehouse.

**Why it adds value:** 148 warehouse-days were flagged by IF but not by SPC over the 2-year window — averaging 18 additional potential anomalies per quarter that SPC alone would have missed. Even if only 20% are true anomalies, that is 3–4 real incidents per quarter caught earlier.

**Cost:** $0.002/month compute. $110/month in FP investigation time (estimated). Net annual saving vs SPC-only: **$7,428** (driven by SPC's higher FP investigation burden).

### Layer 3 — LOF Operator Monitoring (Deploy Month 2, Weekly)

**What it does:** Identifies operators whose performance pattern is unusual relative to their warehouse peers — not just low accuracy, but unusual combinations of task rate, accuracy, and duration.

**Why it matters:** Distinguishes between system issues (a batch of mislabelled inventory affects all operators equally) and individual issues (a specific operator is struggling post-training). The former requires a process fix; the latter requires a people support intervention. Without this layer, supervisors treat all accuracy drops the same way.

**Results:** 139 operator-day anomalies across 2 years. With 60 operators across 3 warehouses, this represents a manageable monthly review (~6 operator flags/month on average).

---

## What We Decided Not to Deploy

**LOF at warehouse level:** IF and LOF agreed on only 15.2% of warehouse-day flags (Jaccard 0.152). Adding both at warehouse level doubles alert volume without meaningful signal gain. IF at warehouse level, LOF at operator level is the right division of labour.

**Daily ML runs:** Compute cost is negligible, but operational overhead of daily ML review before the ops team has built trust in the outputs would create alert fatigue. Weekly cadence for the first 6 months; promote to daily once 3+ real incidents have been confirmed by the system.

---

## Financial Case

| | SPC Only | Hybrid (Recommended) | Saving |
|--|---------|---------------------|--------|
| Annual compute cost | <$0.01 | <$0.20 | — |
| Annual FP investigation cost | $8,748 | $1,320 | $7,428 |
| Annual total cost | $8,748 | $1,320 | **$7,428** |
| Early detection value (3WH × 4 incidents/yr) | $21,000 | $21,000 | — |
| Net annual value vs no monitoring | $12,252 | $19,680 | +$7,428 |

---

## Conditions for Reviewing This Recommendation

1. **If SPC FP rate exceeds 30% on real data** (i.e., >15 supervisor-hours/month investigation fatigue): increase sigma threshold or retire Rule 3, retune on 6 months of real operational baseline.
2. **If a confirmed ML-only anomaly causes >$1,750 downstream impact** within the first year: upgrade IF to daily cadence and begin annotation programme to build labelled ground truth.
3. **New warehouse onboarding**: minimum 90-day data accumulation before SPC rolling baseline is reliable; bootstrap IF from similar warehouse parameters.
4. **Major process changes** (new WMS software, large workforce change, category expansion): reset baselines for all methods. CUSUM in particular is sensitive to step-changes in the mean.

---

## Next Steps

| Action | Owner | Timeline |
|--------|-------|----------|
| Deploy SPC daily dashboard to WMS BI tool | Data Engineering | Week 1–2 |
| Train warehouse supervisors on control chart interpretation | Ops Analytics | Week 2–3 |
| Set up weekly IF batch job (cron/Airflow) | Data Engineering | Week 3–4 |
| First monthly review: FP rate calibration | Data Science | Month 2 |
| Operator LOF weekly report to shift managers | Ops Analytics | Month 2 |
| 6-month review: promote IF to daily? Annotate incidents? | Data Science + Ops | Month 6 |
