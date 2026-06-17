# Production Recommendation — SKU Segmentation

**Prepared for:** VP Supply Chain Operations  
**Date:** 2026-06-17  
**Project:** DS Project 01 — SKU Segmentation  
**Analyst:** DHL Data Science Team

---

## Decision: DEPLOY K-MEANS AS PRIMARY

### In one sentence:
K-Means clustering passes all three deployment gates — statistical quality, robustness, and financial justification — and should replace rule-based ABC/XYZ as the primary SKU segmentation method for replenishment policy assignment.

---

### The numbers:

| Signal | Value | Gate | Result |
|---|---|---|---|
| Silhouette score (k=4) | **0.5265** | > 0.50 | PASS |
| Seed-stability ARI (20 seeds) | **0.9989** | > 0.95 | PASS |
| % seed pairs with ARI > 0.95 | **100%** | — | PASS |
| Net annual value vs rule-based | **+$3,904,833** | > $0 | PASS |

**Compute cost:** Rule-based = $0.001/yr cloud  →  K-Means = $0.013/yr cloud (difference negligible).  
**Maintenance cost:** Rule-based = $2,400/yr  →  K-Means = $2,200/yr (K-Means is actually marginally cheaper to maintain because it requires less manual explainability effort once embedded).  
**Net annual value:** $3,904,633 stockout value captured − $0.01 compute delta − (−$200 maintenance saving) = **$3,904,833**.

---

### What this means for operations:

K-Means identifies **70 A-class SKUs** that ABC over-rates (high revenue but actually low-velocity demand), and correctly places them in Low-Velocity clusters. For these SKUs, current safety stock levels are likely too high relative to actual demand patterns, creating capital lock-up; simultaneously, when a genuine demand spike occurs, replenishment is triggered too slowly because the SKU is on an A-class fast-cycle schedule misaligned with its real demand rhythm. Correcting these 70 SKUs' segmentation is estimated to prevent $3.9M in annual stockout-related revenue risk at a conservative 10% catch rate assumption.

The four K-Means clusters (Low-Velocity/Low-Value, Low-Velocity/High-Value, High-Velocity/Low-Value, High-Velocity/High-Value) map directly onto four replenishment policies: periodic review for the low-velocity tiers, continuous review with vendor-managed inventory for the high-velocity/high-value tier, and bulk-order scheduling for the high-velocity/low-value tier. Rule-based ABC/XYZ is retained as an explainability and audit overlay — planners can still reference A/B/C and X/Y/Z classifications, and the K-Means cluster label is provided alongside.

---

### What would change this recommendation:

1. **Catalogue drops below ~500 SKUs:** At that scale, rule-based ABC provides sufficient granularity and K-Means adds negligible incremental insight relative to its maintenance overhead. Re-evaluate if the active SKU count falls below this threshold.

2. **Stockout patterns shift structurally** (e.g. post-network restructure, major supplier change, new distribution centre opening): Cluster assignments are based on 2022–2023 demand history. A structural shift in demand patterns would invalidate the current feature distributions. Re-run the full pipeline — including the k-selection sweep — before the next quarterly planning cycle if any of these events occur.
