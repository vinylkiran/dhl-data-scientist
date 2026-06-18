# Model Card — Warehouse Optimization Suite
**DHL Data Scientist Portfolio | Project 4**
**Date:** 2026-06-18 | **Author:** Vinyl Kiran

---

## Overview

This project applies four classes of operations research and data science methods to warehouse pick-path planning, SKU slotting, and cadence recommendation across DHL's three-warehouse network (NJ01, IL02, TX03).

**Data:** 219,000 WMS task records (Jan 2022 – Dec 2023), 2,640 storage locations, 1,664 active SKUs.

---

## Methods

### 1. Pick-Path Optimization

| Method | Description | Complexity |
|---|---|---|
| Naive | Follow WMS task order as-is | O(n) |
| Nearest-Neighbor | Greedy from depot; always visit closest unvisited stop | O(n²) |
| 2-Opt | Start from NN, apply pairwise reversal swaps until no improvement | O(n² × iter) |
| Exact TSP | Enumerate all permutations (sessions ≤ 8 stops only) | O(n!) |

**Training / Evaluation Data:** 200-session random sample (seed=42) from 2,766 filtered sessions (5–15 distinct SKUs).

**Validation benchmark:** Exact optimal computed via `itertools.permutations` for sessions with ≤ 8 stops (n=91 sessions).

**Key Results:**
- 2-opt reduces distance **49.8%** vs naive order
- 2-opt mean gap from exact optimal: **1.75%** (sessions ≤ 8 stops)
- Nearest-neighbor gap from optimal: **4.69%**
- 2-opt compute time: ~0.34 ms/session vs 100 ms/session for exact

---

### 2. Market Basket / SKU Affinity Analysis

**Method:** Manual co-occurrence counting (equivalent to Apriori at min_support ≈ 0.001) + permutation test for statistical validation.

**Support threshold rationale:** The catalogue contains 1,664 SKUs across 2,766 sessions. Maximum single-SKU support is 1.05% (29 sessions); maximum pair co-occurrence is 4 sessions (0.14% support). Standard Apriori thresholds (1%) would eliminate all pairs. The effective threshold used is min 2 co-occurrences, consistent with minimum absolute support of 2 observations before computing lift.

**Permutation test:** For each of the top-50 pairs by lift, 100 permutations of column assignments were performed. p-value = fraction of permuted lifts ≥ observed lift. Threshold: p < 0.05.

**Key Results:**
- 5,807 pairs with ≥ 2 co-occurrences
- Top-200 candidate pairs selected by lift
- 50 pairs permutation-tested; **all 50 validated** (p < 0.05, all p = 0.00)
- Top pair: AUT-001772 ↔ FSH-000914, lift = 71.8
- Note: high lifts driven by sparse catalogue — pairs occur rarely but reliably

---

### 3. Slotting Optimization

**Zone structure (actual from warehouse_locations.csv):**
| Zone | Aisles | Distance from Depot | Locations/WH |
|---|---|---|---|
| Pick_Face | P01–P08 | Nearest (aisle_x 6–13) | 320 |
| Reserve | S01–S06 | Far (aisle_x 16–21) | 240 |
| Bulk | B01–B04 | Furthest (aisle_x 0–3) | 160 |

**Depot reference:** Dispatch aisles D01–D02 (aisle_x 4–5)

**Method (a) — Greedy Heuristic:** Sort SKUs by pick count descending; assign to Pick_Face until capacity, then Reserve, then Bulk.

**Method (b) — LP Joint Optimization (PuLP/CBC):**
- Scope: top 200 SKUs by pick frequency
- Variables: binary x[sku, zone]
- Objective: minimize Σ(dist_score[zone] × pick_count[sku] × x[sku,zone]) − 0.5 × Σ(lift × y[pair,zone])
- Constraints: each SKU in exactly one zone; zone capacity not exceeded
- Affinity bonus linearized with auxiliary binary variables y[sku_a, sku_b, zone]
- Solve time: < 1 second (CBC solver)

**Current (baseline) slotting:** Alphabetical-zone sort → top SKUs placed in Bulk (worst zone) — represents a naïve WMS-default assignment.

**Key Results (500-session sample):**
| Method | Mean Travel Distance | vs Current |
|---|---|---|
| Current (Naive) | 60.36 | — |
| Greedy | 60.13 | +0.4% |
| LP Joint | 59.44 | +1.5% |

---

### 4. Robustness / Temporal Stability Testing

**Windows:** 8 quarters, Q1-2022 through Q4-2023.

**Metrics per consecutive window pair:**
- Spearman rank correlation of SKU pick frequencies
- % of SKUs that change tier (Hot/Warm/Cool/Cold)
- Overlap of top-50 affinity pairs

**Key Results:**
- Mean Spearman ρ = **0.023** (near-random — SKU picks are uniformly distributed)
- Mean tier change rate = **74.4%/quarter** (very high volatility)
- Affinity pair overlap = **0.0%** between consecutive quarters
- Cadence recommendation: **MONTHLY** re-optimization

---

## Performance Numbers

| Metric | Value |
|---|---|
| 2-opt distance reduction vs naive | 49.8% |
| 2-opt gap from exact optimal | 1.75% |
| LP slotting improvement vs current | 1.5% |
| Annual net benefit (recommended approach) | $2,434,136 |
| Payback period ($200K implementation) | ~1 month |

---

## Known Limitations

1. **Simplified 2D distance model.** Actual warehouses have one-way aisles, aisle-end turns, physical obstructions, and different horizontal vs vertical travel speeds. The Euclidean + level-penalty model approximates but does not capture these.

2. **No real SKU-to-location mapping.** The source data does not contain a pick-to-location field. SKU-location assignments are synthetic (rank-order within zones). Actual location-level travel savings would differ.

3. **Sessions are approximate.** Pick sessions are defined by Warehouse_ID + Date + Shift — not by physical trolley run. Multiple operators work a shift simultaneously, so a "session" may span multiple actual pick routes.

4. **Static demand assumption.** The optimization assumes demand is stable within the re-optimization window. The robustness analysis shows this is invalid for the synthetic dataset (near-uniform picks), implying monthly cadence. Real warehouse demand is typically more Pareto-distributed, which would produce higher stability.

5. **Zone-level (not location-level) LP.** The LP assigns SKUs to zones, not individual bin locations. A full location-level assignment would require the actual pick-face location database and would produce larger travel-time improvements.

6. **Affinity pairs are sparse.** With 1,664 SKUs across 2,766 sessions, maximum pair co-occurrence is 4 observations. High lifts are reliable (permutation p = 0.00) but represent very rare joint picks. Zone co-placement provides modest aggregate benefit.

---

## Production Recommendation Summary

- **Path planning:** 2-Opt (deploy as WMS route-sequencing step)
- **Slotting:** LP Joint Optimization for top-200 SKUs, greedy for remainder
- **Cadence:** Monthly re-optimization (based on high tier volatility)
- **Not deployed:** Exact TSP (NP-hard, 290× cost, 1.75% incremental gain)
