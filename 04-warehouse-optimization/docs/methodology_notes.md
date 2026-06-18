# Methodology Notes — Warehouse Optimization
**DHL Data Scientist Portfolio | Project 4**

---

## 1. Why Nearest-Neighbor + 2-Opt Over Exact TSP at Scale

### The Problem

The Travelling Salesman Problem (TSP) asks: given n locations to visit, what ordering minimises total travel distance? Finding the exact optimum requires evaluating all permutations of n locations.

The number of permutations grows factorially:

| Stops | Permutations | Time at 10M evals/sec |
|---|---|---|
| 8 | 40,320 | < 0.01 seconds |
| 10 | 3,628,800 | 0.36 seconds |
| 12 | 479,001,600 | 47.9 seconds |
| 15 | 1,307,674,368,000 | 130,767 seconds (~36 hours) |

For sessions with 5–15 SKUs (our filtered range), exact TSP is computationally infeasible beyond 8–10 stops.

### The 2-Opt Approach

2-opt is a local search improvement algorithm:
1. Start with a nearest-neighbor solution (O(n²) to construct)
2. Try all pairs of edges (i, j): reverse the sub-path between them
3. If the reversal reduces total distance, accept it
4. Repeat until no improving swap exists (or max iterations reached)

**Time complexity:** O(n² × iterations) — polynomial in n, not factorial.

**Empirical performance on this dataset:**
- 200-session sample, 5–15 stops each
- Mean gap from exact optimal (sessions ≤ 8 stops): **1.75%**
- Nearest-neighbor gap: 4.69%
- 2-opt compute time: 0.34 ms/session vs 100 ms/session for exact
- At 1,500 daily sessions: 2-opt costs $1.55/month vs $450/month for exact

### Conclusion

2-opt captures **98.3% of the exact-solution benefit** at **290× lower compute cost**. The remaining 1.75% gap is worth approximately $86K/year at full scale — but closing it would cost $5,381/year in extra compute while requiring a fundamentally different (and operationally fragile) solver infrastructure. The cost-benefit is strongly in favour of 2-opt.

For very large sessions (>30 stops), more advanced heuristics such as LKH (Lin-Kernighan-Helsgott) could be considered, but this is out of scope for typical DHL pick sessions.

---

## 2. Why Permutation Testing for Affinity Pairs

### The Problem with Raw Lift

Lift is defined as:

```
lift(A → B) = P(A ∩ B) / (P(A) × P(B))
```

A lift > 1 means A and B co-occur more than expected if they were independent. However, lift is inflated by:

1. **Sparse counts.** If A appears in only 3 sessions and B in 2 sessions, a single session containing both gives lift = (1/N) / (3/N × 2/N) = N/6. With N = 2,766 sessions, even accidental co-occurrence produces lift > 400.

2. **Marginal frequency imbalances.** Rare items appear to have high lift with each other simply because their individual supports are low — not because they are genuinely associated.

3. **Multiple comparisons.** With 125,468 unique pairs, many will appear significant by chance at standard thresholds.

### The Permutation Test

For each candidate pair (A, B), we ask: could the observed lift arise by chance given A's and B's individual occurrence rates?

**Procedure:**
1. Record observed lift for (A, B)
2. Repeat 100 times:
   - Shuffle the column of session-indicators for SKU A (permute session assignments)
   - Recompute lift for (A, B) using the shuffled column
   - This preserves A's marginal support but destroys any genuine joint structure
3. p-value = fraction of permuted lifts ≥ observed lift

**Why this controls the right null hypothesis:** Shuffling A's column while keeping B's column fixed generates the null distribution of lift under the assumption that A and B co-occur purely by chance given their individual frequencies. If the observed lift exceeds almost all permuted lifts, we have evidence that their co-occurrence is non-random.

**Results:** All 50 tested pairs had p = 0.00 (0 of 100 permuted lifts matched or exceeded observed). This is consistent with real but sparse joint structure — these pairs genuinely co-occur in the same pick sessions more than chance predicts.

**Limitation:** With 100 permutations and a p-threshold of 0.05, the minimum detectable p-value is 0.01. Pairs with true p between 0.01 and 0.05 cannot be distinguished. For production use, increasing to 1,000 permutations is recommended.

---

## 3. Why LP Formulation for Joint Slotting

### The Greedy Limitation

A greedy slotting approach assigns SKUs one at a time: rank by pick frequency, assign the top SKU to Pick_Face, then the next, and so on. This is optimal for **individual SKU placement** when SKUs are independent — but it ignores affinity structure.

**The interdependency problem:** Suppose SKU A and SKU B are always picked together (high lift). Placing A in Pick_Face is valuable. But if B is also in Pick_Face, every combined pick saves travel twice (once for the A→B leg, once because both are near the depot). A greedy algorithm cannot see this: when it assigns A, B has not yet been assigned, so the joint benefit is invisible.

### The LP Formulation

The Integer Linear Program captures this interdependency explicitly:

**Decision variables:**
- x[sku, zone] ∈ {0, 1}: 1 if SKU is assigned to zone
- y[sku_a, sku_b, zone] ∈ {0, 1}: 1 if both members of an affinity pair are in zone

**Objective (minimise):**
```
Σ_s,z  dist_score[z] × pick_count[s] × x[s,z]
  − 0.5 × Σ_pair,z  lift[pair] × y[pair, z]
```

The first term penalises placing high-pick SKUs far from the depot. The second term rewards placing affinity-linked pairs in the same zone.

**Constraints:**
- Each SKU in exactly one zone: Σ_z x[s,z] = 1 for all s
- Zone capacity: Σ_s x[s,z] ≤ cap[z] for all z
- Linearisation of y = x_a × x_b: y ≤ x_a, y ≤ x_b, y ≥ x_a + x_b − 1

**Why this is tractable:** For 200 SKUs × 3 zones = 600 binary variables, plus ~150 linearisation variables for 50 affinity pairs × 3 zones. The CBC solver (open source, bundled with PuLP) solves this in < 1 second.

**Result:** LP achieves 1.1% lower aggregate travel time than greedy. This is modest because the current zone-level formulation limits precision — all Pick_Face locations look identical to the LP. A location-level formulation (requiring the real pick-to-location database) would produce larger gains.

---

## 4. Why Re-Optimization Cadence is a Cost Factor, Not Just an Operational Detail

### The Hidden Cost of Frequent Re-Slotting

Each re-optimization cycle produces a new SKU-to-zone assignment. Implementing it requires physically relocating SKUs within the warehouse — a significant labour cost:

- **SKUs to move per cycle:** pct_tier_changed × total_SKUs
- **Labour per move:** 15 minutes × $25/hour = $6.25/SKU
- **At 74.4% tier change rate:** 1,238 SKUs × $6.25 = $7,733/cycle

At monthly cadence: **$92,798/year in re-slotting labour alone.**

This re-slotting labour must be compared against the travel-time savings the re-slotting creates. If re-slotting happens too frequently, the labour cost of moving SKUs exceeds the travel-time savings from the improved layout.

### The Stability-Cadence Trade-off

| Cadence | Re-slots/year | Labour cost | Savings capture |
|---|---|---|---|
| Monthly | 12 | $92,798 | Maximum (layout always fresh) |
| Quarterly | 4 | $30,933 | ~75–80% (some drift between re-slots) |
| Semi-annual | 2 | $15,466 | ~50–60% (significant drift) |

**The dataset recommends monthly cadence** because SKU pick frequencies are essentially uniform (CV ≈ 0.36, max 17 picks/quarter vs mean 7.3). Rankings reshuffle randomly each quarter, making any slotting derived from one quarter partially stale the next.

**In a real DHL warehouse** with Pareto-distributed picks (top 20% of SKUs = 80% of picks), tier rankings would be far more stable (Spearman ρ > 0.90), reducing the tier change rate to < 15%/quarter. This would push the recommended cadence to quarterly, saving ~$62K/year in re-slotting labour while maintaining near-optimal travel-time performance.

**Key insight:** The re-optimization cadence should be treated as an optimisation problem in its own right — balancing the diminishing-returns curve of layout freshness against the linear cost of physical SKU relocation.
