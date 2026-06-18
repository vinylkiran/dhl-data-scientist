"""
03_affinity_analysis.py
=======================
Market basket / co-occurrence analysis on pick sessions.
Apriori + permutation test for statistical validation of SKU affinity pairs.

NOTE on support threshold:
  The synthetic dataset has ~1664 SKUs across ~2766 sessions with 5-15 SKUs each.
  Maximum single SKU support is ~1.05% (29 sessions). Maximum pair co-occurrence
  is 4 sessions (support ≈ 0.14%). Apriori's min_support=0.01 would require
  ~28 sessions of joint occurrence, which no pair achieves. We therefore use a
  manual co-occurrence approach with min_support = max(2/n, 0.001) — i.e., at
  least 2 co-occurrences — and validate with the permutation test. This is
  equivalent to Apriori at min_support=0.001 on a large catalogue dataset.

DHL Warehouse Optimization — DS Project 4
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS  = BASE_DIR / "outputs"
FIGURES  = BASE_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load pick sessions ────────────────────────────────────────────────────────
print("Loading pick sessions …")
sessions = pd.read_csv(OUTPUTS / "pick_sessions.csv")
sessions["sku_set"] = sessions["sku_list"].apply(json.loads)

n_sessions = len(sessions)
print(f"  Sessions: {n_sessions:,}")

# ── Compute pair co-occurrence and lift ───────────────────────────────────────
print("Computing pair co-occurrences …")

sku_counts  = Counter()
pair_counts = Counter()

for lst in sessions["sku_set"]:
    skus = sorted(set(lst))
    for s in skus:
        sku_counts[s] += 1
    for a, b in combinations(skus, 2):
        pair_counts[(a, b)] += 1

print(f"  Unique SKUs : {len(sku_counts):,}")
print(f"  Unique pairs: {len(pair_counts):,}")

# min_support: at least 2 co-occurrences, or 0.001 of sessions
# Justification: too-rare pairs are unreliable even if lift looks high;
# we require observation in multiple independent sessions before computing lift.
MIN_CO_OCC = max(2, int(0.001 * n_sessions))
print(f"  Minimum co-occurrences required: {MIN_CO_OCC} (= max(2, 0.1% of {n_sessions} sessions))")

rows = []
for (a, b), cnt in pair_counts.items():
    if cnt < MIN_CO_OCC:
        continue
    sup   = cnt / n_sessions
    sup_a = sku_counts[a] / n_sessions
    sup_b = sku_counts[b] / n_sessions
    if sup_a == 0 or sup_b == 0:
        continue
    lift = sup / (sup_a * sup_b)
    conf = sup / sup_a   # confidence A→B
    rows.append({"sku_a": a, "sku_b": b, "support": sup, "confidence": conf, "lift": lift,
                 "_count": cnt})

pair_df = (pd.DataFrame(rows)
           .sort_values("lift", ascending=False)
           .reset_index(drop=True))

# Top 200 candidate pairs by lift
top_200 = pair_df.head(200).copy()
print(f"  Pairs with ≥{MIN_CO_OCC} co-occurrences: {len(pair_df):,}")
print(f"  Top-200 candidate pairs selected")

# ── Permutation test ──────────────────────────────────────────────────────────
print("\nRunning permutation test on top 50 pairs …")

N_PERMUTATIONS  = 100
TOP_TEST_PAIRS  = min(50, len(top_200))

# Build arrays for fast permutation
sku_list_all   = [sorted(set(s)) for s in sessions["sku_set"]]
all_skus_uniq  = sorted(sku_counts.keys())
sku_idx        = {s: i for i, s in enumerate(all_skus_uniq)}

# Precompute boolean matrix (n_sessions × n_skus)
mat = np.zeros((n_sessions, len(all_skus_uniq)), dtype=np.float32)
for r, skus in enumerate(sku_list_all):
    for s in skus:
        if s in sku_idx:
            mat[r, sku_idx[s]] = 1.0

rng = np.random.default_rng(seed=42)
perm_results = []

for i in range(TOP_TEST_PAIRS):
    row  = top_200.iloc[i]
    a, b = row["sku_a"], row["sku_b"]
    obs_lift = row["lift"]

    if a not in sku_idx or b not in sku_idx:
        perm_results.append({"sku_a": a, "sku_b": b, "permutation_pval": 1.0})
        continue

    col_a = mat[:, sku_idx[a]]
    col_b = mat[:, sku_idx[b]]
    sup_a = col_a.mean()
    sup_b = col_b.mean()

    if sup_a == 0 or sup_b == 0:
        perm_results.append({"sku_a": a, "sku_b": b, "permutation_pval": 1.0})
        continue

    perm_lifts = np.empty(N_PERMUTATIONS)
    for j in range(N_PERMUTATIONS):
        shuffled = rng.permutation(col_a)
        joint_p  = np.mean(shuffled * col_b)
        perm_lifts[j] = joint_p / (sup_a * sup_b)

    p_val = np.mean(perm_lifts >= obs_lift)
    perm_results.append({"sku_a": a, "sku_b": b, "permutation_pval": p_val})

perm_df = pd.DataFrame(perm_results)
top_tested = top_200.head(TOP_TEST_PAIRS).merge(perm_df, on=["sku_a", "sku_b"], how="left")
remaining  = top_200.iloc[TOP_TEST_PAIRS:].copy()
remaining["permutation_pval"] = np.nan

all_pairs          = pd.concat([top_tested, remaining], ignore_index=True)
all_pairs["validated"] = all_pairs["permutation_pval"] < 0.05

n_validated = int(all_pairs["validated"].sum())
print(f"  Tested: {TOP_TEST_PAIRS} pairs | Validated (p<0.05): {n_validated}")

# ── Export ────────────────────────────────────────────────────────────────────
export_cols = ["sku_a", "sku_b", "support", "confidence", "lift",
               "permutation_pval", "validated"]
all_pairs[export_cols].to_csv(OUTPUTS / "affinity_pairs_validated.csv", index=False)
print(f"  Saved → outputs/affinity_pairs_validated.csv  ({len(all_pairs):,} rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== AFFINITY SUMMARY ===")
print(f"  Sessions analysed         : {n_sessions:,}")
print(f"  Unique SKUs               : {len(sku_counts):,}")
print(f"  Candidate pairs (top-200) : {len(all_pairs):,}")
print(f"  Pairs permutation-tested  : {TOP_TEST_PAIRS}")
print(f"  Validated pairs (p<0.05)  : {n_validated}")

validated = all_pairs[all_pairs["validated"]].sort_values("lift", ascending=False)
if len(validated) > 0:
    print(f"\n  Top 10 validated pairs by lift:")
    for _, r in validated.head(10).iterrows():
        print(f"    {r['sku_a']} ↔ {r['sku_b']}  lift={r['lift']:.3f}  support={r['support']:.4f}  p={r['permutation_pval']:.3f}")
else:
    print("  No pairs passed permutation test at p<0.05.")
    print("  (Expected: lift is driven by random coincidence in sparse catalogue data)")

# ── Figures ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")

# Figure 1: Lift distribution
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(all_pairs["lift"], bins=40, kde=True, ax=ax, color="#1F77B4")
ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Lift = 1 (independence)")
ax.set_title("Lift Score Distribution — Top 200 SKU Affinity Pairs", fontsize=14, fontweight="bold")
ax.set_xlabel("Lift Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "affinity_lift_distribution.png", dpi=150)
plt.close()
print("\nFigure saved → figures/affinity_lift_distribution.png")

# Figure 2: Top 20 pairs by lift (all top-200 if no validated — show p-value info)
top20 = all_pairs.head(20).copy()
top20["pair_label"] = top20["sku_a"] + " ↔ " + top20["sku_b"]
top20["color"] = top20["validated"].map({True: "#2CA02C", False: "#1F77B4"})

fig, ax = plt.subplots(figsize=(12, max(6, len(top20) * 0.45)))
bar_colors = top20["color"].values[::-1].tolist()
y_pos = range(len(top20))
bars = ax.barh(list(y_pos), top20["lift"].values[::-1].tolist(),
               color=bar_colors, alpha=0.85)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(top20["pair_label"].values[::-1].tolist(), fontsize=8)
ax.set_xlabel("Lift Score", fontsize=12)
ax.set_title("Top 20 SKU Affinity Pairs by Lift\n(Green = validated p<0.05, Blue = not validated)",
             fontsize=13, fontweight="bold")
for bar_obj, val in zip(bars, top20["lift"].values[::-1].tolist()):
    ax.text(val + 0.02, bar_obj.get_y() + bar_obj.get_height()/2,
            f"{val:.2f}", va="center", ha="left", fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES / "affinity_top_pairs.png", dpi=150)
plt.close()
print("Figure saved → figures/affinity_top_pairs.png")

print("\nDone — 03_affinity_analysis.py complete.")
