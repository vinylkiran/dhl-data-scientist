"""
05_robustness_validation.py
============================
Test temporal stability of slotting recommendations across 8 rolling quarterly
windows spanning Jan 2022 – Dec 2023.

Metrics:
  - SKU tier stability (Hot/Warm/Cool/Cold by pick frequency rank)
  - Affinity pair stability (top-50 pairs by support, consecutive window overlap)
  - Re-optimization cadence recommendation based on stability thresholds

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
from scipy.stats import spearmanr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS  = BASE_DIR / "outputs"
FIGURES  = BASE_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
tasks    = pd.read_csv(DATA_DIR / "wms_tasks.csv", parse_dates=["Task_Date"])
sessions = pd.read_csv(OUTPUTS / "pick_sessions.csv")
sessions["sku_set"] = sessions["sku_list"].apply(json.loads)
sessions["date_dt"] = pd.to_datetime(sessions["date"])

picks = tasks[tasks["Task_Type"] == "Pick"].copy()
print(f"  Pick tasks: {len(picks):,}")
print(f"  Date range: {picks['Task_Date'].min().date()} → {picks['Task_Date'].max().date()}")

# ── Define 8 quarterly windows ────────────────────────────────────────────────
WINDOWS = [
    ("Q1-2022", "2022-01-01", "2022-03-31"),
    ("Q2-2022", "2022-04-01", "2022-06-30"),
    ("Q3-2022", "2022-07-01", "2022-09-30"),
    ("Q4-2022", "2022-10-01", "2022-12-31"),
    ("Q1-2023", "2023-01-01", "2023-03-31"),
    ("Q2-2023", "2023-04-01", "2023-06-30"),
    ("Q3-2023", "2023-07-01", "2023-09-30"),
    ("Q4-2023", "2023-10-01", "2023-12-31"),
]

def assign_tier(rank_pct: float) -> str:
    """Assign tier based on pick-frequency percentile rank."""
    if rank_pct <= 0.25:  return "Hot"
    if rank_pct <= 0.50:  return "Warm"
    if rank_pct <= 0.75:  return "Cool"
    return "Cold"

# ── Compute per-window metrics ────────────────────────────────────────────────
window_data = []  # list of dicts: {window, sku_rank, sku_tier, top50_pairs}

print("\nComputing per-window SKU rankings and affinity pairs …")

for wname, wstart, wend in WINDOWS:
    # Filter picks in window
    w_picks = picks[(picks["Task_Date"] >= wstart) & (picks["Task_Date"] <= wend)]
    if len(w_picks) == 0:
        print(f"  {wname}: no data — skipping")
        continue

    # Pick frequency per SKU
    sku_cnt = w_picks.groupby("SKU_ID").size().reset_index(name="pick_count")
    sku_cnt = sku_cnt.sort_values("pick_count", ascending=False).reset_index(drop=True)
    sku_cnt["rank"]     = sku_cnt.index + 1
    sku_cnt["rank_pct"] = sku_cnt["rank"] / len(sku_cnt)
    sku_cnt["tier"]     = sku_cnt["rank_pct"].apply(assign_tier)
    sku_cnt["window"]   = wname

    # Affinity pairs from sessions in this window
    w_sessions = sessions[
        (sessions["date_dt"] >= wstart) & (sessions["date_dt"] <= wend)
    ]
    pair_counts = Counter()
    n_wh_sess   = len(w_sessions)
    for _, row in w_sessions.iterrows():
        skus = sorted(set(row["sku_set"]))
        for a, b in combinations(skus, 2):
            pair_counts[(a, b)] += 1

    # Top 50 pairs by raw count (support-equivalent)
    top50_pairs = set(pair for pair, _ in pair_counts.most_common(50))

    window_data.append({
        "window"      : wname,
        "sku_rank"    : sku_cnt.set_index("SKU_ID")["rank"].to_dict(),
        "sku_tier"    : sku_cnt.set_index("SKU_ID")["tier"].to_dict(),
        "pick_count"  : sku_cnt.set_index("SKU_ID")["pick_count"].to_dict(),
        "top50_pairs" : top50_pairs,
        "n_skus"      : len(sku_cnt),
        "n_sessions"  : n_wh_sess,
    })
    print(f"  {wname}: {len(w_picks):,} picks | {len(sku_cnt):,} SKUs | "
          f"{n_wh_sess:,} sessions | {len(top50_pairs)} pairs")

# ── Stability metrics across consecutive window pairs ────────────────────────
print("\nComputing stability metrics …")

stability_rows = []

for i in range(len(window_data) - 1):
    w1 = window_data[i]
    w2 = window_data[i + 1]

    pair_label = f"{w1['window']} → {w2['window']}"

    # Common SKUs
    common_skus = set(w1["sku_rank"].keys()) & set(w2["sku_rank"].keys())
    if len(common_skus) < 10:
        print(f"  {pair_label}: too few common SKUs ({len(common_skus)}) — skip")
        continue

    # Spearman rank correlation of pick-frequency rankings
    ranks1 = [w1["sku_rank"][s] for s in common_skus]
    ranks2 = [w2["sku_rank"][s] for s in common_skus]
    corr, p_corr = spearmanr(ranks1, ranks2)

    # % of SKUs that changed tier
    tier_changes = sum(
        1 for s in common_skus
        if w1["sku_tier"].get(s) != w2["sku_tier"].get(s)
    )
    pct_changed = tier_changes / len(common_skus) * 100

    # Affinity pair overlap
    overlap = len(w1["top50_pairs"] & w2["top50_pairs"])
    overlap_pct = overlap / 50 * 100

    stability_rows.append({
        "window_pair"         : pair_label,
        "w1"                  : w1["window"],
        "w2"                  : w2["window"],
        "n_common_skus"       : len(common_skus),
        "spearman_corr"       : round(corr, 4),
        "spearman_p"          : round(p_corr, 6),
        "pct_tier_changed"    : round(pct_changed, 1),
        "n_tier_changed"      : tier_changes,
        "affinity_overlap_pct": round(overlap_pct, 1),
        "affinity_overlap_n"  : overlap,
    })

stab_df = pd.DataFrame(stability_rows)

print(f"\n=== STABILITY METRICS ===")
print(stab_df[[
    "window_pair", "spearman_corr", "pct_tier_changed", "affinity_overlap_pct"
]].to_string(index=False))

# ── Re-optimization cadence recommendation ────────────────────────────────────
mean_corr    = stab_df["spearman_corr"].mean()
mean_pct_chg = stab_df["pct_tier_changed"].mean()

if mean_corr > 0.95 and mean_pct_chg < 10:
    cadence = "quarterly"
    cadence_reason = f"tier rank correlation={mean_corr:.3f} >0.95, tier change rate={mean_pct_chg:.1f}% <10%"
elif mean_corr > 0.85 and mean_pct_chg < 20:
    cadence = "semi-annual"
    cadence_reason = f"tier rank correlation={mean_corr:.3f} >0.85, tier change rate={mean_pct_chg:.1f}% <20%"
else:
    cadence = "monthly"
    cadence_reason = f"tier rank correlation={mean_corr:.3f}, tier change rate={mean_pct_chg:.1f}% (high volatility)"

print(f"\n=== RE-OPTIMIZATION CADENCE ===")
print(f"  Mean Spearman correlation : {mean_corr:.4f}")
print(f"  Mean tier change rate     : {mean_pct_chg:.1f}% per quarter")
print(f"  Mean affinity overlap     : {stab_df['affinity_overlap_pct'].mean():.1f}%")
print(f"\n  RECOMMENDED CADENCE: {cadence.upper()}")
print(f"  Rationale: {cadence_reason}")

# ── Export ────────────────────────────────────────────────────────────────────
# Flatten window metadata for export
meta_rows = []
for w in window_data:
    meta_rows.append({
        "window"    : w["window"],
        "n_skus"    : w["n_skus"],
        "n_sessions": w["n_sessions"],
    })
meta_df = pd.DataFrame(meta_rows)

export_df = stab_df.merge(
    meta_df.rename(columns={"window": "w1", "n_skus": "n_skus_w1"})[["w1", "n_skus_w1"]],
    on="w1", how="left"
).merge(
    meta_df.rename(columns={"window": "w2", "n_skus": "n_skus_w2"})[["w2", "n_skus_w2"]],
    on="w2", how="left"
)
export_df["recommended_cadence"] = cadence
export_df.to_csv(OUTPUTS / "robustness_results.csv", index=False)
print(f"\nSaved → outputs/robustness_results.csv  ({len(export_df)} rows)")

# ── Figures ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
x_labels = stab_df["window_pair"].str.replace(" → ", "→\n", regex=False)

# Figure 1: Spearman correlation
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(stab_df)), stab_df["spearman_corr"].values, "o-",
        color="#1F77B4", linewidth=2, markersize=8, label="Spearman ρ")
ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, label="Quarterly threshold (0.95)")
ax.axhline(0.85, color="orange", linestyle="--", linewidth=1.5, label="Semi-annual threshold (0.85)")
ax.set_xticks(range(len(stab_df)))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title("SKU Pick-Frequency Rank Stability (Spearman ρ)\nConsecutive Quarterly Windows",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Spearman Rank Correlation", fontsize=12)
ax.set_xlabel("Window Pair", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES / "robustness_tier_correlation.png", dpi=150)
plt.close()
print("Figure saved → figures/robustness_tier_correlation.png")

# Figure 2: Affinity pair overlap
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(stab_df)), stab_df["affinity_overlap_pct"].values,
       color="#2CA02C", alpha=0.75, label="Affinity pair overlap %")
ax.plot(range(len(stab_df)), stab_df["pct_tier_changed"].values, "s--",
        color="#D62728", linewidth=2, markersize=8, label="Tier change rate %")
ax.set_xticks(range(len(stab_df)))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_title("Affinity Pair Overlap & Tier Change Rate\nConsecutive Quarterly Windows",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_xlabel("Window Pair", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES / "robustness_affinity_overlap.png", dpi=150)
plt.close()
print("Figure saved → figures/robustness_affinity_overlap.png")

print(f"\n=== CONCLUSION ===")
print(f"  Recommended re-optimization cadence: {cadence.upper()}")
print(f"  Basis: mean Spearman ρ={mean_corr:.3f}, mean tier change rate={mean_pct_chg:.1f}%/quarter")
print(f"  Affinity pair overlap averaged {stab_df['affinity_overlap_pct'].mean():.1f}% between quarters")
print("\nDone — 05_robustness_validation.py complete.")
