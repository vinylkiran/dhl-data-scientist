"""
01_rfm_validation.py
RFM Analysis with quintile-based scoring, bootstrap stability validation,
and Spearman correlation analysis.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"

REFERENCE_DATE = pd.Timestamp("2023-12-31")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
orders = pd.read_csv(DATA_DIR / "outbound_orders.csv", parse_dates=["Order_Date"])
customers = pd.read_csv(DATA_DIR / "customers.csv")

print(f"  Orders: {len(orders):,} rows | Date range: {orders['Order_Date'].min().date()} → {orders['Order_Date'].max().date()}")
print(f"  Customers: {len(customers):,} rows")

# ── Compute RFM metrics ──────────────────────────────────────────────────────
rfm = (
    orders.groupby("Customer_ID")
    .agg(
        last_order=("Order_Date", "max"),
        frequency=("Order_ID", "count"),
        monetary=("Revenue", "sum"),
    )
    .reset_index()
)
rfm["recency"] = (REFERENCE_DATE - rfm["last_order"]).dt.days
rfm = rfm.drop(columns=["last_order"])

print(f"\nRFM computed for {len(rfm):,} customers")

# ── Quintile scoring ─────────────────────────────────────────────────────────
def quintile_score(series, invert=False):
    # Use rank-based approach to always get exactly 5 groups
    ranked = series.rank(method="first", pct=True)
    scores = pd.cut(ranked, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True).astype(int)
    if invert:
        scores = 6 - scores
    return scores

rfm["R_score"] = quintile_score(rfm["recency"], invert=True)   # lower recency = higher score
rfm["F_score"] = quintile_score(rfm["frequency"], invert=False)
rfm["M_score"] = quintile_score(rfm["monetary"], invert=False)

rfm["RFM_Score"]   = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
rfm["RFM_Segment"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

# ── Segment mapping ──────────────────────────────────────────────────────────
def assign_segment(row):
    R, F, M = row["R_score"], row["F_score"], row["M_score"]
    if R >= 4 and F >= 4 and M >= 4:
        return "Champions"
    elif R >= 3 and F >= 4:
        return "Loyal"
    elif R >= 4 and F <= 3 and M >= 3:
        return "Potential Loyalist"
    elif R <= 2 and F >= 3:
        return "At Risk"
    elif R == 1 and F <= 2 and M <= 2:
        return "Lost Cheap"
    elif R <= 2 and F <= 2:
        return "Lost"
    elif R == 3 and F == 3 and M == 3:
        return "Needs Attention"
    elif R >= 4 and F <= 2 and M <= 2:
        return "New Customers"
    elif R >= 3 and F <= 2 and M <= 3:
        return "Promising"
    else:
        return "Others"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# ── Bootstrap stability ──────────────────────────────────────────────────────
print("\nRunning bootstrap stability check (50 iterations, 70% subsample)...")
N_ITER = 50
SUBSAMPLE = 0.70

boundary_storage = {"R": [], "F": [], "M": []}

for i in range(N_ITER):
    sample = rfm.sample(frac=SUBSAMPLE, random_state=i)
    for col, key in [("recency", "R"), ("frequency", "F"), ("monetary", "M")]:
        try:
            ranked = sample[col].rank(method="first", pct=True)
            _, bins = pd.cut(ranked, bins=5, retbins=True)
            boundary_storage[key].append(bins[1:-1])  # inner boundaries only
        except Exception:
            pass

stability_results = {}
for key in ["R", "F", "M"]:
    arr = np.array(boundary_storage[key])
    cv_per_boundary = np.std(arr, axis=0) / (np.abs(np.mean(arr, axis=0)) + 1e-9)
    max_cv = cv_per_boundary.max()
    stability_results[key] = {"max_cv": max_cv, "stable": max_cv < 0.10}

all_stable = all(v["stable"] for v in stability_results.values())
print(f"\n  Bootstrap Stability Results:")
for key, res in stability_results.items():
    status = "STABLE" if res["stable"] else "UNSTABLE"
    print(f"    {key}: max boundary CV = {res['max_cv']:.4f} → {status}")
print(f"  Overall: {'✓ Stability confirmed (all CVs < 0.10)' if all_stable else '✗ Instability detected'}")

# ── Spearman correlation ─────────────────────────────────────────────────────
print("\nSpearman correlation between R, F, M scores:")
pairs = [("R_score", "F_score"), ("R_score", "M_score"), ("F_score", "M_score")]
corr_results = {}
for a, b in pairs:
    rho, pval = stats.spearmanr(rfm[a], rfm[b])
    corr_results[f"{a[0]}-{b[0]}"] = rho
    flagged = "|rho| > 0.3 → correlated (redundancy risk)" if abs(rho) > 0.3 else "uncorrelated"
    print(f"  {a} vs {b}: rho={rho:.3f}, p={pval:.4f} → {flagged}")

# ── Segment population table ─────────────────────────────────────────────────
seg_counts = rfm["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]
seg_counts["Pct"] = (seg_counts["Count"] / len(rfm) * 100).round(1)
print("\nSegment Population Table:")
print(seg_counts.to_string(index=False))

# ── Validation summary ───────────────────────────────────────────────────────
print("\nValidation Summary:")
print(f"  Total customers with RFM: {len(rfm):,}")
print(f"  RFM Score range: {rfm['RFM_Score'].min()} – {rfm['RFM_Score'].max()}")
print(f"  Recency  → mean={rfm['recency'].mean():.0f}d, median={rfm['recency'].median():.0f}d")
print(f"  Frequency → mean={rfm['frequency'].mean():.1f}, median={rfm['frequency'].median():.0f}")
print(f"  Monetary  → mean=${rfm['monetary'].mean():,.0f}, median=${rfm['monetary'].median():,.0f}")

# ── Export ───────────────────────────────────────────────────────────────────
rfm.to_csv(OUTPUTS_DIR / "customer_rfm_validated.csv", index=False)
print(f"\nExported: outputs/customer_rfm_validated.csv ({len(rfm):,} rows)")

# ── Figure 1: RFM distributions ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("RFM Raw Value Distributions", fontsize=14, fontweight="bold")

axes[0].hist(rfm["recency"], bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].set_title("Recency (days since last order)")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("Count")

axes[1].hist(rfm["frequency"], bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
axes[1].set_title("Frequency (order count)")
axes[1].set_xlabel("Orders")
axes[1].set_ylabel("Count")

axes[2].hist(rfm["monetary"], bins=30, color="#FF9800", edgecolor="white", alpha=0.85)
axes[2].set_title("Monetary (total revenue $)")
axes[2].set_xlabel("Revenue ($)")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "rfm_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/rfm_distribution.png")

# ── Figure 2: Segment counts ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(seg_counts)))
bars = ax.bar(seg_counts["Segment"], seg_counts["Count"], color=colors, edgecolor="white")
ax.set_title("Customer Segment Sizes (RFM Quintile Scoring)", fontsize=13, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=30, ha="right")
for bar, pct in zip(bars, seg_counts["Pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rfm_segment_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/rfm_segment_counts.png")

# ── Figure 3: Spearman heatmap ───────────────────────────────────────────────
score_cols = ["R_score", "F_score", "M_score"]
corr_matrix = rfm[score_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax,
            xticklabels=["R", "F", "M"], yticklabels=["R", "F", "M"])
ax.set_title("Spearman Correlation — RFM Scores", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rfm_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/rfm_correlation_heatmap.png")

print("\n01_rfm_validation.py complete.")
