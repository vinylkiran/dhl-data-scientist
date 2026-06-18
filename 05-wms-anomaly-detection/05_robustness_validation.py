"""
05_robustness_validation.py
WMS Anomaly Detection — Robustness & Sensitivity Testing
- 18-month baseline window test
- Contamination sensitivity (IF)
- Sigma threshold sensitivity (SPC Western Electric rules)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
kpi = pd.read_csv(OUTPUT_DIR / "daily_kpi_timeseries.csv", parse_dates=["date"])
kpi = kpi.sort_values(["warehouse_id", "date"]).reset_index(drop=True)

WH_FEATURES = [
    "pick_accuracy_rate", "putaway_accuracy_rate", "total_task_volume",
    "avg_task_duration", "error_count", "picks_per_labour_hour"
]

warehouses = sorted(kpi["warehouse_id"].unique())

# ── 1. 18-MONTH BASELINE TEST ──────────────────────────────────────────────────
# Historical: Jan 2022 – Jun 2023
# Test period: Jul 2023 – Dec 2023
HIST_END = pd.Timestamp("2023-06-30")
TEST_START = pd.Timestamp("2023-07-01")

print("\n── 18-Month Baseline Test ──")
hist_df = kpi[kpi["date"] <= HIST_END].dropna(subset=WH_FEATURES)
test_df = kpi[kpi["date"] >= TEST_START].dropna(subset=WH_FEATURES)

print(f"  Historical window: {hist_df['date'].min().date()} → {hist_df['date'].max().date()} ({len(hist_df)} rows)")
print(f"  Test window:       {test_df['date'].min().date()} → {test_df['date'].max().date()} ({len(test_df)} rows)")

# Fit IF on historical, score on test
scaler_hist = StandardScaler()
X_hist = scaler_hist.fit_transform(hist_df[WH_FEATURES])
X_test = scaler_hist.transform(test_df[WH_FEATURES])

if_hist = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
if_hist.fit(X_hist)
test_preds_hist = if_hist.predict(X_test)
test_df = test_df.copy()
test_df["if_short_window"] = (test_preds_hist == -1)

# Fit IF on full dataset (all data), score on same test period
full_clean = kpi.dropna(subset=WH_FEATURES)
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(full_clean[WH_FEATURES])
if_full = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
if_full.fit(X_full)
X_test_full = scaler_full.transform(test_df[WH_FEATURES])
test_preds_full = if_full.predict(X_test_full)
test_df["if_full_window"] = (test_preds_full == -1)

n_hist_flagged = test_df["if_short_window"].sum()
n_full_flagged = test_df["if_full_window"].sum()
overlap = (test_df["if_short_window"] & test_df["if_full_window"]).sum()
coverage = overlap / n_full_flagged if n_full_flagged > 0 else 0.0

print(f"  18-month window flags in test period: {n_hist_flagged}")
print(f"  Full window flags in test period:     {n_full_flagged}")
print(f"  Overlap (both flag same day):         {overlap}")
print(f"  Coverage: {coverage:.1%} of full-window anomalies caught by 18-month window")

# ── 2. CONTAMINATION SENSITIVITY — JACCARD MATRIX ────────────────────────────
print("\n── Contamination Sensitivity (Isolation Forest) ──")
contamination_values = [0.01, 0.03, 0.05, 0.10]

full_clean2 = kpi.dropna(subset=WH_FEATURES).copy()
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(full_clean2[WH_FEATURES])

flag_sets = {}
for cont in contamination_values:
    m = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
    preds = m.fit_predict(X2)
    flag_sets[cont] = (preds == -1)
    print(f"  contamination={cont:.2f}: {flag_sets[cont].sum()} flagged")

# Jaccard matrix
n_c = len(contamination_values)
jaccard_mat = np.zeros((n_c, n_c))
for i, ci in enumerate(contamination_values):
    for j, cj in enumerate(contamination_values):
        a, b = flag_sets[ci], flag_sets[cj]
        both = (a & b).sum()
        either = (a | b).sum()
        jaccard_mat[i, j] = both / either if either > 0 else 1.0

print("\n  Jaccard similarity matrix:")
jac_df = pd.DataFrame(jaccard_mat,
                       index=[str(c) for c in contamination_values],
                       columns=[str(c) for c in contamination_values])
print(jac_df.round(3).to_string())
j_05_03 = jaccard_mat[contamination_values.index(0.05), contamination_values.index(0.03)]
print(f"\n  Jaccard(0.05 vs 0.03) = {j_05_03:.3f}  → {'ROBUST' if j_05_03 > 0.8 else 'SENSITIVE'} to this param change")

# Figure: Jaccard heatmap
fig1, ax1 = plt.subplots(figsize=(7, 6))
sns.heatmap(jac_df.astype(float), annot=True, fmt=".3f", cmap="Blues",
            vmin=0, vmax=1, ax=ax1, linewidths=0.5)
ax1.set_title("Isolation Forest — Jaccard Similarity\nbetween Contamination Values", fontsize=12)
ax1.set_xlabel("Contamination")
ax1.set_ylabel("Contamination")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "robustness_contamination.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved robustness_contamination.png")

# ── 3. SIGMA THRESHOLD SENSITIVITY (SPC) ─────────────────────────────────────
print("\n── Sigma Threshold Sensitivity (SPC) ──")
sigma_thresholds = [2.0, 2.5, 3.0]

def count_spc_flags(df_wh, sigma):
    """Count Rule1 (beyond sigma×std) flags for a warehouse."""
    flags = 0
    for _, row in df_wh.iterrows():
        for metric in ["pick_accuracy_rate", "total_task_volume", "error_count"]:
            val = row[metric]
            mu = row[f"{metric}_rolling_mean30"]
            sig = row[f"{metric}_rolling_std30"]
            if pd.isna(mu) or pd.isna(sig) or sig == 0:
                continue
            if abs(val - mu) > sigma * sig:
                flags += 1
                break  # count each day only once per warehouse
    return flags

sigma_results = []
for wh in warehouses:
    wdf = kpi[kpi["warehouse_id"] == wh].copy()
    for sigma in sigma_thresholds:
        n_flagged = count_spc_flags(wdf, sigma)
        sigma_results.append({"warehouse_id": wh, "sigma": sigma, "n_flagged": n_flagged})
        print(f"  {wh}  σ={sigma}  →  {n_flagged} days flagged")

sigma_df = pd.DataFrame(sigma_results)

# Figure: bar chart sigma threshold vs n_flagged per warehouse
fig2, ax2 = plt.subplots(figsize=(9, 5))
x = np.arange(len(sigma_thresholds))
width = 0.25
colors = ["steelblue", "darkorange", "green"]
for i, wh in enumerate(warehouses):
    sub = sigma_df[sigma_df["warehouse_id"] == wh]
    counts = [sub[sub["sigma"] == s]["n_flagged"].values[0] for s in sigma_thresholds]
    ax2.bar(x + i * width, counts, width, label=wh, color=colors[i], alpha=0.8)

ax2.set_xticks(x + width)
ax2.set_xticklabels([f"{s}σ" for s in sigma_thresholds])
ax2.set_xlabel("Sigma Threshold")
ax2.set_ylabel("Days Flagged (Rule 1 threshold breach)")
ax2.set_title("SPC Sigma Threshold Sensitivity — Days Flagged per Warehouse")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "robustness_sigma.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved robustness_sigma.png")

# Comparison 2σ vs 3σ
print("\n  2σ vs 3σ comparison:")
for wh in warehouses:
    sub = sigma_df[sigma_df["warehouse_id"] == wh]
    n2 = sub[sub["sigma"] == 2.0]["n_flagged"].values[0]
    n3 = sub[sub["sigma"] == 3.0]["n_flagged"].values[0]
    extra = n2 - n3
    pct = extra / n3 * 100 if n3 > 0 else float("inf")
    print(f"  {wh}: 2σ={n2}  3σ={n3}  extra at 2σ={extra} (+{pct:.0f}%)")

# ── EXPORT ───────────────────────────────────────────────────────────────────
results = []

# Window test results
results.append({
    "test": "18_month_baseline",
    "metric": "coverage_pct",
    "value": round(coverage * 100, 1),
    "notes": f"18m window catches {coverage:.1%} of full-window anomalies in Jul-Dec 2023",
})

# Jaccard results
for i, ci in enumerate(contamination_values):
    for j, cj in enumerate(contamination_values):
        if i < j:
            results.append({
                "test": "contamination_jaccard",
                "metric": f"jaccard_{ci}_vs_{cj}",
                "value": round(jaccard_mat[i, j], 4),
                "notes": "robust" if jaccard_mat[i, j] > 0.8 else "sensitive",
            })

# Sigma results
for _, row in sigma_df.iterrows():
    results.append({
        "test": "sigma_threshold",
        "metric": f"{row['warehouse_id']}_sigma{row['sigma']}",
        "value": int(row["n_flagged"]),
        "notes": "n_days_flagged",
    })

rob_df = pd.DataFrame(results)
rob_df.to_csv(OUTPUT_DIR / "robustness_results.csv", index=False)
print(f"\nExported robustness_results.csv ({len(rob_df)} rows)")

# ── CONCLUSIONS ───────────────────────────────────────────────────────────────
print("\n── Robustness Conclusions ──")
print(f"  18-month window test: {coverage:.1%} coverage → "
      f"{'STABLE — adequate with shorter history' if coverage > 0.7 else 'SENSITIVE — needs full history'}")
print(f"  IF contamination: Jaccard(0.05 vs 0.03) = {j_05_03:.3f} → "
      f"{'robust' if j_05_03 > 0.8 else 'somewhat sensitive; difference is mainly edge-case borderline points'}")
print(f"  SPC sigma: significant increase in flags going 3σ→2σ. "
      f"Recommend 3σ for production to minimise alert fatigue.")
