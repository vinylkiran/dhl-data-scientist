"""
03_ml_anomaly_detection.py
WMS Anomaly Detection — Unsupervised ML (Isolation Forest + LOF)
Warehouse-day level and operator-day level anomaly detection.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
wh_df = pd.read_csv(OUTPUT_DIR / "daily_kpi_timeseries.csv", parse_dates=["date"])
op_df = pd.read_csv(OUTPUT_DIR / "operator_daily_timeseries.csv", parse_dates=["date"])

warehouses = sorted(wh_df["warehouse_id"].unique())

# ── WAREHOUSE-LEVEL FEATURE MATRIX ────────────────────────────────────────────
WH_FEATURES = [
    "pick_accuracy_rate", "putaway_accuracy_rate", "total_task_volume",
    "avg_task_duration", "error_count", "picks_per_labour_hour"
]

wh_df_clean = wh_df.dropna(subset=WH_FEATURES).copy()
scaler_wh = StandardScaler()
X_wh = scaler_wh.fit_transform(wh_df_clean[WH_FEATURES])

# ── CONTAMINATION SWEEP (Isolation Forest) ────────────────────────────────────
contamination_values = [0.01, 0.03, 0.05, 0.10]
n_flagged_sweep = []

print("\nContamination sweep (Isolation Forest) ...")
for cont in contamination_values:
    if_sweep = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
    preds = if_sweep.fit_predict(X_wh)
    n_flagged = (preds == -1).sum()
    n_flagged_sweep.append(n_flagged)
    print(f"  contamination={cont:.2f}  →  {n_flagged} flagged ({n_flagged/len(X_wh)*100:.1f}%)")

# ── PRODUCTION IF (contamination=0.05) ────────────────────────────────────────
print("\nFitting production Isolation Forest (contamination=0.05) ...")
if_model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
if_preds = if_model.fit_predict(X_wh)
if_scores = -if_model.score_samples(X_wh)  # higher = more anomalous

wh_df_clean = wh_df_clean.copy()
wh_df_clean["if_prediction"] = if_preds
wh_df_clean["if_score"] = if_scores
wh_df_clean["if_anomaly"] = (if_preds == -1)

print(f"  IF flagged: {wh_df_clean['if_anomaly'].sum()} / {len(wh_df_clean)}")

# ── LOF (warehouse level) ─────────────────────────────────────────────────────
print("\nFitting LOF (warehouse level) ...")
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_preds = lof_model.fit_predict(X_wh)
lof_scores = -lof_model.negative_outlier_factor_

wh_df_clean["lof_prediction"] = lof_preds
wh_df_clean["lof_score"] = lof_scores
wh_df_clean["lof_anomaly"] = (lof_preds == -1)

print(f"  LOF flagged: {wh_df_clean['lof_anomaly'].sum()} / {len(wh_df_clean)}")

# ── OPERATOR-LEVEL FEATURES ───────────────────────────────────────────────────
OP_FEATURES = ["tasks_completed", "accuracy_rate", "avg_duration"]
op_df_clean = op_df.dropna(subset=OP_FEATURES).copy()

op_if_records = []
op_lof_records = []

print("\nFitting operator-level models (per warehouse) ...")
for wh in warehouses:
    op_wh = op_df_clean[op_df_clean["warehouse_id"] == wh].copy()
    if len(op_wh) < 20:
        print(f"  {wh}: too few rows ({len(op_wh)}), skipping")
        continue

    scaler_op = StandardScaler()
    X_op = scaler_op.fit_transform(op_wh[OP_FEATURES])

    # IF
    if_op = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    if_op_preds = if_op.fit_predict(X_op)
    if_op_scores = -if_op.score_samples(X_op)

    op_wh = op_wh.copy()
    op_wh["if_prediction"] = if_op_preds
    op_wh["if_score"] = if_op_scores
    op_wh["if_anomaly"] = (if_op_preds == -1)
    op_if_records.append(op_wh)

    # LOF
    lof_op = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_op_preds = lof_op.fit_predict(X_op)
    lof_op_scores = -lof_op.negative_outlier_factor_

    op_wh2 = op_df_clean[op_df_clean["warehouse_id"] == wh].copy()
    op_wh2["lof_prediction"] = lof_op_preds
    op_wh2["lof_score"] = lof_op_scores
    op_wh2["lof_anomaly"] = (lof_op_preds == -1)
    op_lof_records.append(op_wh2)

    print(f"  {wh}: IF={op_wh['if_anomaly'].sum()} LOF={(lof_op_preds == -1).sum()} flagged")

op_if_df = pd.concat(op_if_records, ignore_index=True)
op_lof_df = pd.concat(op_lof_records, ignore_index=True)

# ── FIGURE 1: Time series with IF anomalies per warehouse ─────────────────────
print("\nGenerating figures ...")
fig1, axes = plt.subplots(len(warehouses), 1, figsize=(14, 4 * len(warehouses)), sharex=False)
if len(warehouses) == 1:
    axes = [axes]

for ax, wh in zip(axes, warehouses):
    sub = wh_df_clean[wh_df_clean["warehouse_id"] == wh].sort_values("date")
    normal = sub[~sub["if_anomaly"]]
    anomalous = sub[sub["if_anomaly"]]

    ax.plot(sub["date"], sub["pick_accuracy_rate"], color="steelblue",
            linewidth=0.8, label="pick_accuracy_rate", alpha=0.8)
    if len(anomalous):
        ax.scatter(anomalous["date"], anomalous["pick_accuracy_rate"],
                   color="red", s=40, zorder=5, label="IF Anomaly")
    ax.set_title(f"{wh} — Isolation Forest Anomalies", fontsize=11)
    ax.set_ylabel("Pick Accuracy Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "ml_warehouse_anomalies.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved ml_warehouse_anomalies.png")

# ── FIGURE 2: Operator scatter — LOF anomalies ────────────────────────────────
fig2, axes2 = plt.subplots(1, len(warehouses), figsize=(6 * len(warehouses), 5))
if len(warehouses) == 1:
    axes2 = [axes2]

for ax, wh in zip(axes2, warehouses):
    sub_lof = op_lof_df[op_lof_df["warehouse_id"] == wh]
    normal = sub_lof[~sub_lof["lof_anomaly"]]
    anomalous = sub_lof[sub_lof["lof_anomaly"]]

    ax.scatter(normal["tasks_completed"], normal["accuracy_rate"],
               c="steelblue", alpha=0.4, s=20, label="Normal")
    if len(anomalous):
        ax.scatter(anomalous["tasks_completed"], anomalous["accuracy_rate"],
                   c="red", s=60, zorder=5, label="LOF Anomaly")
    ax.set_title(f"{wh}", fontsize=10)
    ax.set_xlabel("Tasks Completed")
    ax.set_ylabel("Accuracy Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig2.suptitle("Operator Anomalies (LOF) — Accuracy Rate vs Tasks Completed", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ml_operator_anomalies.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved ml_operator_anomalies.png")

# ── FIGURE 3: Contamination sensitivity ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(contamination_values, n_flagged_sweep, marker="o", color="navy", linewidth=2)
for c, n in zip(contamination_values, n_flagged_sweep):
    ax3.annotate(f"{n}", (c, n), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax3.axvline(0.05, color="red", linestyle="--", label="Production choice (0.05)")
ax3.set_xlabel("Contamination Parameter")
ax3.set_ylabel("Number of Flagged Anomalies")
ax3.set_title("Isolation Forest — Contamination Sensitivity")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ml_contamination_sensitivity.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved ml_contamination_sensitivity.png")

# ── EXPORT ───────────────────────────────────────────────────────────────────
wh_if_out = wh_df_clean[["date", "warehouse_id", "if_score", "if_anomaly"]].copy()
wh_if_out["level"] = "warehouse"
wh_if_out["method"] = "IF"
wh_if_out = wh_if_out.rename(columns={"warehouse_id": "entity_id",
                                        "if_score": "anomaly_score",
                                        "if_anomaly": "is_anomaly"})

wh_lof_out = wh_df_clean[["date", "warehouse_id", "lof_score", "lof_anomaly"]].copy()
wh_lof_out["level"] = "warehouse"
wh_lof_out["method"] = "LOF"
wh_lof_out = wh_lof_out.rename(columns={"warehouse_id": "entity_id",
                                          "lof_score": "anomaly_score",
                                          "lof_anomaly": "is_anomaly"})

op_if_out = op_if_df[["date", "operator_id", "warehouse_id", "if_score", "if_anomaly"]].copy()
op_if_out["entity_id"] = op_if_out["operator_id"] + "@" + op_if_out["warehouse_id"]
op_if_out["level"] = "operator"
op_if_out["method"] = "IF"
op_if_out = op_if_out.rename(columns={"if_score": "anomaly_score", "if_anomaly": "is_anomaly"})
op_if_out = op_if_out.drop(columns=["operator_id", "warehouse_id"])

op_lof_out = op_lof_df[["date", "operator_id", "warehouse_id", "lof_score", "lof_anomaly"]].copy()
op_lof_out["entity_id"] = op_lof_out["operator_id"] + "@" + op_lof_out["warehouse_id"]
op_lof_out["level"] = "operator"
op_lof_out["method"] = "LOF"
op_lof_out = op_lof_out.rename(columns={"lof_score": "anomaly_score", "lof_anomaly": "is_anomaly"})
op_lof_out = op_lof_out.drop(columns=["operator_id", "warehouse_id"])

ml_out = pd.concat([wh_if_out, wh_lof_out, op_if_out, op_lof_out], ignore_index=True)
# Add warehouse_id back for warehouse-level rows
wh_map = wh_df_clean[["date", "warehouse_id"]].copy()
wh_map["entity_id"] = wh_map["warehouse_id"]
ml_out = ml_out.merge(wh_map[["date", "entity_id", "warehouse_id"]], on=["date", "entity_id"], how="left")

ml_out.to_csv(OUTPUT_DIR / "ml_anomalies.csv", index=False)
print(f"\nExported ml_anomalies.csv ({len(ml_out):,} rows)")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n── Flagged anomalies by method × level ──")
print(ml_out.groupby(["method", "level"])["is_anomaly"].sum().to_string())

print("\n── Top 5 most anomalous warehouse-days (IF score) ──")
top5 = (wh_df_clean[wh_df_clean["if_anomaly"]]
        .nlargest(5, "if_score")[["date", "warehouse_id", "pick_accuracy_rate",
                                    "total_task_volume", "error_count", "if_score"]])
print(top5.to_string(index=False))
