"""
02_statistical_control.py
WMS Anomaly Detection — Statistical Process Control (SPC)
Western Electric run rules + CUSUM on daily warehouse-level KPIs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading daily_kpi_timeseries.csv ...")
df = pd.read_csv(OUTPUT_DIR / "daily_kpi_timeseries.csv", parse_dates=["date"])
df = df.sort_values(["warehouse_id", "date"]).reset_index(drop=True)

warehouses = sorted(df["warehouse_id"].unique())
METRICS_SPC = ["pick_accuracy_rate", "total_task_volume", "error_count"]

# ── WESTERN ELECTRIC RUN RULES ────────────────────────────────────────────────
def apply_western_electric(series, center, std, ucl2, lcl2, ucl3, lcl3):
    """Apply 4 WE run rules; return list of (index, rule_label) tuples."""
    flags = []
    n = len(series)
    vals = series.values
    above_cl = vals > center.values  # True = above center line

    # Rule 1: 1 point beyond 3σ
    for i in range(n):
        if pd.notna(ucl3.iloc[i]) and (vals[i] > ucl3.iloc[i] or vals[i] < lcl3.iloc[i]):
            flags.append((i, "Rule1_3sigma"))

    # Rule 2: 2 of 3 consecutive beyond 2σ on the same side
    for i in range(2, n):
        window_above = [vals[j] > ucl2.iloc[j] for j in range(i-2, i+1) if pd.notna(ucl2.iloc[j])]
        window_below = [vals[j] < lcl2.iloc[j] for j in range(i-2, i+1) if pd.notna(lcl2.iloc[j])]
        if len(window_above) == 3 and sum(window_above) >= 2:
            flags.append((i, "Rule2_2of3_2sigma"))
        elif len(window_below) == 3 and sum(window_below) >= 2:
            flags.append((i, "Rule2_2of3_2sigma"))

    # Rule 3: 8 consecutive on one side of center
    for i in range(7, n):
        window = [pd.notna(center.iloc[j]) and above_cl[j] for j in range(i-7, i+1)]
        window_below = [pd.notna(center.iloc[j]) and not above_cl[j] for j in range(i-7, i+1)]
        if all(window) or all(window_below):
            flags.append((i, "Rule3_8consec_oneside"))

    # Rule 4: 6 consecutive steadily increasing or decreasing
    for i in range(5, n):
        window = vals[i-5:i+1]
        if all(window[j] < window[j+1] for j in range(5)):
            flags.append((i, "Rule4_6consec_trend"))
        elif all(window[j] > window[j+1] for j in range(5)):
            flags.append((i, "Rule4_6consec_trend"))

    return flags


# ── CUSUM ────────────────────────────────────────────────────────────────────
def compute_cusum(series, center, std, k=0.5, h=4.0):
    """
    CUSUM for gradual drift detection.
    k = reference value (0.5σ), h = decision interval (4σ).
    Returns C_plus, C_minus, and boolean flag array.
    """
    n = len(series)
    C_plus = np.zeros(n)
    C_minus = np.zeros(n)
    flag = np.zeros(n, dtype=bool)

    for t in range(1, n):
        mu = center.iloc[t] if pd.notna(center.iloc[t]) else center.dropna().mean()
        sig = std.iloc[t] if pd.notna(std.iloc[t]) else std.dropna().mean()
        if sig == 0 or pd.isna(sig):
            sig = 1e-9

        x = series.iloc[t]
        if pd.isna(x):
            C_plus[t] = C_plus[t-1]
            C_minus[t] = C_minus[t-1]
            continue

        C_plus[t] = max(0, C_plus[t-1] + (x - mu - k * sig))
        C_minus[t] = max(0, C_minus[t-1] - (x - mu + k * sig))

        if C_plus[t] > h * sig or C_minus[t] > h * sig:
            flag[t] = True
            C_plus[t] = 0
            C_minus[t] = 0

    return C_plus, C_minus, flag


# ── COLLECT ANOMALY RECORDS ──────────────────────────────────────────────────
anomaly_records = []

for wh in warehouses:
    wdf = df[df["warehouse_id"] == wh].reset_index(drop=True)
    print(f"\nWarehouse: {wh}  ({len(wdf)} days)")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"SPC Control Chart — {wh}", fontsize=14, fontweight="bold")

    for ax_idx, metric in enumerate(METRICS_SPC):
        ax = axes[ax_idx]
        series = wdf[metric]
        center = wdf[f"{metric}_rolling_mean30"]
        std_col = wdf[f"{metric}_rolling_std30"]
        ucl2 = center + 2 * std_col
        lcl2 = center - 2 * std_col
        ucl3 = center + 3 * std_col
        lcl3 = center - 3 * std_col

        dates = wdf["date"]

        # Plot bands
        ax.fill_between(dates, lcl3, ucl3, alpha=0.10, color="gray", label="3σ band")
        ax.fill_between(dates, lcl2, ucl2, alpha=0.18, color="steelblue", label="2σ band")
        ax.plot(dates, center, color="navy", linewidth=1.2, label="Center (30d mean)")
        ax.plot(dates, ucl3, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.plot(dates, lcl3, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.plot(dates, series, color="black", linewidth=0.7, alpha=0.8, label=metric)

        # WE run rules
        we_flags = apply_western_electric(series, center, std_col, ucl2, lcl2, ucl3, lcl3)

        # Deduplicate by index
        seen_idx = set()
        for idx, rule in we_flags:
            if 0 <= idx < len(wdf):
                row = wdf.iloc[idx]
                key = (row["date"], metric, rule)
                if idx not in seen_idx:
                    seen_idx.add(idx)
                anomaly_records.append({
                    "date": row["date"],
                    "warehouse_id": wh,
                    "metric": metric,
                    "rule_triggered": rule,
                    "value": series.iloc[idx],
                    "center_line": center.iloc[idx],
                    "ucl3": ucl3.iloc[idx],
                    "lcl3": lcl3.iloc[idx],
                })

        # Plot flagged points (Rule1 only on chart for clarity)
        rule1_idx = [i for i, r in we_flags if r == "Rule1_3sigma" and 0 <= i < len(wdf)]
        if rule1_idx:
            ax.scatter(dates.iloc[rule1_idx], series.iloc[rule1_idx],
                       color="red", zorder=5, s=50, label="3σ breach")

        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"spc_control_chart_{wh}.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_path.name}")

    # CUSUM for pick_accuracy_rate
    fig_c, ax_c = plt.subplots(figsize=(14, 5))
    series_c = wdf["pick_accuracy_rate"]
    center_c = wdf["pick_accuracy_rate_rolling_mean30"]
    std_c = wdf["pick_accuracy_rate_rolling_std30"]

    C_plus, C_minus, cusum_flag = compute_cusum(series_c, center_c, std_c)

    mean_sig = std_c.dropna().mean()
    h_line = 4.0 * mean_sig

    ax_c.plot(wdf["date"], C_plus, color="blue", label="CUSUM C+", linewidth=1.0)
    ax_c.plot(wdf["date"], C_minus, color="orange", label="CUSUM C-", linewidth=1.0)
    ax_c.axhline(h_line, color="red", linestyle="--", linewidth=1, label=f"Decision h={h_line:.4f}")
    ax_c.axhline(0, color="black", linewidth=0.5)

    cusum_flag_idx = np.where(cusum_flag)[0]
    if len(cusum_flag_idx):
        ax_c.scatter(wdf["date"].iloc[cusum_flag_idx], C_plus[cusum_flag_idx],
                     color="red", zorder=5, s=60, label="CUSUM Alert")

        for idx in cusum_flag_idx:
            row = wdf.iloc[idx]
            anomaly_records.append({
                "date": row["date"],
                "warehouse_id": wh,
                "metric": "pick_accuracy_rate",
                "rule_triggered": "CUSUM",
                "value": series_c.iloc[idx],
                "center_line": center_c.iloc[idx],
                "ucl3": center_c.iloc[idx] + 3 * std_c.iloc[idx] if pd.notna(std_c.iloc[idx]) else np.nan,
                "lcl3": center_c.iloc[idx] - 3 * std_c.iloc[idx] if pd.notna(std_c.iloc[idx]) else np.nan,
            })

    ax_c.set_title(f"CUSUM Chart — pick_accuracy_rate — {wh}", fontsize=12)
    ax_c.set_xlabel("Date")
    ax_c.set_ylabel("Cumulative Sum")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)
    plt.tight_layout()
    cusum_path = FIGURES_DIR / f"spc_cusum_{wh}.png"
    plt.savefig(cusum_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {cusum_path.name}")


# ── EXPORT ANOMALIES ─────────────────────────────────────────────────────────
anom_df = pd.DataFrame(anomaly_records)
anom_df = anom_df.drop_duplicates().sort_values(["warehouse_id", "date", "metric"])
anom_df.to_csv(OUTPUT_DIR / "spc_anomalies.csv", index=False)
print(f"\nTotal SPC anomaly records: {len(anom_df):,}")

print("\n── Anomalies per warehouse ──")
print(anom_df.groupby("warehouse_id").size().to_string())

print("\n── Breakdown by rule ──")
print(anom_df.groupby("rule_triggered").size().to_string())

print("\n── Breakdown by metric ──")
print(anom_df.groupby("metric").size().to_string())

print("\n── Breakdown by warehouse × metric ──")
print(anom_df.groupby(["warehouse_id", "metric"]).size().to_string())
