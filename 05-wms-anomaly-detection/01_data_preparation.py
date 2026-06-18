"""
01_data_preparation.py
WMS Anomaly Detection — Data Preparation
Aggregates wms_tasks.csv into daily KPI time series at warehouse and operator level.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading wms_tasks.csv ...")
df = pd.read_csv(DATA_DIR / "wms_tasks.csv", parse_dates=["Task_Date"])

print(f"Rows: {len(df):,}")
print(f"Date range: {df['Task_Date'].min().date()} → {df['Task_Date'].max().date()}")
print(f"Warehouses: {sorted(df['Warehouse_ID'].unique())}")
print(f"Operators:  {df['Operator_ID'].nunique()}")
print(f"Task types: {df['Task_Type'].value_counts().to_dict()}")

# Error flag: Accuracy_Flag==0 OR Error_Code is not null
df["is_error"] = (df["Accuracy_Flag"] == 0) | df["Error_Code"].notna()

# ── WAREHOUSE-LEVEL DAILY AGGREGATES ──────────────────────────────────────────
def warehouse_daily(grp):
    total = len(grp)
    errors = grp["is_error"].sum()

    picks = grp[grp["Task_Type"] == "Pick"]
    putaways = grp[grp["Task_Type"] == "Putaway"]

    pick_acc = (1 - picks["is_error"].mean()) if len(picks) > 0 else np.nan
    putaway_acc = (1 - putaways["is_error"].mean()) if len(putaways) > 0 else np.nan

    avg_dur = grp["Duration_Min"].mean()
    total_hours = grp["Duration_Min"].sum() / 60.0
    picks_per_labour_hr = len(picks) / total_hours if total_hours > 0 else np.nan

    return pd.Series({
        "total_task_volume": total,
        "error_count": errors,
        "pick_accuracy_rate": pick_acc,
        "putaway_accuracy_rate": putaway_acc,
        "avg_task_duration": avg_dur,
        "picks_per_labour_hour": picks_per_labour_hr,
    })

print("\nAggregating warehouse-level daily KPIs ...")
wh_daily = (
    df.groupby(["Warehouse_ID", "Task_Date"])
    .apply(warehouse_daily, include_groups=False)
    .reset_index()
    .rename(columns={"Task_Date": "date", "Warehouse_ID": "warehouse_id"})
    .sort_values(["warehouse_id", "date"])
)

# ── ROLLING 30-DAY BASELINE per warehouse ────────────────────────────────────
METRICS = ["pick_accuracy_rate", "putaway_accuracy_rate", "total_task_volume",
           "avg_task_duration", "error_count", "picks_per_labour_hour"]

rolling_parts = []
for wh, grp in wh_daily.groupby("warehouse_id"):
    grp = grp.set_index("date").sort_index()
    for m in METRICS:
        rolled = grp[m].rolling(30, min_periods=10)
        grp[f"{m}_rolling_mean30"] = rolled.mean()
        grp[f"{m}_rolling_std30"] = rolled.std()
    rolling_parts.append(grp.reset_index().assign(warehouse_id=wh))

wh_daily = pd.concat(rolling_parts, ignore_index=True)
wh_daily = wh_daily.rename(columns={"date": "date"})

print(f"Warehouse-day rows: {len(wh_daily):,}")
print(f"Warehouses: {wh_daily['warehouse_id'].nunique()}")
print(f"Date range: {wh_daily['date'].min()} → {wh_daily['date'].max()}")

# ── OPERATOR-LEVEL DAILY AGGREGATES ──────────────────────────────────────────
print("\nAggregating operator-level daily KPIs ...")
op_daily = (
    df.groupby(["Operator_ID", "Warehouse_ID", "Task_Date"])
    .agg(
        tasks_completed=("Task_ID", "count"),
        accuracy_rate=("is_error", lambda x: 1 - x.mean()),
        avg_duration=("Duration_Min", "mean"),
    )
    .reset_index()
    .rename(columns={
        "Operator_ID": "operator_id",
        "Warehouse_ID": "warehouse_id",
        "Task_Date": "date",
    })
)

# Filter to active operators (≥5 tasks on a day)
op_daily = op_daily[op_daily["tasks_completed"] >= 5].copy()
print(f"Operator-day rows (≥5 tasks): {len(op_daily):,}")
print(f"Operators: {op_daily['operator_id'].nunique()}")

# ── EXPORT ───────────────────────────────────────────────────────────────────
wh_daily.to_csv(OUTPUT_DIR / "daily_kpi_timeseries.csv", index=False)
op_daily.to_csv(OUTPUT_DIR / "operator_daily_timeseries.csv", index=False)
print(f"\nExported daily_kpi_timeseries.csv  ({len(wh_daily):,} rows)")
print(f"Exported operator_daily_timeseries.csv  ({len(op_daily):,} rows)")

# ── SUMMARY STATS ────────────────────────────────────────────────────────────
print("\n── Summary stats (warehouse-level KPIs) ──")
print(wh_daily[METRICS].describe().round(4).to_string())
