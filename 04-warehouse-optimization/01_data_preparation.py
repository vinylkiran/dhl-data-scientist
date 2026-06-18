"""
01_data_preparation.py
======================
Build distance model and pick-session groupings for warehouse optimization.

DHL Warehouse Optimization — DS Project 4
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS  = BASE_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
locs  = pd.read_csv(DATA_DIR / "warehouse_locations.csv")
tasks = pd.read_csv(DATA_DIR / "wms_tasks.csv")

print(f"  Locations: {len(locs):,} rows")
print(f"  Tasks    : {len(tasks):,} rows")

# ── 1. Location distance model ────────────────────────────────────────────────
# Aisle prefix (letter part) → x-coordinate index
# Aisle is like 'P01', 'S03' — extract the prefix letter as zone family and
# the numeric part as the aisle number within that family.
# We assign a global x-coordinate by converting the full aisle label to an
# integer index (alphabetical rank).

AISLE_ORDER = {a: i for i, a in enumerate(sorted(locs["Aisle"].unique()))}
locs["aisle_x"] = locs["Aisle"].map(AISLE_ORDER).astype(float)
locs["bay_y"]   = locs["Bay"].astype(float)
locs["level_z"] = locs["Level"].astype(float)

# Level-change penalty: each level step costs 2.0 distance units (vertical travel)
LEVEL_PENALTY = 2.0

def get_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    """Euclidean 2D distance + level-change penalty."""
    dx = row_a["aisle_x"] - row_b["aisle_x"]
    dy = row_a["bay_y"]   - row_b["bay_y"]
    level_diff = abs(row_a["level_z"] - row_b["level_z"])
    return float(np.sqrt(dx**2 + dy**2) + LEVEL_PENALTY * level_diff)

# Save location metadata (not the full N×N matrix — 2640 locations would be 7M pairs)
loc_meta = locs[["Location_ID", "Warehouse_ID", "Zone", "Aisle", "Bay", "Level",
                  "aisle_x", "bay_y", "level_z", "Capacity_Units", "Storage_Type"]].copy()
loc_meta.to_csv(OUTPUTS / "location_distances.csv", index=False)
print(f"\nLocation metadata saved → outputs/location_distances.csv")
print(f"  Total locations : {len(locs):,}")
print(f"  Warehouses      : {locs['Warehouse_ID'].nunique()}")
print(f"  Zone breakdown  :")
for zone, cnt in locs["Zone"].value_counts().items():
    print(f"    {zone:12s}: {cnt:,}")

# ── 2. Pick-session groupings ─────────────────────────────────────────────────
# Focus on Pick tasks only
picks = tasks[tasks["Task_Type"] == "Pick"].copy()
print(f"\nPick tasks: {len(picks):,} (of {len(tasks):,} total)")

# Create short Shift label
def shorten_shift(s: str) -> str:
    if "Morning"   in s: return "Morning"
    if "Afternoon" in s: return "Afternoon"
    if "Night"     in s: return "Night"
    return s

picks["Shift_Short"] = picks["Shift"].apply(shorten_shift)

# Group by Warehouse_ID + Task_Date + Shift → one picking session
sessions_raw = (
    picks.groupby(["Warehouse_ID", "Task_Date", "Shift_Short"])["SKU_ID"]
    .agg(list)
    .reset_index()
)
sessions_raw.columns = ["warehouse_id", "date", "shift", "sku_list_raw"]

# Distinct SKUs per session
sessions_raw["sku_set"]  = sessions_raw["sku_list_raw"].apply(lambda x: list(set(x)))
sessions_raw["n_skus"]   = sessions_raw["sku_set"].apply(len)
sessions_raw["n_tasks"]  = sessions_raw["sku_list_raw"].apply(len)

# Assign session_id
sessions_raw = sessions_raw.reset_index(drop=True)
sessions_raw["session_id"] = ["SES-" + str(i).zfill(6) for i in range(len(sessions_raw))]

print(f"\nTotal sessions: {len(sessions_raw):,}")
print(f"Session size distribution (distinct SKUs):")
for cutoff in [1, 5, 10, 15, 20, 50]:
    pct = (sessions_raw["n_skus"] <= cutoff).mean() * 100
    print(f"  ≤{cutoff:2d} SKUs: {pct:.1f}%")

# Filter to sessions with 5–15 distinct SKUs
sessions_filtered = sessions_raw[
    sessions_raw["n_skus"].between(5, 15)
].copy()

print(f"\nSessions with 5–15 distinct SKUs: {len(sessions_filtered):,}")
print(f"Size distribution within 5–15 range:")
print(sessions_filtered["n_skus"].value_counts().sort_index().to_string())

# Build output dataframe
out = sessions_filtered[[
    "session_id", "warehouse_id", "date", "shift", "n_skus", "n_tasks"
]].copy()
out["sku_list"] = sessions_filtered["sku_set"].apply(json.dumps)

out = out[["session_id", "warehouse_id", "date", "shift", "sku_list", "n_skus", "n_tasks"]]
out.to_csv(OUTPUTS / "pick_sessions.csv", index=False)
print(f"\nPick sessions saved → outputs/pick_sessions.csv")
print(f"  Rows: {len(out):,}")
print(f"  Columns: {out.columns.tolist()}")
print("\nDone — 01_data_preparation.py complete.")
