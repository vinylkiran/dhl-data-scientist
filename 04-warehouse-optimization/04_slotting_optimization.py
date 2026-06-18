"""
04_slotting_optimization.py
============================
Constrained SKU-to-zone assignment via (a) greedy heuristic and (b) LP/ILP.
Compares estimated travel time: current slotting vs greedy vs LP-optimized.

Zone layout (by aisle_x):
  Bulk      : 0–3   (far from depot/dispatch at 4–5)
  Dispatch  : 4–5   (depot reference point)
  Pick_Face : 6–13  (nearest productive zone to dispatch)
  Reserve   : 16–21 (furthest productive zone)

Current slotting = alphabetical-zone assignment → top SKUs end up in Bulk (worst).
This creates a meaningful before/after comparison.

DHL Warehouse Optimization — DS Project 4
"""

import json
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS  = BASE_DIR / "outputs"
FIGURES  = BASE_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

LEVEL_PENALTY = 2.0

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
locs     = pd.read_csv(DATA_DIR / "warehouse_locations.csv")
tasks    = pd.read_csv(DATA_DIR / "wms_tasks.csv")
sessions = pd.read_csv(OUTPUTS / "pick_sessions.csv")
loc_meta = pd.read_csv(OUTPUTS / "location_distances.csv")
affinity = pd.read_csv(OUTPUTS / "affinity_pairs_validated.csv")
sessions["sku_set"] = sessions["sku_list"].apply(json.loads)

STORAGE_ZONES = ["Pick_Face", "Reserve", "Bulk"]

# ── Pick frequency per (Warehouse_ID, SKU_ID) ────────────────────────────────
picks = tasks[tasks["Task_Type"] == "Pick"].copy()
sku_freq_wh = (
    picks.groupby(["Warehouse_ID", "SKU_ID"])
    .size()
    .reset_index(name="pick_count")
)
# Global SKU frequency
sku_freq = (
    picks.groupby("SKU_ID").size()
    .reset_index(name="pick_count")
    .sort_values("pick_count", ascending=False)
    .reset_index(drop=True)
)
sku_pick_global = dict(zip(sku_freq["SKU_ID"], sku_freq["pick_count"]))
print(f"  Unique SKUs with picks: {len(sku_freq):,}")

# ── Build location pool per warehouse × zone ─────────────────────────────────
# Only active storage zones
active_locs = loc_meta[loc_meta["Zone"].isin(STORAGE_ZONES)].copy()

# Zone distance from depot (aisle_x 4.5 = midpoint of dispatch aisles D01,D02)
DEPOT_X = 4.5

def dist_from_depot(row: pd.Series) -> float:
    return float(abs(row["aisle_x"] - DEPOT_X) + row["bay_y"] * 0.1)

active_locs["depot_dist"] = active_locs.apply(dist_from_depot, axis=1)

# Zone capacity counts (number of distinct locations = number of SKU slots)
zone_loc_counts = (
    active_locs[active_locs["Warehouse_ID"] == "DHL-WH-NJ01"]  # same per wh
    .groupby("Zone").size().to_dict()
)
print(f"\nStorage locations per zone (per warehouse):")
for z, cnt in zone_loc_counts.items():
    print(f"  {z:12s}: {cnt:,}")

# For travel time estimation we need: SKU → (aisle_x, bay_y, level_z)
# We assign each SKU to a specific location in its assigned zone

def build_sku_location_map(sku_zone_map: dict, wh_id: str) -> dict:
    """
    Given {SKU_ID: zone} assignment, assign each SKU to an individual location
    within that zone for the specified warehouse.
    Returns {SKU_ID: (aisle_x, bay_y, level_z)}.
    """
    wh_locs = active_locs[active_locs["Warehouse_ID"] == wh_id].copy()
    wh_locs = wh_locs.sort_values("depot_dist")   # nearest locations first

    zone_loc_pool = {z: [] for z in STORAGE_ZONES}
    for _, r in wh_locs.iterrows():
        zone_loc_pool[r["Zone"]].append((r["aisle_x"], r["bay_y"], r["level_z"]))

    result = {}
    zone_ptr = {z: 0 for z in STORAGE_ZONES}
    for sku, zone in sku_zone_map.items():
        pool = zone_loc_pool.get(zone, zone_loc_pool["Bulk"])
        if len(pool) == 0:
            result[sku] = (10.0, 5.0, 1.0)
        else:
            idx = zone_ptr[zone] % len(pool)
            result[sku] = pool[idx]
            zone_ptr[zone] += 1
    return result

# ── (Current) Status-quo slotting ────────────────────────────────────────────
# Recreate the alphabetical-zone sort used in script 02:
# Bulk (B*) → Pick_Face (P*) → Reserve (S*) → top SKUs end up in Bulk
print("\nBuilding current (alphabetical-zone) slotting …")
current_assignment = {}
for wh_id in sku_freq_wh["Warehouse_ID"].unique():
    wh_skus   = sku_freq_wh[sku_freq_wh["Warehouse_ID"] == wh_id].sort_values(
        "pick_count", ascending=False)["SKU_ID"].tolist()
    wh_alocs  = active_locs[active_locs["Warehouse_ID"] == wh_id].sort_values(
        ["Zone", "aisle_x", "bay_y"])  # alphabetical zone sort: Bulk→Pick_Face→Reserve
    wh_loc_zone = wh_alocs["Zone"].tolist()
    for i, sku in enumerate(wh_skus):
        zone = wh_loc_zone[i % len(wh_loc_zone)]
        current_assignment[sku] = zone

current_zones = pd.Series(current_assignment)
print(f"  Zone distribution under current slotting:")
print(current_zones.value_counts().to_string())

# ── (a) Greedy heuristic slotting ────────────────────────────────────────────
print("\nBuilding greedy slotting …")
greedy_assignment = {}
for wh_id in sku_freq_wh["Warehouse_ID"].unique():
    wh_skus = sku_freq_wh[sku_freq_wh["Warehouse_ID"] == wh_id].sort_values(
        "pick_count", ascending=False)["SKU_ID"].tolist()
    wh_locs = active_locs[active_locs["Warehouse_ID"] == wh_id]
    zone_cap = wh_locs.groupby("Zone").size().to_dict()
    rem = {z: zone_cap.get(z, 0) for z in STORAGE_ZONES}
    for sku in wh_skus:
        for zone in ["Pick_Face", "Reserve", "Bulk"]:
            if rem[zone] > 0:
                greedy_assignment[sku] = zone
                rem[zone] -= 1
                break
        if sku not in greedy_assignment:
            greedy_assignment[sku] = "Bulk"

greedy_zones = pd.Series(greedy_assignment)
print(f"  Zone distribution under greedy slotting:")
print(greedy_zones.value_counts().to_string())

# ── (b) LP joint optimization (top 200 SKUs only) ────────────────────────────
print("\nRunning LP slotting …")
TOP_N     = 200
top_skus  = sku_freq.head(TOP_N)["SKU_ID"].tolist()
top200_set = set(top_skus)

validated_pairs = affinity[affinity["validated"] == True][["sku_a", "sku_b", "lift"]].copy()
aff_top50 = (
    validated_pairs[
        validated_pairs["sku_a"].isin(top200_set) &
        validated_pairs["sku_b"].isin(top200_set)
    ]
    .sort_values("lift", ascending=False)
    .head(50)
    .reset_index(drop=True)
)

# Zone-level capacity (per warehouse, averaged across 3 warehouses)
# Use NJ01 as representative
nj01_caps = (
    active_locs[active_locs["Warehouse_ID"] == "DHL-WH-NJ01"]
    .groupby("Zone").size().to_dict()
)
# LP capacity = number of available slots (capped at TOP_N total)
zone_caps_lp = {}
total_alloc  = 0
for z in ["Pick_Face", "Reserve", "Bulk"]:
    cap = nj01_caps.get(z, 0)
    zone_caps_lp[z] = cap
    total_alloc += cap

ZONE_DIST_SCORE = {"Pick_Face": 1, "Reserve": 6, "Bulk": 10}
LP_METHOD_LABEL = "LP Joint Optimization"

try:
    import pulp
    prob = pulp.LpProblem("SKU_Slotting_LP", pulp.LpMinimize)
    x = {(s, z): pulp.LpVariable(f"x_{i}_{z}", cat="Binary")
         for i, s in enumerate(top_skus) for z in STORAGE_ZONES}

    # Affinity linearization variables
    y = {}
    for _, ar in aff_top50.iterrows():
        for z in STORAGE_ZONES:
            y[(ar["sku_a"], ar["sku_b"], z)] = pulp.LpVariable(
                f"y_{ar.name}_{z}", cat="Binary")

    travel_cost = pulp.lpSum(
        ZONE_DIST_SCORE[z] * sku_pick_global.get(s, 1) * x[(s, z)]
        for s in top_skus for z in STORAGE_ZONES
    )
    aff_bonus = pulp.lpSum(
        0.5 * ar["lift"] * y[(ar["sku_a"], ar["sku_b"], z)]
        for _, ar in aff_top50.iterrows()
        for z in STORAGE_ZONES
        if (ar["sku_a"], ar["sku_b"], z) in y
    ) if len(y) > 0 else 0

    prob += travel_cost - aff_bonus

    for s in top_skus:
        prob += pulp.lpSum(x[(s, z)] for z in STORAGE_ZONES) == 1

    for z in STORAGE_ZONES:
        prob += pulp.lpSum(x[(s, z)] for s in top_skus) <= zone_caps_lp[z]

    for (sa, sb, z) in y:
        prob += y[(sa, sb, z)] <= x[(sa, z)]
        prob += y[(sa, sb, z)] <= x[(sb, z)]
        prob += y[(sa, sb, z)] >= x[(sa, z)] + x[(sb, z)] - 1

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    t_lp = time.perf_counter() - t0
    print(f"  LP status: {pulp.LpStatus[prob.status]} | {t_lp:.1f}s")

    lp_assignment = {}
    if pulp.LpStatus[prob.status] in ["Optimal", "Feasible"]:
        for s in top_skus:
            for z in STORAGE_ZONES:
                if pulp.value(x[(s, z)]) is not None and pulp.value(x[(s, z)]) > 0.5:
                    lp_assignment[s] = z
                    break
    # Fill remaining with greedy
    for sku in sku_freq["SKU_ID"]:
        if sku not in lp_assignment:
            lp_assignment[sku] = greedy_assignment.get(sku, "Bulk")

    lp_zones = pd.Series({k: v for k, v in lp_assignment.items() if k in top200_set})
    print(f"  LP zone distribution (top-200 SKUs):")
    print(lp_zones.value_counts().to_string())

except Exception as e:
    print(f"  LP failed ({e}) — using greedy as LP proxy")
    lp_assignment = greedy_assignment.copy()
    LP_METHOD_LABEL = "Greedy (LP proxy)"

# ── Travel time estimation ────────────────────────────────────────────────────
print("\nEstimating travel times (500-session sample) …")

SESSION_SAMPLE = 500
sample_sess = sessions.sample(n=min(SESSION_SAMPLE, len(sessions)), random_state=42)

# Build per-warehouse SKU→coord maps for all three methods
def make_wh_coord_maps(zone_map: dict) -> dict:
    """Returns {wh_id: {SKU_ID: (ax, ay, az)}}"""
    maps = {}
    for wh_id in ["DHL-WH-NJ01", "DHL-WH-IL02", "DHL-WH-TX03"]:
        wh_map = {s: z for s, z in zone_map.items()}
        maps[wh_id] = build_sku_location_map(wh_map, wh_id)
    return maps

print("  Building coordinate maps …")
current_coord_maps = make_wh_coord_maps(current_assignment)
greedy_coord_maps  = make_wh_coord_maps(greedy_assignment)
lp_coord_maps      = make_wh_coord_maps(lp_assignment)

def nn_travel(coords: list) -> float:
    """Nearest-neighbor travel distance from depot (4.5, 0, 1)."""
    if len(coords) < 2:
        return 0.0
    depot = (DEPOT_X, 0.0, 1.0)
    remaining = list(range(len(coords)))
    cur = depot
    total = 0.0
    while remaining:
        dists = [
            np.sqrt((cur[0]-coords[i][0])**2 + (cur[1]-coords[i][1])**2)
            + LEVEL_PENALTY * abs(cur[2] - coords[i][2])
            for i in remaining
        ]
        best_idx = int(np.argmin(dists))
        total += dists[best_idx]
        cur = coords[remaining[best_idx]]
        remaining.pop(best_idx)
    return total

results = []
for method_name, coord_maps in [
    ("Current (Naive Slotting)", current_coord_maps),
    ("Greedy Heuristic",         greedy_coord_maps),
    (LP_METHOD_LABEL,            lp_coord_maps),
]:
    total_travel = 0.0
    for _, row in sample_sess.iterrows():
        wh_id  = row["warehouse_id"]
        sku_map = coord_maps.get(wh_id, {})
        coords = []
        for sku in row["sku_set"]:
            if sku in sku_map:
                c = sku_map[sku]
                if c not in coords:
                    coords.append(c)
        total_travel += nn_travel(coords)

    results.append({
        "method"      : method_name,
        "total_travel": total_travel,
        "mean_travel" : total_travel / len(sample_sess),
        "n_sessions"  : len(sample_sess),
    })

comp_df = pd.DataFrame(results)
base_tt = comp_df.loc[comp_df["method"] == "Current (Naive Slotting)", "total_travel"].values[0]
if base_tt > 0:
    comp_df["pct_vs_current"] = (base_tt - comp_df["total_travel"]) / base_tt * 100
else:
    comp_df["pct_vs_current"] = 0.0

greedy_tt = comp_df.loc[comp_df["method"] == "Greedy Heuristic", "total_travel"].values[0]
comp_df["pct_vs_greedy"] = 0.0
if greedy_tt > 0:
    comp_df["pct_vs_greedy"] = (greedy_tt - comp_df["total_travel"]) / greedy_tt * 100

comp_df.to_csv(OUTPUTS / "slotting_comparison.csv", index=False)
print(f"\n=== SLOTTING COMPARISON (n={len(sample_sess)} sessions) ===")
for _, r in comp_df.iterrows():
    print(f"  {r['method']:35s}  mean_dist={r['mean_travel']:6.2f}  vs_current={r['pct_vs_current']:+.1f}%  vs_greedy={r['pct_vs_greedy']:+.1f}%")

print(f"\nSaved → outputs/slotting_comparison.csv")
print("\nDone — 04_slotting_optimization.py complete.")
