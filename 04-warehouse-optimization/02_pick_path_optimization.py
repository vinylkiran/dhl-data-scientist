"""
02_pick_path_optimization.py
============================
Compare three pick-path planning approaches on a 200-session sample.
Methods: (1) Naive order, (2) Nearest-neighbor heuristic, (3) 2-opt improvement.
Exact optimal for sessions with ≤8 stops.

DHL Warehouse Optimization — DS Project 4
"""

import json
import time
import itertools
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

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
sessions = pd.read_csv(OUTPUTS / "pick_sessions.csv")
loc_meta = pd.read_csv(OUTPUTS / "location_distances.csv")
locs_raw = pd.read_csv(DATA_DIR / "warehouse_locations.csv")
tasks    = pd.read_csv(DATA_DIR / "wms_tasks.csv")

print(f"  Sessions : {len(sessions):,}")
print(f"  Locations: {len(loc_meta):,}")

# ── Build SKU → primary location mapping ─────────────────────────────────────
# Each SKU needs a primary warehouse location. Since SKU_ID doesn't appear in
# warehouse_locations.csv directly, we assign each SKU a location deterministically
# based on pick frequency: SKUs are ranked by total picks; locations are ranked
# by zone (Pick_Face first), then assigned in rank order.
# This simulates a realistic ABC-class slotting.

picks_only = tasks[tasks["Task_Type"] == "Pick"].copy()

# Frequency per (Warehouse_ID, SKU_ID)
sku_freq = (
    picks_only.groupby(["Warehouse_ID", "SKU_ID"])
    .size()
    .reset_index(name="pick_count")
)

# Active Pick_Face locations per warehouse, sorted by aisle_x then bay_y
active_locs = loc_meta[loc_meta["Zone"].isin(["Pick_Face", "Reserve", "Bulk"])].copy()
active_locs = active_locs.sort_values(["Warehouse_ID", "Zone", "aisle_x", "bay_y"])

# Assign SKUs to locations: within each warehouse, rank SKUs by pick_count desc
# and cycle through available locations
def assign_sku_locations(sku_freq_wh: pd.DataFrame, locs_wh: pd.DataFrame) -> pd.DataFrame:
    skus   = sku_freq_wh.sort_values("pick_count", ascending=False)["SKU_ID"].tolist()
    lids   = locs_wh["Location_ID"].tolist()
    # Cycle locations if more SKUs than locations
    assignments = []
    for i, sku in enumerate(skus):
        loc = lids[i % len(lids)]
        assignments.append({"SKU_ID": sku, "Location_ID": loc})
    return pd.DataFrame(assignments)

wh_assignments = []
for wh_id in sku_freq["Warehouse_ID"].unique():
    sku_wh  = sku_freq[sku_freq["Warehouse_ID"] == wh_id]
    locs_wh = active_locs[active_locs["Warehouse_ID"] == wh_id]
    if len(locs_wh) == 0:
        locs_wh = loc_meta[loc_meta["Warehouse_ID"] == wh_id]
    df = assign_sku_locations(sku_wh, locs_wh)
    df["Warehouse_ID"] = wh_id
    wh_assignments.append(df)

sku_loc_map = pd.concat(wh_assignments, ignore_index=True)
# Merge with coordinate info
sku_loc_map = sku_loc_map.merge(
    loc_meta[["Location_ID", "aisle_x", "bay_y", "level_z"]],
    on="Location_ID", how="left"
)
# Build fast lookup: (Warehouse_ID, SKU_ID) → (aisle_x, bay_y, level_z)
sku_loc_map["wh_sku"] = sku_loc_map["Warehouse_ID"] + "|" + sku_loc_map["SKU_ID"]
sku_coord = sku_loc_map.set_index("wh_sku")[["aisle_x", "bay_y", "level_z"]].to_dict("index")

print(f"  SKU-location assignments: {len(sku_loc_map):,}")

# ── Distance function ─────────────────────────────────────────────────────────
LEVEL_PENALTY = 2.0

def get_distance(ax, ay, az, bx, by, bz) -> float:
    return float(np.sqrt((ax - bx)**2 + (ay - by)**2) + LEVEL_PENALTY * abs(az - bz))

def path_distance(coords: list) -> float:
    """Total distance for an ordered list of (x, y, z) tuples, starting from depot."""
    if not coords:
        return 0.0
    depot = (0.0, 0.0, 1.0)
    total = get_distance(*depot, *coords[0])
    for i in range(len(coords) - 1):
        total += get_distance(*coords[i], *coords[i+1])
    return total

# ── Path optimization methods ─────────────────────────────────────────────────

def naive_path(coords: list) -> float:
    """Total distance in original pick order."""
    return path_distance(coords)

def nearest_neighbor(coords: list) -> tuple:
    """Greedy nearest-neighbor from depot (0,0,1)."""
    if not coords:
        return [], 0.0
    remaining = list(range(len(coords)))
    cur = (0.0, 0.0, 1.0)
    route = []
    while remaining:
        dists = [get_distance(*cur, *coords[i]) for i in remaining]
        best  = remaining[int(np.argmin(dists))]
        route.append(best)
        cur = coords[best]
        remaining.remove(best)
    ordered = [coords[i] for i in route]
    return ordered, path_distance(ordered)

def two_opt(coords: list, max_iter: int = 100) -> tuple:
    """2-opt improvement starting from nearest-neighbor solution."""
    if len(coords) <= 2:
        route, d = nearest_neighbor(coords)
        return route, d
    route, _ = nearest_neighbor(coords)
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # Cost before: depot→…→route[i]→route[i+1]→…→route[j]→route[j+1]→…
                # Cost after : depot→…→route[i]→route[j]→…→route[i+1]→route[j+1]→…
                a, b = route[i], route[i+1]
                c, d = route[j], route[(j+1) % len(route)] if j+1 < len(route) else None
                old = get_distance(*a, *b)
                new = get_distance(*a, *c)
                if d is not None:
                    old += get_distance(*c, *d)
                    new += get_distance(*b, *d)
                else:
                    pass
                if new < old - 1e-10:
                    route[i+1:j+1] = route[i+1:j+1][::-1]
                    improved = True
    return route, path_distance(route)

def exact_optimal(coords: list) -> float:
    """Exact brute-force over all permutations (only for ≤8 stops)."""
    if len(coords) > 8:
        return None
    best = float("inf")
    for perm in itertools.permutations(range(len(coords))):
        ordered = [coords[i] for i in perm]
        d = path_distance(ordered)
        if d < best:
            best = d
    return best

# ── Sample 200 sessions ───────────────────────────────────────────────────────
sample = sessions.sample(n=min(200, len(sessions)), random_state=42).reset_index(drop=True)
print(f"\nSampling {len(sample)} sessions for path optimization …")

results = []
skipped = 0

for _, row in sample.iterrows():
    sid     = row["session_id"]
    wh_id   = row["warehouse_id"]
    sku_lst = json.loads(row["sku_list"])

    # Get coordinates for each SKU in this session
    coords = []
    for sku in sku_lst:
        key = f"{wh_id}|{sku}"
        if key in sku_coord:
            c = sku_coord[key]
            coords.append((c["aisle_x"], c["bay_y"], c["level_z"]))
    # Deduplicate coordinates
    coords_unique = list(dict.fromkeys(coords))
    if len(coords_unique) < 2:
        skipped += 1
        continue

    n_stops = len(coords_unique)

    # 1. Naive
    t0 = time.perf_counter()
    d_naive = naive_path(coords_unique)
    t_naive = (time.perf_counter() - t0) * 1000

    # 2. Nearest-neighbor
    t0 = time.perf_counter()
    _, d_nn = nearest_neighbor(coords_unique)
    t_nn = (time.perf_counter() - t0) * 1000

    # 3. 2-opt
    t0 = time.perf_counter()
    _, d_2opt = two_opt(coords_unique)
    t_2opt = (time.perf_counter() - t0) * 1000

    # Exact optimal for ≤8 stops
    d_opt = None
    if n_stops <= 8:
        t0 = time.perf_counter()
        d_opt = exact_optimal(coords_unique)
        _ = (time.perf_counter() - t0) * 1000

    # Gap calculations
    def gap(d_heuristic, d_exact):
        if d_exact is None or d_exact == 0:
            return None
        return (d_heuristic - d_exact) / d_exact * 100

    for method, d, t in [("naive", d_naive, t_naive),
                          ("nearest_neighbor", d_nn, t_nn),
                          ("two_opt", d_2opt, t_2opt)]:
        results.append({
            "session_id"      : sid,
            "method"          : method,
            "total_distance"  : d,
            "compute_time_ms" : t,
            "n_stops"         : n_stops,
            "optimal_distance": d_opt,
            "gap_pct"         : gap(d, d_opt),
        })

print(f"  Processed: {len(sample) - skipped} sessions ({skipped} skipped — too few unique locations)")

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUTS / "path_optimization_results.csv", index=False)
print(f"  Results saved → outputs/path_optimization_results.csv  ({len(df_res):,} rows)")

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n=== METHOD SUMMARY ===")
summary = df_res.groupby("method").agg(
    mean_distance    = ("total_distance", "mean"),
    median_distance  = ("total_distance", "median"),
    mean_compute_ms  = ("compute_time_ms", "mean"),
    n_sessions       = ("session_id", "nunique"),
).round(3)
print(summary.to_string())

# Gap from optimal (sessions ≤8 stops only)
gap_df = df_res[df_res["gap_pct"].notna()]
if len(gap_df) > 0:
    print(f"\n=== GAP FROM OPTIMAL (sessions ≤8 stops, n={gap_df['session_id'].nunique()}) ===")
    gap_summary = gap_df.groupby("method")["gap_pct"].agg(["mean", "median"]).round(2)
    print(gap_summary.to_string())

# Distance reduction vs naive
naive_dist = df_res[df_res["method"] == "naive"][["session_id", "total_distance"]].rename(
    columns={"total_distance": "naive_dist"})
for method in ["nearest_neighbor", "two_opt"]:
    m_dist = df_res[df_res["method"] == method][["session_id", "total_distance"]].rename(
        columns={"total_distance": "method_dist"})
    merged = naive_dist.merge(m_dist, on="session_id")
    pct_red = ((merged["naive_dist"] - merged["method_dist"]) / merged["naive_dist"] * 100).mean()
    print(f"  {method} vs naive: {pct_red:.1f}% distance reduction on average")

# ── Figures ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
METHOD_LABELS = {
    "naive": "Naive\n(current order)",
    "nearest_neighbor": "Nearest\nNeighbor",
    "two_opt": "2-Opt\nImproved",
}
df_plot = df_res.copy()
df_plot["Method"] = df_plot["method"].map(METHOD_LABELS)

# Figure 1: Distance box plots
fig, ax = plt.subplots(figsize=(9, 6))
order = ["Naive\n(current order)", "Nearest\nNeighbor", "2-Opt\nImproved"]
sns.boxplot(data=df_plot, x="Method", y="total_distance", order=order, ax=ax,
            palette=["#E8A020", "#1F77B4", "#2CA02C"], width=0.5)
ax.set_title("Pick-Path Total Distance by Planning Method\n(200-session sample)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Method", fontsize=12)
ax.set_ylabel("Total Distance (distance units)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "path_comparison_distance.png", dpi=150)
plt.close()
print("\nFigure saved → figures/path_comparison_distance.png")

# Figure 2: Compute time vs n_stops scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = {"naive": "#E8A020", "nearest_neighbor": "#1F77B4", "two_opt": "#2CA02C"}
for ax, method in zip(axes, ["naive", "nearest_neighbor", "two_opt"]):
    subset = df_res[df_res["method"] == method]
    ax.scatter(subset["n_stops"], subset["compute_time_ms"],
               color=colors[method], alpha=0.5, s=20)
    ax.set_title(METHOD_LABELS[method].replace("\n", " "), fontsize=11)
    ax.set_xlabel("Number of Stops", fontsize=10)
    ax.set_ylabel("Compute Time (ms)", fontsize=10)
    ax.set_yscale("log")
fig.suptitle("Compute Time vs Session Size by Method", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES / "path_comparison_compute.png", dpi=150)
plt.close()
print("Figure saved → figures/path_comparison_compute.png")
print("\nDone — 02_pick_path_optimization.py complete.")
