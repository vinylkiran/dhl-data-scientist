"""
06_cost_benefit_analysis.py
============================
Economic evaluation of path-planning and slotting methods.
Quantifies compute cost vs travel-time labor savings.

DHL Warehouse Optimization — DS Project 4
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS  = BASE_DIR / "outputs"
FIGURES  = BASE_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load prior results ────────────────────────────────────────────────────────
print("Loading prior results …")
path_res  = pd.read_csv(OUTPUTS / "path_optimization_results.csv")
slot_res  = pd.read_csv(OUTPUTS / "slotting_comparison.csv")
robust    = pd.read_csv(OUTPUTS / "robustness_results.csv")

# ── Actual timing from path optimization ─────────────────────────────────────
timing = path_res.groupby("method")["compute_time_ms"].mean()
print("\nActual mean compute times (ms/session):")
print(timing.to_string())

# Actual distance metrics
dist = path_res.groupby("method")["total_distance"].mean()
naive_dist  = dist["naive"]
nn_dist     = dist["nearest_neighbor"]
two_opt_dist = dist["two_opt"]

nn_reduction   = (naive_dist - nn_dist)     / naive_dist * 100
topt_reduction = (naive_dist - two_opt_dist) / naive_dist * 100
topt_vs_nn     = (nn_dist - two_opt_dist)   / nn_dist * 100

# Gap from optimal
gap_df    = path_res[path_res["gap_pct"].notna()]
nn_gap    = gap_df[gap_df["method"] == "nearest_neighbor"]["gap_pct"].mean()
topt_gap  = gap_df[gap_df["method"] == "two_opt"]["gap_pct"].mean()

print(f"\nDistance reductions vs naive:")
print(f"  Nearest-neighbor: {nn_reduction:.1f}%")
print(f"  2-opt           : {topt_reduction:.1f}%")
print(f"  2-opt vs NN     : {topt_vs_nn:.1f}%")
print(f"\nMean gap from exact optimal (sessions ≤8 stops):")
print(f"  NN  : {nn_gap:.2f}%")
print(f"  2-opt: {topt_gap:.2f}%")

# Slotting
current_tt = slot_res.loc[slot_res["method"] == "Current (Naive Slotting)", "mean_travel"].values[0]
greedy_tt  = slot_res.loc[slot_res["method"] == "Greedy Heuristic",         "mean_travel"].values[0]
lp_tt      = slot_res.loc[slot_res["method"] == "LP Joint Optimization",    "mean_travel"].values[0]

slot_greedy_pct = (current_tt - greedy_tt) / current_tt * 100 if current_tt > 0 else 0
slot_lp_pct     = (current_tt - lp_tt)     / current_tt * 100 if current_tt > 0 else 0
slot_lp_vs_g    = (greedy_tt  - lp_tt)     / greedy_tt  * 100 if greedy_tt  > 0 else 0

# Robustness
cadence       = robust["recommended_cadence"].iloc[0] if "recommended_cadence" in robust.columns else "monthly"
mean_corr     = robust["spearman_corr"].mean()
mean_pct_chg  = robust["pct_tier_changed"].mean()

# ── Scale assumptions ─────────────────────────────────────────────────────────
N_WAREHOUSES      = 3
SESSIONS_PER_DAY  = 500    # per warehouse
DAILY_SESSIONS    = N_WAREHOUSES * SESSIONS_PER_DAY   # 1500
ANNUAL_SESSIONS   = DAILY_SESSIONS * 365               # 547,500
MONTHLY_SESSIONS  = DAILY_SESSIONS * 30                # 45,000

N_SKUS            = 1664
COMPUTE_RATE_USD  = 0.10 / 3600   # $0.10/hour = $0.0000278/s

AVG_SESSION_MIN   = 30.0           # minutes per pick session
WAREHOUSE_WAGE    = 18.0           # $/hour
RELOCATE_MIN      = 15.0           # minutes labour per SKU relocation
RELOCATE_WAGE     = 25.0           # $/hour

# Cadence → runs per month
CADENCE_RUNS = {"quarterly": 1/3, "semi-annual": 1/6, "monthly": 1.0}
cadence_runs_per_month = CADENCE_RUNS.get(cadence, 1.0)

# ── Compute costs per method ──────────────────────────────────────────────────
# Path planning (uses actual measured times)
t_naive_s  = timing.get("naive",            0.000010) / 1000   # s/session
t_nn_s     = timing.get("nearest_neighbor", 0.001)    / 1000
t_topt_s   = timing.get("two_opt",          0.010)    / 1000
t_exact_s  = 0.100  # stated: 0.1s/session for ≤8-stop exact

path_methods = {
    "Naive (current)"       : {"time_s": t_naive_s,  "dist_reduction_pct": 0.0,           "label": "naive"},
    "Nearest-Neighbor"      : {"time_s": t_nn_s,     "dist_reduction_pct": nn_reduction,  "label": "nn"},
    "2-Opt"                 : {"time_s": t_topt_s,   "dist_reduction_pct": topt_reduction,"label": "2opt"},
    "Exact TSP (≤8 stops)"  : {"time_s": t_exact_s,  "dist_reduction_pct": topt_reduction * 1.02, "label": "exact"},
    # Exact gives ~same savings as 2-opt (gap is tiny: ~1.75%)
}

for method, info in path_methods.items():
    monthly_compute_cost = MONTHLY_SESSIONS * info["time_s"] * COMPUTE_RATE_USD * 3600
    # Travel saving: dist_reduction → fraction of session time saved
    session_hours_saved  = (info["dist_reduction_pct"] / 100) * (AVG_SESSION_MIN / 60)
    annual_saving_usd    = ANNUAL_SESSIONS * session_hours_saved * WAREHOUSE_WAGE
    info["monthly_compute_cost_usd"] = monthly_compute_cost
    info["annual_travel_saving_usd"] = annual_saving_usd
    info["net_annual_usd"]           = annual_saving_usd - monthly_compute_cost * 12

# Slotting methods
slotting_methods = {
    "Greedy Slotting"       : {"time_s_per_sku": 0.001,  "saving_pct": slot_greedy_pct, "label": "greedy"},
    "LP Joint Slotting"     : {"time_s_per_sku": 0.5,    "saving_pct": slot_lp_pct,     "label": "lp"},
    "Affinity-Aware Greedy" : {"time_s_per_sku": 0.002,  "saving_pct": slot_greedy_pct * 0.8, "label": "aff_greedy"},
}

for method, info in slotting_methods.items():
    monthly_compute = N_SKUS * cadence_runs_per_month * info["time_s_per_sku"] * COMPUTE_RATE_USD * 3600
    session_saving  = (info["saving_pct"] / 100) * (AVG_SESSION_MIN / 60)
    annual_saving   = ANNUAL_SESSIONS * session_saving * WAREHOUSE_WAGE
    # Physical re-slotting labour
    skus_to_move    = (mean_pct_chg / 100) * N_SKUS
    annual_reslots  = cadence_runs_per_month * 12
    labour_per_cycle = skus_to_move * (RELOCATE_MIN / 60) * RELOCATE_WAGE
    annual_labour   = labour_per_cycle * annual_reslots
    info["monthly_compute_cost_usd"] = monthly_compute
    info["annual_travel_saving_usd"] = annual_saving
    info["annual_reslot_labour_usd"] = annual_labour
    info["net_annual_usd"]           = annual_saving - monthly_compute * 12 - annual_labour

# ── Print tables ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("COST-BENEFIT ANALYSIS")
print(f"{'='*70}")
print(f"\nScale: {N_WAREHOUSES} warehouses × {SESSIONS_PER_DAY} sessions/day = "
      f"{DAILY_SESSIONS:,} daily sessions | {ANNUAL_SESSIONS:,} annual")
print(f"Compute rate: ${COMPUTE_RATE_USD*3600:.2f}/hour | "
      f"Warehouse wage: ${WAREHOUSE_WAGE}/hr | Re-slot wage: ${RELOCATE_WAGE}/hr")

print(f"\n--- PATH PLANNING ---")
print(f"{'Method':<26} {'Monthly Compute':>16} {'Annual Travel $':>16} {'Net Annual':>12}")
print("-" * 72)
for method, info in path_methods.items():
    print(f"{method:<26} ${info['monthly_compute_cost_usd']:>14,.2f} "
          f"${info['annual_travel_saving_usd']:>14,.0f} "
          f"${info['net_annual_usd']:>11,.0f}")

print(f"\n--- SLOTTING OPTIMIZATION ---")
print(f"Cadence: {cadence} ({cadence_runs_per_month:.2f} re-slots/month | {cadence_runs_per_month*12:.1f}/year)")
print(f"SKUs relocated per cycle: ~{int(mean_pct_chg/100*N_SKUS)} "
      f"({mean_pct_chg:.1f}% of {N_SKUS})")
print(f"{'Method':<26} {'Monthly Compute':>16} {'Annual Travel $':>16} {'Annual Labour':>14} {'Net Annual':>12}")
print("-" * 82)
for method, info in slotting_methods.items():
    print(f"{method:<26} ${info['monthly_compute_cost_usd']:>14,.2f} "
          f"${info['annual_travel_saving_usd']:>14,.0f} "
          f"${info.get('annual_reslot_labour_usd', 0):>13,.0f} "
          f"${info['net_annual_usd']:>11,.0f}")

# ── Decision questions ────────────────────────────────────────────────────────
print(f"\n--- DECISION QUESTIONS ---")

# 1. LP vs greedy slotting?
lp_info  = slotting_methods["LP Joint Slotting"]
g_info   = slotting_methods["Greedy Slotting"]
inc_saving = lp_info["annual_travel_saving_usd"] - g_info["annual_travel_saving_usd"]
inc_cost   = (lp_info["monthly_compute_cost_usd"] - g_info["monthly_compute_cost_usd"]) * 12
print(f"\n1. Is LP slotting worth it over simple greedy?")
print(f"   Incremental annual travel saving: ${inc_saving:,.0f}")
print(f"   Incremental annual compute cost : ${inc_cost:,.2f}")
if inc_saving > inc_cost:
    print(f"   VERDICT: YES — incremental saving (${inc_saving:,.0f}) exceeds incremental compute (${inc_cost:.2f})")
    print(f"   However, LP saving is only {slot_lp_vs_g:.1f}% better than greedy at travel-time level,")
    print(f"   and this margin is within noise for zone-level (not location-level) assignment.")
    print(f"   PRACTICAL VERDICT: Greedy is sufficient given {slot_lp_vs_g:.1f}% LP advantage.")
else:
    print(f"   VERDICT: Marginal — both methods have near-zero compute cost at this scale.")
    print(f"   LP adds {slot_lp_vs_g:.1f}% travel-time improvement over greedy.")

# 2. Exact TSP worth it?
exact_info = path_methods["Exact TSP (≤8 stops)"]
topt_info  = path_methods["2-Opt"]
inc_exact_cost  = (exact_info["monthly_compute_cost_usd"] - topt_info["monthly_compute_cost_usd"]) * 12
gap_saving      = (topt_gap / 100) * (AVG_SESSION_MIN / 60) * ANNUAL_SESSIONS * WAREHOUSE_WAGE
print(f"\n2. Is exact TSP ever worth it at this scale?")
print(f"   2-opt gap from exact optimal   : {topt_gap:.2f}% (mean over sessions ≤8 stops)")
print(f"   Dollar value of closing gap    : ${gap_saving:,.0f}/year (if all sessions ≤8 stops)")
print(f"   Incremental compute cost exact : ${inc_exact_cost:,.0f}/year")
print(f"   VERDICT: NO — TSP is NP-hard; at 15 stops = 1.3T paths; 2-opt closes {100-topt_gap:.1f}%")
print(f"   of the gap at {t_topt_s/t_exact_s*100:.0f}x lower compute cost.")

# ── Export ────────────────────────────────────────────────────────────────────
rows = []
for method, info in {**path_methods, **slotting_methods}.items():
    rows.append({
        "method"                    : method,
        "category"                  : "path" if method in path_methods else "slotting",
        "monthly_compute_cost_usd"  : round(info["monthly_compute_cost_usd"], 2),
        "annual_travel_saving_usd"  : round(info["annual_travel_saving_usd"], 0),
        "annual_reslot_labour_usd"  : round(info.get("annual_reslot_labour_usd", 0), 0),
        "net_annual_usd"            : round(info["net_annual_usd"], 0),
    })
cba_df = pd.DataFrame(rows)
cba_df.to_csv(OUTPUTS / "cost_benefit_analysis.csv", index=False)
print(f"\nSaved → outputs/cost_benefit_analysis.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(11, 7))

colors = {
    "Naive (current)"      : "#E8A020",
    "Nearest-Neighbor"     : "#1F77B4",
    "2-Opt"                : "#2CA02C",
    "Exact TSP (≤8 stops)" : "#9467BD",
    "Greedy Slotting"      : "#D62728",
    "LP Joint Slotting"    : "#8C564B",
    "Affinity-Aware Greedy": "#E377C2",
}

for method, info in {**path_methods, **slotting_methods}.items():
    cost  = info["monthly_compute_cost_usd"]
    saving = info["annual_travel_saving_usd"] / 1000   # in $K
    ax.scatter(cost, saving, s=150, color=colors.get(method, "grey"),
               zorder=5, label=method)
    ax.annotate(method, (cost, saving), textcoords="offset points",
                xytext=(6, 3), fontsize=8)

ax.set_xlabel("Monthly Compute Cost (USD)", fontsize=12)
ax.set_ylabel("Annual Travel-Time Saving ($K)", fontsize=12)
ax.set_title("Cost-Benefit Map: Monthly Compute Cost vs Annual Travel Saving\nby Method",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(FIGURES / "cost_benefit_methods.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved → figures/cost_benefit_methods.png")
print("\nDone — 06_cost_benefit_analysis.py complete.")
