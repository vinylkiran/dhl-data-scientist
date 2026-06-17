"""
06_cost_benefit_analysis.py — Cost-Benefit Analysis
DHL Data Scientist Portfolio — Project 01

Compares the total cost of ownership of rule-based ABC/XYZ segmentation
against K-Means clustering, and estimates the dollar value of improved
stockout-risk separation.

Outputs:
  outputs/cost_benefit_analysis.csv
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_inputs():
    clusters   = pd.read_csv(OUTPUT_DIR / "kmeans_clusters.csv")
    robustness = pd.read_csv(OUTPUT_DIR / "robustness_results.csv")
    sep_qual   = pd.read_csv(OUTPUT_DIR / "separation_quality.csv")
    method_cmp = pd.read_csv(OUTPUT_DIR / "method_comparison.csv")
    return clusters, robustness, sep_qual, method_cmp


def compute_rule_based_costs():
    """
    Rule-based ABC/XYZ classification cost model.
    Compute time: sort + cumsum + threshold ≈ 2 seconds.
    """
    compute_time_s   = 2.0
    cloud_rate_usd_h = 0.10          # $/hr cloud compute
    runs_per_year    = 12

    cost_per_run_usd = cloud_rate_usd_h * (compute_time_s / 3600)
    annual_cloud_usd = cost_per_run_usd * runs_per_year

    # Maintenance effort scores (1=low … 5=high)
    maintenance = {
        "revalidation":  1,
        "explainability": 5,
        "justification":  5,
        "integration":    1,
    }
    maintenance_avg = np.mean(list(maintenance.values()))

    return {
        "method":               "Rule-Based (ABC/XYZ)",
        "compute_time_s":       compute_time_s,
        "cost_per_run_usd":     cost_per_run_usd,
        "runs_per_year":        runs_per_year,
        "annual_cloud_usd":     annual_cloud_usd,
        "maintenance_scores":   maintenance,
        "maintenance_avg":      maintenance_avg,
    }


def compute_kmeans_costs():
    """
    K-Means total pipeline cost model.
    Feature engineering ≈30 s, k-sweep (11 values × 20 init) ≈8 s,
    final clustering ≈2 s → total ≈40 s.
    """
    compute_time_s   = 40.0
    cloud_rate_usd_h = 0.10
    runs_per_year    = 12

    cost_per_run_usd = cloud_rate_usd_h * (compute_time_s / 3600)
    annual_cloud_usd = cost_per_run_usd * runs_per_year

    maintenance = {
        "revalidation":   3,
        "explainability": 2,
        "justification":  2,
        "integration":    4,
    }
    maintenance_avg = np.mean(list(maintenance.values()))

    return {
        "method":               "K-Means Clustering",
        "compute_time_s":       compute_time_s,
        "cost_per_run_usd":     cost_per_run_usd,
        "runs_per_year":        runs_per_year,
        "annual_cloud_usd":     annual_cloud_usd,
        "maintenance_scores":   maintenance,
        "maintenance_avg":      maintenance_avg,
    }


def estimate_incremental_value(clusters: pd.DataFrame, sep_qual: pd.DataFrame):
    """
    Dollar value of accuracy improvement from K-Means over rule-based.

    Logic:
    1. Total dataset revenue and average per-SKU metrics.
    2. Misclassified SKUs = A-class SKUs placed in Low-Velocity clusters by K-Means
       (these are SKUs that ABC over-rates; conversely C-class in High-Velocity clusters
       are SKUs ABC under-rates).
    3. For each misclassified SKU, improved classification is assumed to catch 10% of
       their stockout events, preventing lost revenue equal to avg daily revenue × stockouts.
    4. Returns itemised breakdown.
    """
    total_revenue       = clusters["total_revenue"].sum()
    n_skus              = len(clusters)
    avg_revenue_per_sku = total_revenue / n_skus
    mean_stockout_rate  = clusters["stockout_rate"].mean()

    # Misclassified: A-class in Low-Velocity clusters
    low_vel_mask   = clusters["cluster_label"].str.startswith("Low-Velocity")
    a_in_low       = clusters[low_vel_mask & (clusters["abc_class"] == "A")]
    misclass_count = len(a_in_low)

    # Average revenue of those SKUs
    avg_rev_misclass = a_in_low["total_revenue"].mean() if misclass_count > 0 else avg_revenue_per_sku
    avg_so_misclass  = a_in_low["stockout_rate"].mean()  if misclass_count > 0 else mean_stockout_rate

    catch_rate = 0.10   # assumption: 10% of stockout events caught by reclassification
    stockout_value_captured = (
        misclass_count * avg_rev_misclass * avg_so_misclass * catch_rate
    )

    # Also: C-class in High-Velocity clusters (under-rated SKUs deserving more attention)
    high_vel_mask = clusters["cluster_label"].str.startswith("High-Velocity")
    c_in_high     = clusters[high_vel_mask & (clusters["abc_class"] == "C")]
    underrated_count = len(c_in_high)

    # Separation quality comparison
    km_row  = sep_qual[sep_qual["method"] == "K-Means (cluster)"].iloc[0]
    abc_row = sep_qual[sep_qual["method"] == "ABC Class"].iloc[0]
    km_within_cv  = km_row["within_group_stockout_cv"]
    abc_within_cv = abc_row["within_group_stockout_cv"]
    separation_improvement = abc_within_cv - km_within_cv   # positive = K-Means better

    return {
        "total_revenue":             total_revenue,
        "n_skus":                    n_skus,
        "avg_revenue_per_sku":       avg_revenue_per_sku,
        "mean_stockout_rate":        mean_stockout_rate,
        "misclassified_a_count":     misclass_count,
        "avg_rev_misclassified":     avg_rev_misclass,
        "avg_stockout_misclassified":avg_so_misclass,
        "catch_rate":                catch_rate,
        "stockout_value_captured":   stockout_value_captured,
        "underrated_c_count":        underrated_count,
        "km_within_cv":              km_within_cv,
        "abc_within_cv":             abc_within_cv,
        "separation_improvement":    separation_improvement,
    }


def compute_net_value(rb_costs, km_costs, incremental):
    """
    Net annual value = stockout_value_captured
                     - annual_compute_cost_delta
                     - maintenance_cost_delta
    """
    annual_compute_delta = km_costs["annual_cloud_usd"] - rb_costs["annual_cloud_usd"]

    # Maintenance cost:  complexity_score × $50/hr × 4 hrs/quarter × 4 quarters
    hourly_rate = 50.0
    hrs_per_qtr = 4.0
    quarters    = 4.0
    rb_maint_cost = rb_costs["maintenance_avg"] * hourly_rate * hrs_per_qtr * quarters
    km_maint_cost = km_costs["maintenance_avg"] * hourly_rate * hrs_per_qtr * quarters
    maintenance_cost_delta = km_maint_cost - rb_maint_cost

    net_annual_value = (
        incremental["stockout_value_captured"]
        - annual_compute_delta
        - maintenance_cost_delta
    )

    return {
        "annual_compute_delta_usd":    annual_compute_delta,
        "rb_annual_maintenance_usd":   rb_maint_cost,
        "km_annual_maintenance_usd":   km_maint_cost,
        "maintenance_cost_delta_usd":  maintenance_cost_delta,
        "stockout_value_captured_usd": incremental["stockout_value_captured"],
        "net_annual_value_usd":        net_annual_value,
    }


def build_export_rows(rb_costs, km_costs, incremental, net):
    rows = [
        # ── Compute costs ──────────────────────────────────────────────
        {"category": "Compute",  "item": "Rule-Based — compute time (s)",
         "rule_based": rb_costs["compute_time_s"],  "kmeans": km_costs["compute_time_s"],  "unit": "seconds"},
        {"category": "Compute",  "item": "Cost per run (USD)",
         "rule_based": round(rb_costs["cost_per_run_usd"], 8),
         "kmeans":     round(km_costs["cost_per_run_usd"], 8),  "unit": "USD"},
        {"category": "Compute",  "item": "Annual cloud cost (12 runs/yr, USD)",
         "rule_based": round(rb_costs["annual_cloud_usd"], 6),
         "kmeans":     round(km_costs["annual_cloud_usd"], 6),  "unit": "USD/yr"},
        # ── Maintenance ────────────────────────────────────────────────
        {"category": "Maintenance", "item": "Complexity score (avg, 1–5 scale)",
         "rule_based": round(rb_costs["maintenance_avg"], 2),
         "kmeans":     round(km_costs["maintenance_avg"], 2),   "unit": "score"},
        {"category": "Maintenance", "item": "Annual maintenance cost (USD)",
         "rule_based": round(net["rb_annual_maintenance_usd"], 2),
         "kmeans":     round(net["km_annual_maintenance_usd"], 2), "unit": "USD/yr"},
        # ── Incremental value ──────────────────────────────────────────
        {"category": "Value",    "item": "Total portfolio revenue (USD)",
         "rule_based": round(incremental["total_revenue"], 0),
         "kmeans":     round(incremental["total_revenue"], 0),  "unit": "USD"},
        {"category": "Value",    "item": "A-class SKUs in Low-Velocity clusters",
         "rule_based": "N/A",
         "kmeans":     incremental["misclassified_a_count"],    "unit": "count"},
        {"category": "Value",    "item": "C-class SKUs in High-Velocity clusters",
         "rule_based": "N/A",
         "kmeans":     incremental["underrated_c_count"],       "unit": "count"},
        {"category": "Value",    "item": "Avg revenue — misclassified A-class (USD)",
         "rule_based": "N/A",
         "kmeans":     round(incremental["avg_rev_misclassified"], 0), "unit": "USD"},
        {"category": "Value",    "item": "Avg stockout rate — misclassified A-class",
         "rule_based": "N/A",
         "kmeans":     round(incremental["avg_stockout_misclassified"], 5), "unit": "rate"},
        {"category": "Value",    "item": "Within-group stockout CV (lower = better)",
         "rule_based": round(incremental["abc_within_cv"], 4),
         "kmeans":     round(incremental["km_within_cv"], 4),   "unit": "CV"},
        {"category": "Value",    "item": "Stockout value captured at 10% catch rate (USD)",
         "rule_based": 0.0,
         "kmeans":     round(net["stockout_value_captured_usd"], 2), "unit": "USD/yr"},
        # ── Net ────────────────────────────────────────────────────────
        {"category": "Net",      "item": "Annual compute cost delta (USD)",
         "rule_based": 0.0,
         "kmeans":     round(net["annual_compute_delta_usd"], 6), "unit": "USD/yr"},
        {"category": "Net",      "item": "Annual maintenance cost delta (USD)",
         "rule_based": 0.0,
         "kmeans":     round(net["maintenance_cost_delta_usd"], 2), "unit": "USD/yr"},
        {"category": "Net",      "item": "NET ANNUAL VALUE — K-Means vs Rule-Based (USD)",
         "rule_based": 0.0,
         "kmeans":     round(net["net_annual_value_usd"], 2),   "unit": "USD/yr"},
    ]
    return pd.DataFrame(rows)


def main():
    log.info("Loading prior outputs ...")
    clusters, robustness, sep_qual, method_cmp = load_inputs()

    log.info("Computing rule-based costs ...")
    rb_costs = compute_rule_based_costs()

    log.info("Computing K-Means costs ...")
    km_costs = compute_kmeans_costs()

    log.info("Estimating incremental value ...")
    incremental = estimate_incremental_value(clusters, sep_qual)

    log.info("Computing net annual value ...")
    net = compute_net_value(rb_costs, km_costs, incremental)

    df_out = build_export_rows(rb_costs, km_costs, incremental, net)
    out_path = OUTPUT_DIR / "cost_benefit_analysis.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")

    # ── Print ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COST-BENEFIT ANALYSIS: K-MEANS vs RULE-BASED ABC/XYZ")
    print("=" * 72)

    print("\n--- COMPUTE COSTS ---")
    print(f"{'Item':<45} {'Rule-Based':>14} {'K-Means':>14}")
    print("-" * 75)
    print(f"{'Compute time (seconds)':<45} {rb_costs['compute_time_s']:>14.1f} {km_costs['compute_time_s']:>14.1f}")
    print(f"{'Cost per run (USD)':<45} {rb_costs['cost_per_run_usd']:>14.8f} {km_costs['cost_per_run_usd']:>14.8f}")
    print(f"{'Annual cloud cost (12 runs, USD)':<45} {rb_costs['annual_cloud_usd']:>14.6f} {km_costs['annual_cloud_usd']:>14.6f}")

    print("\n--- MAINTENANCE COSTS (annual, USD) ---")
    print(f"{'Complexity score (avg 1–5)':<45} {rb_costs['maintenance_avg']:>14.2f} {km_costs['maintenance_avg']:>14.2f}")
    print(f"{'Annual maintenance cost (USD)':<45} {net['rb_annual_maintenance_usd']:>14.2f} {net['km_annual_maintenance_usd']:>14.2f}")

    print("\n--- INCREMENTAL VALUE FROM IMPROVED ACCURACY ---")
    print(f"  Total portfolio revenue:          ${incremental['total_revenue']:>16,.0f}")
    print(f"  A-class SKUs in Low-Vel clusters: {incremental['misclassified_a_count']:>6}  (over-rated by ABC)")
    print(f"  C-class SKUs in High-Vel clusters:{incremental['underrated_c_count']:>6}  (under-rated by ABC)")
    print(f"  Avg revenue of misclassified A:   ${incremental['avg_rev_misclassified']:>16,.0f}")
    print(f"  Avg stockout rate (misclassified):{incremental['avg_stockout_misclassified']:>9.4f}")
    print(f"  Catch rate assumption:            {incremental['catch_rate']*100:.0f}%")
    print(f"  Stockout value captured (USD/yr): ${net['stockout_value_captured_usd']:>16,.2f}")
    print(f"  Within-group stockout CV — ABC:   {incremental['abc_within_cv']:.4f}")
    print(f"  Within-group stockout CV — KMeans:{incremental['km_within_cv']:.4f}  "
          f"({'BETTER' if incremental['separation_improvement']>0 else 'WORSE'})")

    print("\n--- NET ANNUAL VALUE ---")
    print(f"  Stockout value captured:          ${net['stockout_value_captured_usd']:>16,.2f}")
    print(f"  Less: annual compute delta:       ${net['annual_compute_delta_usd']:>16.6f}")
    print(f"  Less: annual maintenance delta:   ${net['maintenance_cost_delta_usd']:>16.2f}")
    print(f"  ──────────────────────────────────────────────────────")
    print(f"  NET ANNUAL VALUE (K-Means):       ${net['net_annual_value_usd']:>16,.2f}")
    verdict = "POSITIVE — K-Means justified on financial grounds" if net["net_annual_value_usd"] > 0 else "NEGATIVE — Rule-based preferred on cost grounds"
    print(f"\n  Verdict: {verdict}")

    print("\n" + "=" * 72)
    print(f"OUTPUT SAVED: outputs/cost_benefit_analysis.csv")
    print("=" * 72)

    return net


if __name__ == "__main__":
    net = main()
