"""
05_robustness_testing.py — K-Means Robustness and Stability Testing
DHL Data Scientist Portfolio — Project 01

Three robustness tests:
  1. Seed stability    — Run K-Means 20× with different random seeds, measure
                         cluster assignment consistency via ARI pairwise
  2. Feature ablation  — Remove one feature at a time, measure change in
                         cluster assignments vs full-feature baseline
  3. Outlier sensitivity — Remove top 1% revenue outliers, rerun, compare

Outputs:
  outputs/robustness_results.csv
  figures/robustness_seed_stability.png
  figures/robustness_feature_ablation.png
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = BASE_DIR / "figures"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "mean_daily_demand_z", "std_demand_z", "cv_demand_z",
    "total_revenue_z", "revenue_rank_pct_z",
    "demand_frequency_z", "avg_order_size_z",
    "demand_trend_z", "seasonality_strength_z",
]

FEATURE_DISPLAY = [
    "Mean Demand", "Std Demand", "CV", "Revenue", "Rev Rank",
    "Demand Freq", "Avg Order", "Trend", "Seasonality",
]

OPTIMAL_K = 4
N_SEEDS   = 20
N_INIT    = 20


# ---------------------------------------------------------------------------
# 1. Seed stability test
# ---------------------------------------------------------------------------

def test_seed_stability(X: np.ndarray) -> dict:
    log.info(f"Seed stability test: {N_SEEDS} runs with different random seeds ...")
    seeds = list(range(N_SEEDS))
    all_labels = []

    for seed in seeds:
        km = KMeans(n_clusters=OPTIMAL_K, random_state=seed, n_init=N_INIT, max_iter=500)
        labels = km.fit_predict(X)
        all_labels.append(labels)
        if seed % 5 == 0:
            log.info(f"  Completed seed {seed}")

    # Pairwise ARI between all run pairs
    ari_pairs = []
    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_pairs.append(ari)

    mean_ari   = np.mean(ari_pairs)
    std_ari    = np.std(ari_pairs)
    min_ari    = np.min(ari_pairs)
    pct_stable = np.mean(np.array(ari_pairs) > 0.95)  # % of pairs with near-perfect agreement

    log.info(f"  Pairwise ARI: mean={mean_ari:.4f}  std={std_ari:.4f}  min={min_ari:.4f}")
    log.info(f"  % of run pairs with ARI > 0.95: {pct_stable*100:.1f}%")

    # Per-SKU consistency: % of runs where SKU is assigned to the same majority cluster
    # We use run 0 as reference and measure ARI of each other run against it
    ari_vs_ref = [adjusted_rand_score(all_labels[0], all_labels[i]) for i in range(1, N_SEEDS)]
    mean_vs_ref = np.mean(ari_vs_ref)

    return {
        "mean_pairwise_ari":   mean_ari,
        "std_pairwise_ari":    std_ari,
        "min_pairwise_ari":    min_ari,
        "pct_pairs_ari_gt_95": pct_stable,
        "mean_ari_vs_seed0":   mean_vs_ref,
        "all_ari_pairs":       ari_pairs,
        "all_labels":          all_labels,
    }


def plot_seed_stability(stability: dict):
    ari_pairs = stability["all_ari_pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram of pairwise ARIs
    ax = axes[0]
    ax.hist(ari_pairs, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(stability["mean_pairwise_ari"], color="red", linestyle="--", linewidth=2,
               label=f"Mean ARI = {stability['mean_pairwise_ari']:.4f}")
    ax.axvline(0.95, color="orange", linestyle=":", linewidth=2,
               label="ARI = 0.95 (near-perfect)")
    ax.set_xlabel("Pairwise ARI")
    ax.set_ylabel("Count of Run Pairs")
    ax.set_title(f"Seed Stability: Distribution of Pairwise ARI\n({N_SEEDS} seeds, {len(ari_pairs)} pairs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ARI of each run vs run 0
    ax = axes[1]
    ari_vs_ref = [adjusted_rand_score(stability["all_labels"][0], stability["all_labels"][i])
                  for i in range(1, N_SEEDS)]
    ax.bar(range(1, N_SEEDS), ari_vs_ref, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axhline(np.mean(ari_vs_ref), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(ari_vs_ref):.4f}")
    ax.set_xlabel("Run Index (vs Seed 0)")
    ax.set_ylabel("ARI")
    ax.set_title("ARI vs Reference Run (Seed 0)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURE_DIR / "robustness_seed_stability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Feature ablation test
# ---------------------------------------------------------------------------

def test_feature_ablation(X: np.ndarray, baseline_labels: np.ndarray) -> pd.DataFrame:
    log.info("Feature ablation test: removing one feature at a time ...")
    results = []

    for i, (feat_col, feat_name) in enumerate(zip(FEATURE_COLS, FEATURE_DISPLAY)):
        # Remove feature i
        X_ablated = np.delete(X, i, axis=1)
        km = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=N_INIT, max_iter=500)
        labels = km.fit_predict(X_ablated)
        ari  = adjusted_rand_score(baseline_labels, labels)
        sil  = silhouette_score(X_ablated, labels, sample_size=min(len(X), 1000), random_state=42)
        results.append({
            "removed_feature": feat_name,
            "ari_vs_baseline": ari,
            "silhouette":      sil,
            "ari_drop":        1.0 - ari,
        })
        log.info(f"  Remove '{feat_name}': ARI vs baseline = {ari:.4f}  sil = {sil:.4f}")

    return pd.DataFrame(results)


def plot_feature_ablation(ablation: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ARI vs baseline when feature removed
    ax = axes[0]
    colors = ["#F44336" if x < 0.8 else "#FF9800" if x < 0.95 else "#4CAF50"
              for x in ablation["ari_vs_baseline"]]
    bars = ax.barh(ablation["removed_feature"], ablation["ari_vs_baseline"],
                   color=colors, edgecolor="white", alpha=0.9)
    ax.axvline(0.95, color="orange", linestyle="--", linewidth=2, label="ARI = 0.95")
    ax.axvline(0.80, color="red", linestyle="--", linewidth=2, label="ARI = 0.80")
    ax.set_xlabel("ARI vs Full-Feature Baseline")
    ax.set_title("Feature Ablation: Stability of Cluster\nAssignments when Each Feature Removed")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # Silhouette with feature removed
    ax = axes[1]
    ax.barh(ablation["removed_feature"], ablation["silhouette"],
            color="#2196F3", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Silhouette Score (without feature)")
    ax.set_title("Silhouette Score After Feature Removal\n(lower = feature was important)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out = FIGURE_DIR / "robustness_feature_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Outlier sensitivity test
# ---------------------------------------------------------------------------

def test_outlier_sensitivity(feat: pd.DataFrame, X: np.ndarray,
                              baseline_labels: np.ndarray) -> dict:
    log.info("Outlier sensitivity test: removing top 1% revenue SKUs ...")
    rev_threshold = feat["total_revenue"].quantile(0.99)
    outlier_mask  = feat["total_revenue"] <= rev_threshold
    n_outliers    = (~outlier_mask).sum()

    X_clean    = X[outlier_mask.values]
    feat_clean = feat[outlier_mask].copy()
    labels_clean_base = baseline_labels[outlier_mask.values]

    km = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=N_INIT, max_iter=500)
    labels_clean = km.fit_predict(X_clean)
    ari  = adjusted_rand_score(labels_clean_base, labels_clean)
    sil  = silhouette_score(X_clean, labels_clean, sample_size=min(len(X_clean), 1000),
                             random_state=42)

    log.info(f"  Outliers removed: {n_outliers} (revenue > £{rev_threshold:,.0f})")
    log.info(f"  ARI vs baseline (non-outlier SKUs): {ari:.4f}")
    log.info(f"  Silhouette after outlier removal: {sil:.4f}")

    return {
        "n_outliers_removed": n_outliers,
        "outlier_threshold":  rev_threshold,
        "ari_after_removal":  ari,
        "sil_after_removal":  sil,
        "n_remaining":        len(X_clean),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    feat_path = OUTPUT_DIR / "kmeans_clusters.csv"
    log.info(f"Loading cluster assignments from {feat_path} ...")
    feat = pd.read_csv(feat_path)
    X    = feat[FEATURE_COLS].values
    baseline_labels = feat["cluster"].values
    log.info(f"  {len(feat):,} SKUs loaded")

    # 1. Seed stability
    log.info("=" * 55)
    stability = test_seed_stability(X)
    plot_seed_stability(stability)

    # 2. Feature ablation
    log.info("=" * 55)
    ablation = test_feature_ablation(X, baseline_labels)
    plot_feature_ablation(ablation)

    # 3. Outlier sensitivity
    log.info("=" * 55)
    outlier = test_outlier_sensitivity(feat, X, baseline_labels)

    # --- Export ---
    records = []
    records.append({"test": "seed_stability", "metric": "mean_pairwise_ari",
                     "value": stability["mean_pairwise_ari"], "notes": f"std={stability['std_pairwise_ari']:.4f}"})
    records.append({"test": "seed_stability", "metric": "min_pairwise_ari",
                     "value": stability["min_pairwise_ari"], "notes": ""})
    records.append({"test": "seed_stability", "metric": "pct_pairs_ari_gt_95",
                     "value": stability["pct_pairs_ari_gt_95"], "notes": ""})
    for _, row in ablation.iterrows():
        records.append({"test": "feature_ablation", "metric": f"ari_remove_{row['removed_feature']}",
                         "value": row["ari_vs_baseline"], "notes": f"sil={row['silhouette']:.4f}"})
    records.append({"test": "outlier_sensitivity", "metric": "ari_after_removal",
                     "value": outlier["ari_after_removal"],
                     "notes": f"n_outliers={outlier['n_outliers_removed']}"})
    records.append({"test": "outlier_sensitivity", "metric": "sil_after_removal",
                     "value": outlier["sil_after_removal"], "notes": ""})

    out_path = OUTPUT_DIR / "robustness_results.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    log.info(f"  Saved: {out_path}")

    # --- Print summary ---
    print("\n" + "=" * 65)
    print("ROBUSTNESS TESTING RESULTS")
    print("=" * 65)

    print("\n1. SEED STABILITY (20 random seeds)")
    print(f"   Mean pairwise ARI: {stability['mean_pairwise_ari']:.4f}")
    print(f"   Std pairwise ARI:  {stability['std_pairwise_ari']:.4f}")
    print(f"   Min pairwise ARI:  {stability['min_pairwise_ari']:.4f}")
    print(f"   % pairs with ARI > 0.95: {stability['pct_pairs_ari_gt_95']*100:.1f}%")
    print(f"   Mean ARI vs seed-0 reference: {stability['mean_ari_vs_seed0']:.4f}")

    print("\n2. FEATURE ABLATION (remove one feature at a time)")
    print(ablation[["removed_feature", "ari_vs_baseline", "silhouette"]].to_string(index=False))
    most_critical = ablation.loc[ablation["ari_vs_baseline"].idxmin(), "removed_feature"]
    least_critical = ablation.loc[ablation["ari_vs_baseline"].idxmax(), "removed_feature"]
    print(f"\n   Most critical feature (largest ARI drop when removed): {most_critical}")
    print(f"   Least critical feature (smallest ARI drop when removed): {least_critical}")

    print("\n3. OUTLIER SENSITIVITY (top 1% revenue removed)")
    print(f"   SKUs removed:   {outlier['n_outliers_removed']} (revenue > £{outlier['outlier_threshold']:,.0f})")
    print(f"   SKUs remaining: {outlier['n_remaining']:,}")
    print(f"   ARI vs baseline (on non-outlier set): {outlier['ari_after_removal']:.4f}")
    print(f"   Silhouette after removal:             {outlier['sil_after_removal']:.4f}")

    # Conclusion
    seed_stable   = stability["mean_pairwise_ari"] >= 0.95
    feat_stable   = ablation["ari_vs_baseline"].min() >= 0.70
    outlier_stable = outlier["ari_after_removal"] >= 0.80

    print("\n" + "=" * 65)
    print("PRODUCTION READINESS CONCLUSION")
    print("=" * 65)
    print(f"""
  Seed stability:    {'PASS ✓' if seed_stable else 'CAUTION ⚠'}  (mean ARI={stability['mean_pairwise_ari']:.4f}, threshold=0.95)
  Feature stability: {'PASS ✓' if feat_stable else 'CAUTION ⚠'}  (min ARI={ablation['ari_vs_baseline'].min():.4f}, threshold=0.70)
  Outlier stability: {'PASS ✓' if outlier_stable else 'CAUTION ⚠'}  (ARI={outlier['ari_after_removal']:.4f}, threshold=0.80)

  Overall: {"STABLE — suitable for production use" if all([seed_stable, feat_stable, outlier_stable])
            else "CONDITIONALLY STABLE — review caution flags before production deployment"}

  Details:
  - Seed stability: {'High' if seed_stable else 'Moderate'} — the K-Means solution converges
    consistently across different random initialisations, confirming the
    clusters are well-separated rather than an artefact of initialisation.

  - Feature stability: {'Good' if feat_stable else 'Moderate'} — removing any single feature still
    produces cluster assignments broadly consistent with the full model.
    The most critical feature is '{most_critical}', suggesting it provides
    the most unique information for separating clusters.

  - Outlier stability: {'Good' if outlier_stable else 'Moderate'} — cluster structure persists
    after removing the top 1% most extreme SKUs by revenue. The clustering
    is not driven by a handful of outliers.

  Limitations to monitor:
  - New SKUs with < 90 days history may not have stable seasonality_strength
    estimates and should be assigned by nearest-centroid rule with caution.
  - If the SKU mix changes significantly (new categories, major listings),
    retrain is recommended rather than forcing new SKUs into old clusters.
  - Retraining frequency: quarterly, or whenever > 15% of the catalogue
    changes in ABC class.
""")

    return stability, ablation, outlier


if __name__ == "__main__":
    stability, ablation, outlier = main()
