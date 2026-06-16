"""
03_kmeans_clustering.py — K-Means Clustering with k=4
DHL Data Scientist Portfolio — Project 01

Runs K-Means with the validated optimal k=4.
Profiles each cluster and visualises in PCA-reduced 2D space.

Outputs:
  outputs/kmeans_clusters.csv  — SKU-level cluster assignments + features
  figures/kmeans_pca_scatter.png
  figures/cluster_profiles_radar.png
  figures/silhouette_per_cluster.png
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

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

RAW_FEATURES = [
    "mean_daily_demand", "std_demand", "cv_demand",
    "total_revenue", "revenue_rank_pct",
    "demand_frequency", "avg_order_size",
    "demand_trend", "seasonality_strength",
]

OPTIMAL_K    = 4
RANDOM_STATE = 42
N_INIT       = 50   # more initialisations for final solution stability

# Business-friendly cluster labels assigned after profiling
CLUSTER_LABELS = {
    # Will be assigned after profiling based on mean_daily_demand + total_revenue
}


def assign_cluster_labels(profiles: pd.DataFrame) -> dict:
    """
    Assign interpretable labels based on the two most business-relevant features:
    mean_daily_demand (velocity) and total_revenue (value).
    Returns a dict mapping cluster int → label string.
    """
    p = profiles[["cluster", "mean_daily_demand", "total_revenue"]].copy()
    med_demand  = p["mean_daily_demand"].median()
    med_revenue = p["total_revenue"].median()

    labels = {}
    for _, row in p.iterrows():
        c = int(row["cluster"])
        high_demand  = row["mean_daily_demand"] >= med_demand
        high_revenue = row["total_revenue"] >= med_revenue
        if high_demand and high_revenue:
            labels[c] = "High-Velocity / High-Value"
        elif high_demand and not high_revenue:
            labels[c] = "High-Velocity / Low-Value"
        elif not high_demand and high_revenue:
            labels[c] = "Low-Velocity / High-Value"
        else:
            labels[c] = "Low-Velocity / Low-Value"
    return labels


# ---------------------------------------------------------------------------
# PCA scatter plot
# ---------------------------------------------------------------------------

def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, cluster_name_map: dict):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(11, 8))

    for c in sorted(np.unique(labels)):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[c % len(colors)], label=f"Cluster {c}: {cluster_name_map.get(c, '')}",
                   alpha=0.6, s=25, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"K-Means Clusters (k={OPTIMAL_K}) in PCA Space  "
                 f"[{(var_explained[:2].sum()*100):.1f}% variance explained]",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = FIGURE_DIR / "kmeans_pca_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out}")
    return var_explained


# ---------------------------------------------------------------------------
# Silhouette per cluster
# ---------------------------------------------------------------------------

def plot_silhouette(X: np.ndarray, labels: np.ndarray, overall_sil: float, cluster_name_map: dict):
    sil_samples = silhouette_samples(X, labels)
    n_clusters  = len(np.unique(labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    cluster_sils = {}

    for c in sorted(np.unique(labels)):
        vals = np.sort(sil_samples[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                          facecolor=colors[c], edgecolor=colors[c], alpha=0.7)
        ax.text(-0.05, (y_lower + y_upper) / 2, f"C{c}", ha="right", fontsize=9)
        cluster_sils[c] = vals.mean()
        y_lower = y_upper + 10

    ax.axvline(x=overall_sil, color="red", linestyle="--", linewidth=2,
               label=f"Overall silhouette = {overall_sil:.4f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=11)
    ax.set_ylabel("SKU (sorted by cluster)", fontsize=11)
    ax.set_title(f"Silhouette Plot per Cluster (k={OPTIMAL_K})", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.3, 1.0])
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURE_DIR / "silhouette_per_cluster.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out}")
    return cluster_sils


# ---------------------------------------------------------------------------
# Cluster profile radar chart
# ---------------------------------------------------------------------------

def plot_radar(profiles: pd.DataFrame, feature_display_names: list):
    n_vars = len(feature_display_names)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()

    # Normalise feature values to [0, 1] range for radar
    feat_raw = profiles[feature_display_names].copy()
    feat_norm = (feat_raw - feat_raw.min()) / (feat_raw.max() - feat_raw.min() + 1e-9)

    for idx, (_, row) in enumerate(profiles.iterrows()):
        c = int(row["cluster"])
        ax = axes_flat[c]
        vals = feat_norm.iloc[idx].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=colors[c], linewidth=2)
        ax.fill(angles, vals, alpha=0.25, color=colors[c])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_display_names, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Cluster {c}\n{row.get('cluster_label','')}\n(n={int(row['size'])})",
                     size=10, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cluster Profiles — Normalised Feature Radar", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURE_DIR / "cluster_profiles_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    feat_path = OUTPUT_DIR / "sku_features.csv"
    log.info(f"Loading features from {feat_path} ...")
    feat = pd.read_csv(feat_path)
    X    = feat[FEATURE_COLS].values
    log.info(f"  {X.shape[0]:,} SKUs × {X.shape[1]} features")

    # --- Run K-Means ---
    log.info(f"Running K-Means k={OPTIMAL_K} (n_init={N_INIT}, max_iter=1000) ...")
    km = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=N_INIT, max_iter=1000)
    km.fit(X)
    labels = km.labels_
    feat["cluster"] = labels

    # --- Silhouette ---
    overall_sil = silhouette_score(X, labels)
    log.info(f"  Overall silhouette score: {overall_sil:.4f}")

    # --- Cluster profiles (on raw features) ---
    log.info("Building cluster profiles ...")
    profiles = feat.groupby("cluster")[RAW_FEATURES].mean().reset_index()
    profiles["size"] = feat.groupby("cluster").size().values

    # Dominant category and ABC class per cluster
    def mode_series(s):
        return s.mode().iloc[0] if len(s) > 0 else "Unknown"

    dom_cat = feat.groupby("cluster")["category"].agg(mode_series).reset_index().rename(
        columns={"category": "dominant_category"})
    dom_abc = feat.groupby("cluster")["abc_class"].agg(mode_series).reset_index().rename(
        columns={"abc_class": "dominant_abc_class"})
    profiles = profiles.merge(dom_cat, on="cluster").merge(dom_abc, on="cluster")

    cluster_name_map = assign_cluster_labels(profiles)
    profiles["cluster_label"] = profiles["cluster"].map(cluster_name_map)
    feat["cluster_label"]     = feat["cluster"].map(cluster_name_map)

    # --- Plots ---
    log.info("Generating PCA scatter plot ...")
    var_explained = plot_pca_scatter(X, labels, cluster_name_map)

    log.info("Generating silhouette plot ...")
    cluster_sils = plot_silhouette(X, labels, overall_sil, cluster_name_map)

    # Map raw feature column names to display names for the radar chart
    profile_display = profiles[["cluster", "cluster_label", "size"] + RAW_FEATURES].copy()
    profile_display.columns = (["cluster", "cluster_label", "size"] +
                                ["Mean Demand", "Std Demand", "CV", "Revenue", "Rev Rank",
                                 "Freq", "Avg Order", "Trend", "Seasonality"])
    display_names = ["Mean Demand", "Std Demand", "CV", "Revenue", "Rev Rank",
                     "Freq", "Avg Order", "Trend", "Seasonality"]
    log.info("Generating radar chart ...")
    plot_radar(profile_display, display_names)

    # --- Export ---
    out_path = OUTPUT_DIR / "kmeans_clusters.csv"
    feat.to_csv(out_path, index=False)
    log.info(f"  Saved cluster assignments: {out_path}")

    # --- Print summary ---
    print("\n" + "=" * 65)
    print(f"K-MEANS CLUSTERING RESULTS (k={OPTIMAL_K})")
    print("=" * 65)
    print(f"\nOverall Silhouette Score: {overall_sil:.4f}  "
          f"[range -1 to +1; >0.5 = well-clustered]")

    print("\nCluster Profiles (raw feature means):")
    display_prof = profiles[["cluster", "cluster_label", "size",
                              "mean_daily_demand", "total_revenue",
                              "cv_demand", "demand_frequency",
                              "dominant_category", "dominant_abc_class"]].copy()
    display_prof["total_revenue"] = display_prof["total_revenue"].apply(lambda x: f"£{x:,.0f}")
    display_prof["mean_daily_demand"] = display_prof["mean_daily_demand"].round(1)
    display_prof["cv_demand"]         = display_prof["cv_demand"].round(3)
    display_prof["demand_frequency"]  = (display_prof["demand_frequency"] * 100).round(1)
    print(display_prof.to_string(index=False))

    print("\nSilhouette Score Per Cluster:")
    for c, sil in sorted(cluster_sils.items()):
        flag = " ← POORLY SEPARATED" if sil < 0.3 else ""
        print(f"  Cluster {c} ({cluster_name_map.get(c,'')}): {sil:.4f}{flag}")

    print(f"\nPCA Variance Explained: PC1={var_explained[0]*100:.1f}%  "
          f"PC2={var_explained[1]*100:.1f}%  "
          f"Total={var_explained[:2].sum()*100:.1f}%")

    print("\nABC Class Distribution per Cluster:")
    abc_dist = feat.groupby(["cluster", "abc_class"]).size().unstack(fill_value=0)
    print(abc_dist.to_string())

    print("\nCategory Distribution per Cluster:")
    cat_dist = feat.groupby(["cluster", "category"]).size().unstack(fill_value=0)
    print(cat_dist.to_string())

    print("\n" + "=" * 65)
    print("FIGURES SAVED:")
    print("  figures/kmeans_pca_scatter.png")
    print("  figures/silhouette_per_cluster.png")
    print("  figures/cluster_profiles_radar.png")
    print("OUTPUT SAVED:")
    print("  outputs/kmeans_clusters.csv")
    print("=" * 65)

    return overall_sil, cluster_sils


if __name__ == "__main__":
    overall_sil, cluster_sils = main()
