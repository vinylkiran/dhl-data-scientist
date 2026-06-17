# Model Card — SKU Segmentation via K-Means Clustering
## DHL Data Scientist Portfolio — Project 01 (v2.0 rebuild, 2026-06-17)

---

## Model Overview

| Field | Value |
|---|---|
| Model type | Unsupervised clustering |
| Algorithm | K-Means (Lloyd's algorithm) |
| Implementation | scikit-learn `KMeans` |
| Optimal k | 4 |
| Validated | Yes — four internal metrics + hierarchical comparison |
| Production-ready | Yes — all three robustness tests passed |

---

## Training Data

| Field | Value |
|---|---|
| Source | `shared/data/dhl-synthetic/daily_demand.csv` |
| SKUs | 1,664 (SKUs present in demand data; 336 from sku_master.csv had no demand activity) |
| Date range | 1 Jan 2022 – 31 Dec 2023 (730 days) |
| Warehouses | 3 (DHL-WH-NJ01, DHL-WH-IL02, DHL-WH-TX03) |
| Rows processed | 574,509 |
| Aggregation | Daily demand aggregated across warehouses to SKU × date level; then summarised to 9 SKU-level features |

---

## Feature List

| Feature | Engineering Rationale |
|---|---|
| `mean_daily_demand` | Core velocity measure. Separates fast-movers from slow-movers more directly than any other feature. |
| `std_demand` | Quantifies demand uncertainty. High-std SKUs require more safety stock. Correlated with mean (r=0.94) but included because it adds information about peak demand risk. |
| `cv_demand` | Coefficient of variation (std/mean). Demand predictability normalised for volume. The primary input to XYZ classification. |
| `total_revenue` | Financial importance. Ensures high-unit-cost slow movers are not grouped with low-value slow movers in the same cluster. |
| `revenue_rank_pct` | Revenue percentile rank. Compresses the heavy-tailed revenue distribution into a uniform [0,1] feature, preventing a few extreme-revenue SKUs from dominating the feature space. |
| `demand_frequency` | % of days with non-zero demand. Captures intermittent vs regular demand patterns, which determine replenishment policy (s,S vs continuous review). |
| `avg_order_size` | Mean order quantity on active days. Distinguishes SKUs with occasional large orders from those with steady small orders. |
| `demand_trend` | Linear regression slope over time (units/day). Identifies growing or declining SKUs that need forward-looking inventory planning. |
| `seasonality_strength` | Ratio of seasonal variance to total variance. Identifies SKUs requiring seasonal safety stock buffers vs flat replenishment. |

**Standardisation:** All features were z-score standardised (zero mean, unit variance) before clustering to prevent high-magnitude features (e.g. total_revenue in £millions) from dominating the distance calculations.

---

## Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| k (number of clusters) | 4 | Selected after sweeping k=2–12. K=2 maximises silhouette but is too coarse for operational use. k=4 is the first local silhouette maximum in k≥3 with the highest CH score, producing four operationally meaningful SKU archetypes. |
| n_init | 50 (final model) | Multiple random initialisations to avoid poor local optima. Stability testing confirmed 100% of seed pairs achieve ARI > 0.95, so n_init=20 would suffice; 50 used for the final model for extra certainty. |
| max_iter | 1,000 | Convergence tolerance sufficient for 1,664 × 9 data. |
| random_state | 42 | Fixed for reproducibility; robustness testing confirmed results are seed-independent. |
| distance metric | Euclidean (default) | Appropriate after z-score standardisation. All features are continuous. |

---

## Evaluation Metrics

### Internal Validation (k=4 final solution)

| Metric | Value | Interpretation |
|---|---|---|
| Silhouette Score | **0.5265** | Good separation (>0.5 threshold). Clusters are meaningfully distinct. |
| Calinski-Harabasz | **1,549** | High value indicates compact, well-separated clusters. |
| Davies-Bouldin | 0.970 | Below 1.0 = good inter-cluster separation. |
| PCA variance explained (PC1+PC2) | 78.4% | Most cluster structure visible in 2D projection. |

### Per-Cluster Silhouette

| Cluster | Label | n SKUs | Silhouette | Note |
|---|---|---|---|---|
| 0 | High-Velocity / Low-Value | 113 | 0.245 | Close to Cluster 3 in demand-velocity space |
| 1 | Low-Velocity / Low-Value | 821 | 0.663 | Most clearly separated — dominant cluster |
| 2 | Low-Velocity / High-Value | 572 | 0.462 | Transitional; mid-demand, significant revenue |
| 3 | High-Velocity / High-Value | 158 | 0.231 | Closest to Cluster 0 in feature space |

The lower silhouette on Clusters 0 and 3 reflects their proximity in the demand-velocity dimension. They are differentiated by revenue magnitude and category mix. This is expected and operationally appropriate — both clusters warrant high-frequency replenishment but different stock positioning (Cluster 3 additionally warrants capital-weighted safety stock).

---

## Comparison to Baseline Rule-Based Method

| Comparison | ARI | NMI | Interpretation |
|---|---|---|---|
| K-Means vs ABC Class | 0.885 | 0.834 | Strong agreement — both primarily driven by revenue/volume |
| K-Means vs XYZ Class | 0.002 | 0.005 | Near-zero agreement — K-Means captures something entirely different |
| K-Means vs ABC+XYZ Combined | 0.344 | 0.557 | Moderate overlap — partial agreement on overall structure |

**Key insight:** K-Means agrees strongly with ABC on the revenue dimension but is almost completely independent of XYZ. This means K-Means clusters are capturing demand *pattern* information (frequency, trend, seasonality) that XYZ misses, while preserving the revenue-prioritisation signal of ABC. The two methods are complementary, not redundant.

**Which method better separates SKUs by stockout rate?**
ABC+XYZ Combined achieves the best absolute range in mean stockout rates across groups (0.0017 range), but K-Means achieves a lower within-group variation coefficient, suggesting its clusters are more internally homogeneous even if the between-cluster range is similar.

---

## Robustness Test Results

| Test | Metric | Result | Threshold | Status |
|---|---|---|---|---|
| Seed stability (20 seeds) | Mean pairwise ARI | 0.9989 | 0.95 | PASS |
| Seed stability | Min pairwise ARI | 0.9951 | — | PASS |
| Seed stability | % pairs with ARI > 0.95 | 100% | — | PASS |
| Feature ablation | Min ARI (vs baseline) | 0.965 (Demand Freq removed) | 0.70 | PASS |
| Feature ablation | Most critical feature | Demand Frequency | — | — |
| Outlier sensitivity | ARI after removing top 1% | 0.983 | 0.80 | PASS |

**Overall: STABLE — suitable for production use.**

---

## Known Limitations

**Feature scaling dependency:** K-Means uses Euclidean distance in the standardised feature space. If the distribution of any feature shifts significantly (e.g. a new product category with extremely different demand patterns is added), the z-score standardisation from the original training run will misrepresent the new SKUs. Refit the scaler when retraining.

**Spherical cluster assumption:** K-Means assumes clusters are roughly spherical in feature space. The high-velocity clusters (0 and 3) are elongated and close together, which explains their lower per-cluster silhouette. DBSCAN or GMM might better capture their shape, but K-Means was preferred for interpretability and scalability.

**Insufficient history for new SKUs:** SKUs with fewer than 90 days of demand data produce unreliable seasonality_strength and demand_trend estimates. These SKUs should be assigned to clusters using only the first 7 features (excluding trend and seasonality) until sufficient history is accumulated.

**Static clusters:** SKU demand patterns change over time. A cluster assignment from this run represents the SKU's behaviour over 2022–2023. Assignments should be refreshed quarterly or when more than 15% of SKUs experience a meaningful ABC class change.

**No soft assignment:** K-Means assigns each SKU to exactly one cluster. SKUs on cluster boundaries (silhouette score < 0.1) may legitimately belong to multiple clusters. Consider Gaussian Mixture Models for a soft-assignment alternative.

---

## Recommended Use Cases

**Use K-Means clustering when:**
- Setting replenishment policies (order frequency, review period, safety stock formula)
- Determining which SKUs need vendor-managed inventory vs self-managed
- Grouping SKUs for warehouse slotting strategy (high-frequency cluster → Pick Face)
- Segmenting for ABC-adjusted forecasting models (cluster-specific MAPE targets)

**Prefer rule-based ABC when:**
- Communicating financial prioritisation to non-technical stakeholders
- Making quick procurement decisions where speed matters
- Comparing performance across time periods (ABC class is stable and interpretable)

**Do not use K-Means for:**
- Real-time stockout risk scoring (use a supervised model trained on stockout_flag)
- Individual SKU replenishment quantity calculation (use the demand forecasting pipeline)
- Decisions requiring audit trail and simple explanation to regulators

---

## Recommended Retraining Frequency

**Quarterly** is recommended. At each retraining:
1. Re-run `01_feature_engineering.py` on the updated demand data
2. Re-run `02_cluster_validation.py` — check if optimal k has changed
3. If k is unchanged, re-run `03_kmeans_clustering.py` with the existing k
4. Re-run `04_method_comparison.py` and `05_robustness_testing.py` to confirm stability
5. If more than 20% of SKUs have changed cluster assignment, review business impact before deploying new assignments

**Trigger retraining immediately** if: a new product category is added, a major warehouse opens or closes, or demand patterns shift significantly (e.g. macro-economic event, supply disruption).
