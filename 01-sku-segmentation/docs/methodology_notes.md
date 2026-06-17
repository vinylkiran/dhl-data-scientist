# Methodology Notes — SKU Segmentation
## DHL Data Scientist Portfolio — Project 01 (v2.0 rebuild, 2026-06-17)

This document records every significant modelling decision made during the SKU segmentation project, along with the alternatives considered and the reasoning behind each choice. It is intended for a technical reviewer who wants to understand not just what was done, but why.

---

## 1. Why Standardise Before Clustering?

**Decision:** Z-score standardise all 9 features before fitting K-Means.

**Why it matters:** K-Means minimises the sum of squared Euclidean distances between points and their assigned centroid. If features are on different scales, features with larger magnitudes will dominate the distance calculation. In this dataset, `total_revenue` ranges from £323 to £598 million — a raw difference of £10 million would completely overshadow a 10-unit difference in `mean_daily_demand`. Without standardisation, K-Means would effectively cluster on revenue alone.

**Alternative considered — MinMax scaling:** MinMax scaling (rescale to [0,1]) preserves the shape of the distribution but is sensitive to outliers. Since `total_revenue` has a very heavy right tail (a handful of SKUs with £100M+ revenue), MinMax would compress all mid-revenue SKUs into a narrow band. Z-score scaling is more robust to this.

**Alternative considered — No scaling:** Rejected. Without scaling, the clustering would reproduce the ABC classification by revenue, adding no new insight.

**Alternative considered — Log transformation:** Log-transforming skewed features (revenue, mean_daily_demand) before z-scoring would reduce the influence of extreme outliers and make the distribution closer to normal. This was considered but not implemented because: (1) the robustness test showed the current scaling is already outlier-stable (ARI=0.983 after removing top 1%), and (2) log transformation would reduce the interpretability of feature weights in the centroids.

---

## 2. Why These 9 Features?

The feature set was designed to capture three independent dimensions of SKU demand behaviour:

**Volume dimension (what is the scale of demand?)**
- `mean_daily_demand` — primary velocity signal
- `total_revenue` — financial importance, independent of unit count
- `revenue_rank_pct` — compressed revenue percentile to handle the right tail

**Pattern dimension (how regular and predictable is demand?)**
- `std_demand` — absolute variability
- `cv_demand` — relative variability (XYZ signal)
- `demand_frequency` — intermittent vs continuous demand
- `avg_order_size` — small-frequent vs large-occasional pattern

**Trajectory dimension (how is demand evolving?)**
- `demand_trend` — growing vs declining SKUs
- `seasonality_strength` — seasonal buffering requirement

**Features explicitly excluded:**

`Unit_Cost` / `Unit_Price` from sku_master.csv — excluded because these are *product attributes*, not demand patterns. Including them would make the clusters sensitive to pricing strategy rather than operational demand behaviour. Revenue is already included and implicitly captures the value of demand.

`Warehouse_ID` — excluded. Clustering is performed at SKU level aggregated across warehouses. Warehouse-specific patterns are relevant for slotting (Project 04) but would fragment the SKU segments unnecessarily here.

`Stockout_Flag` / `Stockout_Rate` — excluded as a feature because it is a *target outcome* we want to evaluate the clusters against, not an input to the segmentation. Including it would create circular validation.

`Quantity_Fulfilled` — redundant with `Quantity_Demanded` in most rows; `Stockout_Flag` already captures the gap.

**Correlation analysis:** The correlation matrix from `01_feature_engineering.py` shows high correlation between `mean_daily_demand`, `std_demand`, and `avg_order_size` (r=0.94–0.99). Despite the collinearity, all three were retained because they each capture a subtly different aspect of the volume dimension and the feature ablation test confirms removing any one of them has minimal impact on cluster structure (ARI > 0.97 in all cases). K-Means does not have the same problems with collinearity as regression models.

---

## 3. Why K-Means Over DBSCAN or Gaussian Mixture Models?

**K-Means** was selected as the primary algorithm. The alternatives considered:

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**

Advantages:
- Does not require specifying k in advance
- Can detect arbitrarily shaped clusters
- Classifies outliers as noise rather than forcing them into a cluster

Disadvantages and why it was rejected:
- Two hyperparameters (epsilon, min_samples) that are harder to tune than k, and sensitive to scale
- Performance degrades in high-dimensional spaces (this data has 9 features — moderate, but not ideal for DBSCAN)
- The resulting clusters are harder to profile and explain to business stakeholders
- DBSCAN would likely classify the long-tail C-class SKUs as a single large cluster and the high-value A-class as multiple tiny clusters, which is not operationally useful
- Does not naturally produce centroids, making it impossible to assign new SKUs to clusters without re-running the full algorithm

**Gaussian Mixture Models (GMM):**

Advantages:
- Soft cluster assignments (each SKU has a probability of belonging to each cluster)
- Can model elliptical clusters (not just spherical, unlike K-Means)
- Better for overlapping clusters

Disadvantages and why it was rejected:
- Computationally more expensive (EM algorithm convergence can be slow)
- Requires the same k selection problem as K-Means, plus covariance structure selection (full, tied, diag, spherical)
- More prone to overfitting on small datasets; with k=4 and 1,664 SKUs the sample/cluster ratio is adequate, but GMM's additional parameters reduce confidence
- The clusters in this data are relatively compact and well-separated (silhouette=0.524), so the soft-assignment advantage of GMM would not change many assignments
- Cluster interpretability for warehouse managers is better with hard assignments ("this SKU is a fast-mover") than probabilities ("this SKU is 60% fast-mover, 40% mid-tier")

**Hierarchical clustering (Ward linkage) was used as a validation method, not a production method:**

The dendrogram from `02_cluster_validation.py` confirmed that the Ward-linkage hierarchy also suggested k=4–5 as natural cut points (silhouette peaked at k=4 in hierarchical as well). However, hierarchical clustering has O(n²) memory complexity, making it unsuitable for production use when new SKUs must be added incrementally.

**Conclusion:** K-Means is the correct choice for this use case because: (1) the data is in a continuous 9-dimensional space; (2) clusters are compact and relatively spherical (validated by silhouette); (3) centroids provide a natural mechanism for classifying new SKUs; and (4) the method is transparent and explainable to inventory managers.

---

## 4. How Was Optimal k Determined?

**Four metrics were computed for k=2 through k=12:**

1. **Inertia (elbow method):** The within-cluster sum of squares. Decreasing rapidly then levelling off. In this data, the second derivative of inertia suggested k=11 (a spurious result driven by the smooth exponential decay of inertia, which makes the "elbow" ambiguous).

2. **Silhouette score:** Measures average cohesion minus average separation for each point, normalised by the maximum of the two. Range [-1, +1]. This dataset peaks at k=2 (silhouette=0.589), then shows a secondary peak at k=4 (0.527). Silhouette is theoretically the most sound single metric.

3. **Calinski-Harabasz (CH) score:** Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better. Peaks at k=2 in this data. The CH score is biased towards smaller k because adding clusters always increases within-cluster dispersion faster than between-cluster dispersion at higher k.

4. **Davies-Bouldin (DB) score:** Average of the worst-case ratio of within-cluster scatter to between-cluster separation for each cluster pair. Lower is better. Peaks (i.e. is minimised) at k=3 in this data.

**Why multiple metrics instead of just one:**

No single metric is universally optimal. Silhouette captures the user-relevant notion of "good clustering" but can mislead when clusters have very different densities. CH is biased towards small k. DB penalises clusters that are too close but does not reward compactness. The elbow is a heuristic that requires subjective judgement. Using all four provides a richer picture and reduces the chance that a quirk of any one metric drives the decision.

**Why k=4 rather than k=2 (the statistical optimum):**

k=2 maximises both silhouette and CH, but k=2 produces two groups that map almost perfectly to "high-demand" and "low-demand". This reproduces the ABC classification with one class instead of three, adds no new information, and provides no actionable segmentation for inventory policy. The constraint of k≥3 was imposed because any inventory management strategy requires at minimum three tiers (premium, standard, long-tail).

Within the k≥3 range, k=4 is the first local silhouette maximum (0.527 > k=3 at 0.509, then declining again at k=5 before rising slowly). k=4 also achieves the highest CH score in the k≥3 range (1,549 at k=4 vs 1,519 at k=3), confirming it as the natural cluster structure.

The resulting 4 clusters map cleanly to four inventory archetypes:
- High-Velocity / High-Value: A-class, continuous high-demand, high-revenue
- High-Velocity / Low-Value: High-frequency commodity items, low unit price
- Low-Velocity / Low-Value (transitional): B-class mid-tier, moderate demand
- Low-Velocity / Low-Value (long tail): C-class, intermittent, low revenue

**Hierarchical cross-check:** The Ward-linkage dendrogram and silhouette sweep for hierarchical clustering also favoured k=4–5 (hierarchical sil: k=2→0.576, k=3→0.478, k=4→0.503, k=5→0.502). This independent validation from a completely different algorithm supports the choice of k=4.

---

## 5. Seasonality Strength Calculation

**Decision:** Use a classical seasonal decomposition proxy (monthly mean deviation) rather than STL decomposition.

**Why not STL (Seasonal-Trend decomposition using LOESS):** STL produces superior seasonality estimates for longer series, but it requires at least 2 complete seasonal cycles and is computationally expensive for 1,664 SKUs. At 730 days, each SKU has exactly 2 years — marginal for STL.

**The proxy method:**
1. Compute the monthly mean of daily demand for each month (Jan–Dec)
2. The seasonal component = monthly_mean - overall_mean
3. Seasonal variance = variance of the seasonal component
4. Seasonality strength = seasonal_variance / total_variance

This produces a value in [0, 1] where 0 = no seasonality and 1 = perfectly seasonal. The average value across the 1,664 SKUs was 0.043, indicating most SKUs have low seasonality in this synthetic dataset — as expected for a diversified logistics catalogue.

---

## 6. Demand Trend Calculation

**Decision:** Simple linear regression slope of demand over the 730-day period.

**Why not more sophisticated trend detection:** The goal is a simple, interpretable feature that distinguishes growing from declining SKUs. The regression slope (in units/day) achieves this. A positive slope of 0.03 means demand is growing by about 1 unit/month. The mean slope across all SKUs was approximately 0.004 units/day (very modest growth), with a standard deviation of 0.012, suggesting most SKUs are relatively stable. The feature_ablation test shows trend is one of the more impactful features (ARI drops to 0.969 when removed), suggesting it contributes meaningfully to cluster separation beyond what the other features capture.

---

## 7. Aggregation: Why SKU Level Across All Warehouses?

Demand was aggregated across all three warehouses (NJ01, IL02, TX03) to the SKU × date level before engineering features. An alternative would be to cluster at the SKU × warehouse level, producing 3× as many data points and potentially different assignments per location.

**Reason for SKU-level aggregation:**
- The goal is to characterise intrinsic demand behaviour of each SKU, not warehouse-specific stocking patterns
- Warehouse allocation decisions are a downstream use case (handled in Project 04 — Warehouse Operations)
- Aggregating across warehouses smooths out location-specific noise and produces more stable feature estimates
- 1,664 SKUs × 3 warehouses = 4,992 data points would make cluster profiles harder to translate into SKU-level procurement decisions

---

## 8. Handling of Inactive SKUs

Of the 2,000 SKUs in `sku_master.csv`, 336 (16.8%) had no records in `daily_demand.csv`. These were excluded from the feature matrix and clustering entirely. In production, these SKUs should be classified by business rule (e.g. assign to "Low-Velocity / Low-Value" by default until sufficient demand history is available) rather than excluded from the inventory management framework.

---

## 9. Why Cost and Maintenance Were Treated as First-Class Criteria

A technically superior model is not always the right model to deploy. This analysis explicitly modelled total cost of ownership alongside statistical quality for two reasons:

**Reason 1 — Maintenance is real cost:** K-Means requires periodic revalidation of the optimal k, feature-set review, and more complex integration than a simple threshold rule. These tasks have real engineer-hours attached. The methodology assigns a 1–5 complexity score to four maintenance dimensions (revalidation, explainability, justification, integration) and converts these to dollar cost at $50/hr × 4 hrs/quarter. This makes the comparison honest: a 40% compute cost increase is trivial ($0.013/yr vs $0.001/yr), but a maintenance complexity increase is not.

**Reason 2 — Incremental value must exceed incremental cost:** The primary quantified value from K-Means over rule-based is the reclassification of 70 A-class SKUs placed in Low-Velocity clusters — SKUs that ABC over-rates and therefore over-stocks. At a 10% catch rate on their stockout exposure (a deliberately conservative assumption), the avoided revenue-at-risk from correcting these misclassifications is estimated at **$3.9M/yr**. This comfortably exceeds all incremental costs.

**What "10% catch rate" means:** We do not assume K-Means eliminates all stockout risk on misclassified SKUs. We assume that better segmentation leads to adjusted safety stock targets that prevent 10 out of 100 stockout events that would otherwise occur. This is a conservative, auditable assumption that can be replaced with an empirical estimate once the model is in production and stockout outcomes can be measured against cluster assignments.

**The net value ($3,904,833/yr) should be interpreted as a lower bound**, because it counts only the value of preventing stockouts on over-rated A-class SKUs. It does not count: (a) savings from reducing over-stock on these same SKUs, (b) improved forecast model accuracy when cluster-specific MAPE targets are applied, or (c) warehouse slotting efficiency gains from velocity-aligned cluster assignments.
