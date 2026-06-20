# Master Model Registry
## DHL Data Scientist Portfolio — All Projects

This registry consolidates every model, method, and statistical technique evaluated across the five core data science projects in this portfolio. It includes both deployed and rejected methods — recording what was tested and why each decision was made. The purpose is to demonstrate that production decisions are the output of a structured evaluation process, not a single model run.

All projects share the same underlying synthetic DHL dataset: 1,664 SKUs, 3 warehouses (NJ01, IL02, TX03), 730 days of operational data (Jan 2022 – Dec 2023), 500 customers, 68,941 outbound orders, and 219,000 WMS task records.

---

## Project 1 — SKU Segmentation

**Goal:** Replace rule-based ABC/XYZ classification with a data-driven segmentation that captures demand patterns, not just revenue rank.

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| K-Means (k=4) | Unsupervised clustering | 1,664 SKUs, 9 z-scored features, Jan 2022–Dec 2023 | Silhouette=0.5265, CH=1,549, DB=0.970, Seed-stability ARI=0.9989 (20 seeds) | **DEPLOYED — PRIMARY** | Passes all three gates: statistical quality (silhouette >0.50), robustness (ARI >0.95 across all 20 seeds), financial justification ($3,904,833 net annual value). Captures demand-pattern information that rule-based XYZ entirely misses (ARI vs XYZ=0.002). |
| Hierarchical Clustering (Ward linkage) | Unsupervised clustering | Same 1,664 SKUs × 9 features | Silhouette peak at k=4, dendogram agrees with K-Means optimal k | **VALIDATION ONLY** | Used as an independent check on K-Means k-selection. Ward linkage independently confirmed k=4 as the natural cut. Not deployed because K-Means is more scalable and equally interpretable for this catalogue size. |
| Rule-Based ABC Classification | Heuristic classification | Revenue-based ranking (Pareto cutoffs: A=top 20%, B=next 30%, C=bottom 50%) | ARI vs K-Means=0.885 — strong agreement on revenue dimension | **RETAINED — AUDIT OVERLAY** | High explainability for non-technical stakeholders and regulators. Retained alongside K-Means clusters as an audit overlay — planners can reference A/B/C labels while policy assignment uses K-Means clusters. |
| Rule-Based XYZ Classification | Heuristic classification | Coefficient of variation (CV) of demand | ARI vs K-Means=0.002 — near-zero agreement | **REPLACED BY K-MEANS** | XYZ captures only demand variability (CV), which K-Means encodes far more richly through 9 features including trend, seasonality, frequency, and order size. ARI=0.002 confirms K-Means is capturing something XYZ entirely misses. |

### K-Means Cluster Profiles

| Cluster | Label | n SKUs | Per-Cluster Silhouette | Policy Implication |
|---|---|---|---|---|
| 0 | High-Velocity / Low-Value | 113 | 0.245 | Continuous review, bulk order scheduling |
| 1 | Low-Velocity / Low-Value | 821 | 0.663 | Periodic review, minimal safety stock |
| 2 | Low-Velocity / High-Value | 572 | 0.462 | Periodic review, capital-weighted safety stock |
| 3 | High-Velocity / High-Value | 158 | 0.231 | Continuous review, VMI, peak safety stock |

### Robustness Tests (Project 1)

| Test | Result | Threshold | Pass/Fail |
|---|---|---|---|
| Seed stability — mean pairwise ARI (20 seeds) | 0.9989 | 0.95 | PASS |
| Seed stability — min pairwise ARI | 0.9951 | — | PASS |
| Seed stability — % pairs with ARI >0.95 | 100% | — | PASS |
| Feature ablation — min ARI (demand_frequency removed) | 0.965 | 0.70 | PASS |
| Outlier sensitivity — ARI after removing top 1% | 0.983 | 0.80 | PASS |

---

## Project 2 — Demand Forecasting

**Goal:** Identify the most cost-effective forecasting method for 1,664 SKUs across ABC classes and demand patterns. Training period: Jan 2022 – Sep 2023 (638 days). Test period: Oct 2023 – Dec 2023 (92 days).

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| Naive (last value) | Statistical baseline | Per-SKU daily demand | Overall MAPE=79.9%; A=85.4%, B=71.4%, C=83.0%. Monthly cost=$0.0002 | **NOT DEPLOYED — MONITORING BENCHMARK** | Worst MAPE across all classes. Retained only as a monitoring benchmark (a live model should always beat Naive) and emergency fallback during retraining failures. |
| Seasonal Naive (repeat prior week) | Statistical baseline | Per-SKU daily demand (period=7) | Overall MAPE=83.8%; A=93.1%, B=77.3%, C=80.9% | **NOT DEPLOYED** | Worst MAPE of all methods evaluated (83.8%), underperforming even the Naive baseline for A-class SKUs. Weekly seasonality at this level of demand aggregation is not a reliable signal. |
| SES (Simple Exponential Smoothing) | Statistical | Per-SKU, α optimised by MLE | Overall MAPE=66.5%; A=67.0%, B=56.9%, C=75.5%. Monthly cost=$0.0092 | **NOT DEPLOYED — PARETO FRONTIER** | Significant improvement over Naive (13.5pp MAPE gain). Forms the upper step of the Pareto-efficient frontier; LightGBM dominates it on both cost and accuracy. Retained as the cold-start fallback for new SKUs with <28 days of history. |
| ARIMA(p,d,q) | Statistical time series | 12-SKU sample; AIC-selected from 18 combinations | Mean MAPE=80.3%, Mean RMSE=49.0. Monthly cost=$1.48 | **NOT DEPLOYED** | MAPE is worse than Naive (80.3% vs 79.9%). Cost is $1.48/month vs $0.0022 for LightGBM — 670× more expensive. Strictly dominated on both dimensions. Most SKUs selected ARIMA(0,1,1), mathematically equivalent to SES, confirming the simpler method captures the same signal at a fraction of the cost. |
| SARIMA(p,d,q)(1,0,1,7) | Statistical time series | Same 12-SKU sample; seasonal block added to best ARIMA | Mean MAPE=80.6%, Mean RMSE=48.8. Monthly cost=$2.77 | **NOT DEPLOYED** | Even worse than ARIMA on MAPE (80.6% vs 80.3%). Monthly cost is $2.77 — 1,250× more expensive than LightGBM. Maintenance complexity 3.75/5. The seasonal component adds no predictive value over the non-seasonal ARIMA structure for weekly demand aggregates. |
| XGBoost (global model) | ML — gradient boosting | 112 eval SKUs, 15 features, TimeSeriesSplit CV (3 folds) | Overall MAPE=57.5%; A=68.5%, B=55.5%, C=48.6%. Monthly cost=$0.0033 | **NOT DEPLOYED** | Marginally worse than LightGBM (57.5% vs 57.4% MAPE) at 50% higher compute cost. LightGBM dominates on all dimensions. No case for deploying XGBoost alongside LightGBM. |
| LightGBM (global model) | ML — gradient boosting | 112 eval SKUs, 15 features, TimeSeriesSplit CV (3 folds); n_estimators=100, num_leaves=31, lr=0.05 | Overall MAPE=57.4%; A=66.5%, B=55.5%, C=57.4%. Monthly cost=$0.0022. Beats SES: p=0.046 (paired t-test), p=0.015 (Wilcoxon) | **DEPLOYED — PRIMARY (all ABC classes)** | Best MAPE, lowest ML cost, statistically significant improvement over SES best baseline (p=0.046). Pareto-dominant: no evaluated method achieves lower MAPE at lower cost. Single global model across 1,664 SKUs — retrains in ~30 seconds weekly. MAPE efficiency: 11,265 percentage-points per dollar spent. |
| Croston (demand level + interval) | Intermittent demand | 36 SKUs with >40% zero-demand days | MAPE=70.2%, RMSE=12.72, MAE=10.71, Bias=−1.39. Monthly cost=$0.0002 | **DEPLOYED — OVERRIDE for intermittent SKUs** | Purpose-built for sparse demand series. Reduces negative bias from −1.88 (Naive) to −1.39 on intermittent SKUs. MAPE 70.2% vs Naive 88.6% on the same 36 SKUs — 18.4pp improvement. Automatically triggered for SKUs with >40% zero-demand days in trailing 90 days. |
| Croston SBA (Syntetos-Boylan Approximation) | Intermittent demand | Same 36 intermittent SKUs | MAPE=70.3%, Bias=−1.88 | **NOT DEPLOYED — VARIANT EVALUATED** | Marginally worse MAPE than standard Croston (70.3% vs 70.2%) and identical bias to Naive on this dataset. The SBA correction over-adjusts for the synthetic data structure. Croston standard is preferred. |

### Statistical Significance (Project 2)

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| Paired t-test: LightGBM vs SES (n=112) | t=2.016 | 0.046 | Significant at p<0.05 |
| Wilcoxon signed-rank: LightGBM vs SES | W=2326 | 0.015 | Significant at p<0.05 |
| Mean improvement (LightGBM over SES) | — | — | 2.97pp (62.2% → 59.2% MAPE) |

---

## Project 3 — RFM Segmentation and A/B Test

**Goal:** Identify high-churn-risk customers using RFM scoring, design a statistically rigorous A/B test for a retention campaign, and quantify the economic case for scaling.

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| Quintile RFM Scoring | Segmentation / ranking | 398 customers with orders, Jan 2022–Dec 2023, 68,941 orders | Bootstrap stability CV=0.0000 across all 3 dimensions (50 iterations, 70% subsample). At Risk segment: 95 customers (23.9%) | **DEPLOYED — CUSTOMER SEGMENTATION** | Rank-percentile quintiles (1–5) across Recency, Frequency, Monetary dimensions. Bootstrap stability test confirms quintile boundaries are stable (CV < 0.10 threshold). F and M modestly correlated (Spearman ρ=0.313); R is orthogonal to both, confirming independent signals. |
| Two-Proportion Z-Test | Statistical hypothesis test | At Risk segment: n=47 treatment, n=47 control, stratified by Customer_Type | Z=2.457, p=0.014, 95% CI=[3.5pp, 27.0pp]. Post-hoc power=0.772 | **DEPLOYED — A/B TEST PRIMARY ANALYSIS** | Binary metric (converted/not) → proportions framework is the natural choice. Fully explainable to non-technical decision-makers. CLT applies at n≥30. Pre-registered before outcome data was observed. |
| Bootstrap Stability Test | Validation method | RFM scoring on 50 iterations, 70% subsamples | Max boundary CV=0.0000 across all 3 dimensions | **DEPLOYED — VALIDATION** | Confirms quintile boundaries are not artefacts of the specific sample. All CVs=0.0000 on synthetic data (consistent order patterns). Establishes that segmentation is reproducible. |
| Power Analysis (pre-test) | Experimental design | At Risk n=95, baseline conversion=83.2%, α=0.05, power=0.80 | Required n per group for 5pp MDE: 766. Detectable MDE at n=47: 15.1pp | **DEPLOYED — DESIGN VALIDATION** | Pre-registered MDE calculation determines what the test can detect given the available population. Confirms 15.1pp is the minimum detectable effect, which the cost model validates as economically meaningful (ROI ~62,739% at that lift). |
| Permutation Baseline | Validation | Randomised label shuffling to establish null distribution | Used to validate that stratified randomisation achieved balance | **DEPLOYED — VALIDATION** | Confirms stratified randomisation across 6 Customer_Type strata achieved balance. All 6 strata perfectly balanced (identical counts in both arms). |

### A/B Test Results (Project 3)

| Metric | Value |
|---|---|
| Treatment conversion | 97.8% (45/46) |
| Control conversion | 82.6% (38/46) |
| Absolute lift | 15.2pp |
| Relative lift | 18.4% |
| p-value (two-proportion z-test) | 0.014 |
| 95% CI | [3.5pp, 27.0pp] |
| Campaign cost (full rollout) | $760 ($8/customer) |
| Incremental revenue at observed lift | $482,652 |
| Net revenue | $481,892 |
| Realised ROI | ~63,407% |

---

## Project 4 — Warehouse Optimization

**Goal:** Optimise picker travel paths and SKU slotting assignments across three DHL warehouses to reduce operational travel distance and cost. Data: 219,000 WMS task records, 2,640 storage locations.

### Path Planning

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| Naive Path (WMS task order) | Heuristic baseline | 200-session validation sample, 5–15 stops/session | Mean distance=112.5 units | **NOT DEPLOYED — BASELINE** | Current state. Picker follows WMS task order with no route optimisation. Establishes the baseline against which all alternatives are measured. |
| Nearest-Neighbor Heuristic | Greedy path algorithm | Same 200-session sample | Mean distance=59.1 (−47.4% vs Naive). Gap from exact optimal=4.69% | **NOT DEPLOYED** | Evaluated as a stepping stone. Achieves 47.4% reduction but leaves 4.69% gap from optimal vs 2-Opt's 1.75% gap — at negligible additional compute cost. 2-Opt strictly dominates on quality. |
| 2-Opt Local Search | Combinatorial optimisation | Same 200-session sample | Mean distance=56.5 (−49.8% vs Naive). Gap from exact=1.75%. Runtime=0.34 ms/session. Monthly cost=$1.55 | **DEPLOYED — ROUTE SEQUENCING** | 98.3% of exact-optimal benefit at 0.34 ms/session vs 100 ms/session for exact enumeration. 49.8% distance reduction vs baseline. Annual travel saving=$2,452,887. WMS integration: add route-sequencing step between task generation and task assignment. |
| Exact TSP (permutation enumeration) | Combinatorial optimisation | 91 sessions with ≤8 stops | Mean distance=~55.5 (−50.7% vs Naive). Gap=0%. Monthly cost=$450 | **NOT DEPLOYED — REFERENCE ONLY** | NP-hard: 15 stops = 15! ≈ 1.3 trillion permutations. Monthly cost at full scale: $450 vs $1.55 for 2-Opt. Incremental improvement over 2-Opt: 1.75% — negative ROI. Used as the ground-truth benchmark to validate 2-Opt quality only. |

### SKU Affinity / Market Basket Analysis

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| Manual Co-occurrence Counting (Apriori equivalent at min_support≈0.001) | Association rules | 2,766 pick sessions, 1,664 SKUs | 5,807 pairs with ≥2 co-occurrences; top pair lift=71.8 (AUT-001772 ↔ FSH-000914) | **DEPLOYED — AFFINITY DETECTION** | Standard Apriori at 1% min_support would eliminate all 5,807 pairs (max pair support=0.14%). Manual co-occurrence counting with permutation validation achieves the same result with full transparency and no library dependency. |
| Standard Apriori (mlxtend) | Association rules | Same sessions | Produces zero pairs at standard min_support=1% | **NOT DEPLOYED** | With max pair support=0.14% across 2,766 sessions, standard Apriori thresholds eliminate the entire candidate set. The algorithm is appropriate for denser catalogues. |
| Permutation Test (100 iterations per pair) | Statistical validation | Top-50 candidate pairs by lift | All 50 pairs validated at p<0.05 (all p=0.00) | **DEPLOYED — PAIR VALIDATION** | Confirms that observed co-occurrence lifts are not due to random chance. 100 permutations per pair is sufficient for p=0.00 resolution. Provides statistical backing for affinity-based slotting decisions. |

### Slotting Optimisation

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| Current Slotting (alphabetical/default WMS) | Heuristic baseline | 500-session validation sample | Mean travel distance=60.36 | **NOT DEPLOYED — BASELINE** | Alphabetical WMS-default assignment places top SKUs in the Bulk zone (furthest from depot). This is the operational status quo being replaced. |
| Greedy Heuristic (rank by pick count) | Heuristic | SKU pick frequency rankings | Mean travel distance=60.13 (−0.4% vs current) | **NOT DEPLOYED** | Simple rank-order assignment. Marginal improvement (0.4%) vs LP Joint's 1.5%. Does not incorporate affinity bonus. LP dominates on quality with similar compute cost. |
| LP Joint Optimization (PuLP/CBC) | Linear programming | Top 200 SKUs by pick count, affinity pairs, zone capacities. Solve time <1 second | Mean travel distance=59.44 (−1.5% vs current). Annual slotting saving=$75,064. Monthly compute=$83.20 | **DEPLOYED — MONTHLY BATCH** | Simultaneously optimises distance minimisation and affinity co-location. Solves in <1 second (CBC solver). Improvement bounded at 1.5% because current data maps to zones, not individual bin locations — location-level reformulation estimated at 5–10% improvement. |

### Temporal Stability / Cadence Optimisation

| Method | Key Result | Cadence Recommendation |
|---|---|---|
| Spearman rank correlation of pick frequencies (8 quarterly windows) | Mean ρ=0.023 (near-random on synthetic uniform data) | Monthly re-optimisation |
| SKU tier change rate per quarter | Mean=74.4%/quarter | Monthly re-optimisation |
| Affinity pair overlap between consecutive quarters | 0.0% | Monthly affinity refresh |

---

## Project 5 — WMS Anomaly Detection

**Goal:** Detect warehouse accuracy degradation, volume anomalies, and operator-level performance outliers across 3 DHL warehouses. Data: 2,190 warehouse-day records, 3,662 operator-day records (Jan 2022 – Dec 2023).

### Warehouse-Level Detection

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| SPC — Western Electric Run Rules (4 rules) | Statistical process control | 2,190 warehouse-day KPI records, 30-day rolling baseline | 286 anomaly records flagged across 3 warehouses over 2 years. Applied to pick_accuracy_rate, total_task_volume, error_count | **DEPLOYED — LAYER 1 (Daily)** | Universally understood by operations teams with zero explanation overhead. Rules 1–4 detect point breaches, clusters, sustained shifts, and trends. Primary daily operational layer. CUSUM detects an additional 15 gradual drift events over 2 years. Annual cost (FP investigation): $8,748 for SPC-only; $1,320 under hybrid (IF replaces SPC as primary filter). |
| CUSUM (Cumulative Sum Chart) | Statistical process control | Same 2,190 records. Parameters: k=0.5, h=4.0 | 15 gradual drift events detected that Western Electric rules alone would have missed over 2-year window | **DEPLOYED — LAYER 1 COMPLEMENT (Daily)** | Detects gradual accuracy drift 2–3 weeks earlier than threshold-only monitoring. Resets after each alert to prevent cascade. Applied to pick_accuracy_rate. Standard industry parameters k=0.5, h=4.0. |
| Isolation Forest (warehouse-day level) | Unsupervised ML — anomaly detection | Same 2,190 records, 6 KPI features, StandardScaler normalised. n_estimators=200, contamination=0.05 | 110 warehouse-days flagged (5.0%). 148 IF-flagged days not flagged by SPC over 2 years. SPC vs IF agreement Jaccard=0.103 | **DEPLOYED — LAYER 2 (Weekly)** | Catches multi-feature anomalies where no single metric breaches its individual threshold but the combination of accuracy, volume, duration, and error counts is unusual. 18 additional potential anomalies per quarter vs SPC-only. Annual cost=$1,320 — $7,428 less than SPC-only due to lower FP investigation burden. |
| LOF — Local Outlier Factor (warehouse-day level) | Unsupervised ML — anomaly detection | Same 2,190 records, n_neighbors=20, contamination=0.05 | IF vs LOF Jaccard=0.152, Cohen's κ=0.225. SPC vs LOF Jaccard=0.063 | **NOT DEPLOYED at warehouse level** | Low Jaccard with IF (0.152) — both methods flag different signals at warehouse level, doubling alert volume without proportionate signal gain. LOF is re-deployed at operator level where it adds distinct value. |

### Sigma Threshold Sensitivity (SPC)

| Threshold | IL02 flags | NJ01 flags | TX03 flags |
|---|---|---|---|
| 2.0σ | 85 | 81 | 84 |
| 2.5σ | 44 | 29 | 37 |
| **3.0σ (deployed)** | **11** | **12** | **12** |

Moving from 3σ to 2σ increases daily flags by 600–673%. Production decision: 3σ to minimise alert fatigue.

### Operator-Level Detection

| Model / Method | Type | Training Data | Key Metrics | Production Status | Rationale |
|---|---|---|---|---|---|
| LOF — Local Outlier Factor (operator-day level) | Unsupervised ML — anomaly detection | 3,662 operator-day records (≥5 tasks/day), n_neighbors=20, contamination=0.05. Fitted per warehouse | 139 operator-day anomalies across 2 years (~6 flags/month across 60 operators) | **DEPLOYED — LAYER 3 (Weekly)** | Identifies operators with unusual combinations of task rate, accuracy, and duration — not just low accuracy. Distinguishes system issues (affects all operators) from individual issues (specific operator). ~6 flags/month is a manageable review volume for shift managers. Annual cost=$2,184. |

### Cross-Method Agreement (Project 5)

| Method Pair | Both Flagged | Jaccard | Cohen's κ | Interpretation |
|---|---|---|---|---|
| SPC vs IF | 33 days | 0.103 | 0.127 | Low — methods detect different signals (temporal vs multi-feature) |
| SPC vs LOF | 21 days | 0.063 | 0.054 | Very low — complementary |
| IF vs LOF | 29 days | 0.152 | 0.225 | Low — IF and LOF agree most among ML methods but still detect different anomaly types |

**Interpretation:** Days flagged by both SPC and IF simultaneously (33 days) represent the highest-confidence anomaly signals. Low pairwise agreement is not a failure — it confirms the methods are capturing different anomaly types and the hybrid system adds coverage that no single method achieves alone.

### Robustness Tests (Project 5)

| Test | Result | Conclusion |
|---|---|---|
| Baseline window: 18-month vs 24-month | 70.8% overlap in flagged anomalies | Stable with shorter history; 24-month window preferred for initial deployment |
| Contamination sensitivity: 0.03 vs 0.05 | Jaccard=0.600 | Moderate sensitivity — treat as tunable hyperparameter post-deployment |
| Contamination sensitivity: 0.05 vs 0.10 | Jaccard=0.502 | Core extreme anomalies are stable; borderline cases are sensitive |

---

## Registry Summary

| Project | Total Methods Evaluated | Methods Deployed | Decision Complexity |
|---|---|---|---|
| 1 — SKU Segmentation | 4 | 2 (K-Means primary + ABC audit overlay) | Simple vs complex both justified; K-Means adds unique demand-pattern signal |
| 2 — Demand Forecasting | 8 | 2 (LightGBM + Croston override) | 6 methods rejected; ARIMA dominated by simpler/cheaper options |
| 3 — RFM and A/B Test | 4 | 4 (all used in different roles) | No "winner/loser" — each method plays a distinct role in the pipeline |
| 4 — Warehouse Optimization | 7 | 4 (2-Opt + co-occurrence + permutation + LP) | Exact TSP rejected on ROI; Apriori rejected on data sparsity |
| 5 — WMS Anomaly Detection | 5 | 4 (SPC + CUSUM + IF + LOF-operator) | Hybrid layers cover distinct signal types; LOF at warehouse level rejected |
| **TOTAL** | **28** | **16** | — |
