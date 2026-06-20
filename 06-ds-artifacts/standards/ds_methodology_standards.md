# Data Science Methodology Standards
## DHL Data Scientist Portfolio

These standards were applied consistently across all five projects in this portfolio. They reflect production-grade data science practice — the discipline required when models inform operational decisions, not just when they demonstrate academic competence. Each standard is stated as a rule, followed by the rationale and its concrete application across the five projects.

---

## Standard 1: Always Compare Against a Simple Baseline First

**Rule:** Before evaluating any complex model, establish a naive or rule-based baseline and measure it rigorously. A model that does not beat a naive baseline provides zero value regardless of its sophistication.

**Rationale:** The value of a model is measured relative to what you already have, not in absolute terms. A 60% MAPE for a demand forecast sounds poor; a 60% MAPE that beats a 79.9% Naive baseline is a significant operational improvement. Without an explicit baseline, it is impossible to state whether a model is adding value or just adding cost. Baselines also provide a floor for monitoring: if a deployed model's live MAPE approaches the Naive baseline, something has broken.

**Applied in this portfolio:**

- **Project 1 (SKU Segmentation):** K-Means compared against rule-based ABC (ARI=0.885 agreement on revenue dimension) and XYZ (ARI=0.002 — near-zero agreement, revealing that K-Means adds genuinely independent demand-pattern information). The baseline comparison demonstrated that K-Means did not simply replicate existing classifications but added new signal.
- **Project 2 (Demand Forecasting):** Naive (MAPE=79.9%) and Seasonal Naive (MAPE=83.8%) established before any statistical model was evaluated. SES was then benchmarked against Naive (13.5pp improvement). All ML models were benchmarked against SES. ARIMA's evaluation showed it performs no better than Naive — a finding only visible because the baseline was measured.
- **Project 3 (RFM):** Quintile boundary stability was validated against a permuted-random-assignment baseline. The A/B test used the At Risk segment's historical conversion rate (83.2%) as the explicit pre-specified baseline for MDE and power calculations.
- **Project 4 (Warehouse Optimization):** Naive WMS task order (mean distance 112.5 units) was the baseline for path planning. Alphabetical WMS default slotting (mean distance 60.36) was the baseline for slotting. All improvements are stated relative to these.
- **Project 5 (WMS Anomaly Detection):** SPC with Western Electric rules was the operations-standard baseline before any ML method was considered. IF and LOF were evaluated on how much additional signal they contributed beyond SPC, not as standalone replacements.

---

## Standard 2: Validate with Multiple Independent Metrics

**Rule:** Never trust a single evaluation metric. Use at least two independent measures of performance, and prefer metrics that capture different failure modes.

**Rationale:** Goodhart's Law states that when a measure becomes a target, it ceases to be a good measure. Any single metric can be gamed, overfit, or misrepresent model quality in specific conditions. MAPE breaks down on intermittent series (division by near-zero). Silhouette score can be artificially inflated by choosing k=2. Precision alone misses recall failures. Using at least two independent metrics ensures that a model performing well on one while failing on another is visible before deployment — not after.

**Applied in this portfolio:**

- **Project 1 (SKU Segmentation):** Four independent internal validation metrics: Silhouette Score (0.5265), Calinski-Harabasz (1,549), Davies-Bouldin (0.970), and hierarchical cross-check (Ward linkage independently confirms k=4). ARI and NMI used for external comparison against ABC/XYZ. No single metric could have been sufficient — CH peaks at k=2, DB at k=6; only the combination of all four, alongside operational interpretability, supports k=4.
- **Project 2 (Demand Forecasting):** Four accuracy metrics reported per model: MAPE, RMSE, MAE, and Bias. MAPE alone misleads on intermittent SKUs (denominator near zero). RMSE penalises large errors quadratically and was used to capture A-class high-value miss risk. Bias was used to detect systematic over- or under-forecasting patterns (Croston reduces bias from −1.88 to −1.39 on intermittent SKUs). Statistical significance confirmed by both paired t-test and Wilcoxon signed-rank test.
- **Project 3 (RFM and A/B Test):** Primary metric (conversion rate lift, p-value) supplemented by confidence interval, guardrail metrics (AOV t-test, unsubscribe, OTIF), and post-hoc power. A significant p-value without a valid CI or passing guardrails would not have justified scale.
- **Project 4 (Warehouse Optimization):** 2-Opt evaluated on mean distance reduction (49.8% vs Naive) AND gap from exact optimal (1.75%). A method that achieves 49.8% reduction but is 20% from optimal would have a different risk profile. LP slotting evaluated on both travel distance AND affinity co-location objective. Affinity pairs evaluated on raw lift AND permutation test p-value — neither alone is sufficient.
- **Project 5 (WMS Anomaly Detection):** Jaccard similarity and Cohen's kappa used jointly to assess cross-method agreement. Jaccard(SPC vs IF)=0.103, κ=0.127. The two metrics together show both the absolute overlap and the agreement adjusted for chance. Sigma threshold evaluated across multiple values (2σ, 2.5σ, 3σ) to characterise sensitivity. Contamination evaluated across multiple values to characterise IF stability.

---

## Standard 3: Test Robustness Before Trusting Any Result

**Rule:** Re-run with different random seeds, different time windows, different hyperparameter values, and different data subsets. A result that only holds under one specific configuration is fragile in production.

**Rationale:** A model that achieves good metrics in a single evaluation run may be fitting the noise in that run. Random seeds affect clustering initialisation, train/test splits, and tree ensemble construction. A model that is sensitive to these should not be trusted without stability confirmation. In production, the model will be retrained on new data — if the result is seed-dependent or window-dependent, the production behaviour is unpredictable.

**Applied in this portfolio:**

- **Project 1 (SKU Segmentation):** 20-seed stability test (mean pairwise ARI=0.9989, min=0.9951, 100% of seed pairs with ARI >0.95). Feature ablation test across all 9 features (min ARI=0.965 when demand_frequency removed). Outlier sensitivity test (ARI=0.983 after removing top 1% outlier SKUs). All three robustness tests passed before the deployment recommendation was written.
- **Project 2 (Demand Forecasting):** TimeSeriesSplit cross-validation with 3 folds — no data leakage (no test periods are visible during training). This is the correct validation strategy for time-series: a random K-fold split would allow future data to inform past-period model training, producing optimistically biased metrics.
- **Project 3 (RFM and A/B Test):** Bootstrap stability test on quintile boundaries: 50 iterations, 70% subsamples — all CVs = 0.0000, confirming RFM quintile boundaries are stable. Power analysis pre-specified the minimum detectable effect before outcome data was observed — preventing post-hoc re-framing of the result.
- **Project 4 (Warehouse Optimization):** 8 rolling quarterly windows (Q1 2022 – Q4 2023) for pick-frequency rank correlation (mean Spearman ρ=0.023 — near-random) and tier-change rate (mean 74.4%/quarter). This stability analysis directly drives the monthly re-optimisation cadence. Affinity pair stability tested across consecutive quarters (0% pair overlap — further evidence for monthly refresh).
- **Project 5 (WMS Anomaly Detection):** Baseline window sensitivity: 18-month vs 24-month history (70.8% overlap in flagged anomalies — stable). Sigma threshold sensitivity: 2σ produces 600–673% more flags than 3σ — directly informs the production sigma choice. Contamination sensitivity: Jaccard(0.03 vs 0.05)=0.600, Jaccard(0.05 vs 0.10)=0.502 — documents the tuning surface before deployment.

---

## Standard 4: Treat Cost and Explainability as First-Class Criteria

**Rule:** For every model comparison, quantify: (a) compute cost per month, (b) maintenance complexity on a 1–5 scale with breakdown, (c) explainability effort for operations teams, and (d) incremental value over the simple baseline. The decision is not made on accuracy alone.

**Rationale:** A model's full cost includes compute, retraining cadence, analyst time for monitoring, operations team training, and the cost of misaligned decisions when the model is misunderstood. A highly accurate model that requires a data scientist to interpret every output before operations can act is not production-grade. Conversely, a model that is cheaper but provides demonstrably less value than an alternative of similar complexity has a negative cost-benefit ratio on the accuracy dimension.

**Applied in this portfolio:**

- **Project 1 (SKU Segmentation):** Compute cost comparison: Rule-based $0.001/yr vs K-Means $0.013/yr (delta negligible). Maintenance: K-Means $2,200/yr vs Rule-based $2,400/yr (K-Means marginally cheaper due to reduced manual explainability effort once embedded). Net annual value quantified at $3,904,833. Explainability preserved by retaining ABC labels alongside K-Means cluster assignments.
- **Project 2 (Demand Forecasting):** Explicit cost-accuracy Pareto frontier across all 8 methods. ARIMA: $1.48/month, MAPE=80.3%. LightGBM: $0.0022/month, MAPE=57.4%. ARIMA is strictly dominated — 670× more expensive, 22.9pp worse. MAPE efficiency (pp per dollar) calculated for all methods: LightGBM=11,265 pp/$, ARIMA=−0.24 pp/$ (negative — costs more, performs worse). These calculations made the decision unambiguous.
- **Project 3 (RFM and A/B Test):** Explainability was explicit in method selection: two-proportion z-test chosen partly because it is "fully explainable to non-technical stakeholders: we compare the fraction who responded in each group." The statistical significance threshold and guardrail metrics were pre-specified in plain language for the VP audience. $8 cost-per-contact and $33K AOV stated upfront so the break-even calculation (0.024pp lift) required no data science knowledge to interpret.
- **Project 4 (Warehouse Optimization):** Exact TSP rejected despite being provably optimal: monthly cost $450 vs $1.55 for 2-Opt, for 1.75% incremental benefit. The incremental saving (1.75% of $2.45M = ~$43K/year) minus the incremental cost ($450 − $1.55 = $5,376/year) is positive but marginal, and operational complexity at scale makes it negative in practice. Monthly re-optimisation labour cost ($92,798/year) explicitly included in the net benefit calculation — not hidden.
- **Project 5 (WMS Anomaly Detection):** Explainability drove the deployment of SPC as the primary layer: "control charts are universally understood by ops teams — zero explanation overhead." IF was explicitly not deployed daily in the first phase because "alert fatigue risk before trust baseline established." Cost comparison included FP investigation time ($8,748/year for SPC-only vs $1,320/year for hybrid) — demonstrating that the simpler-looking method was in fact the more expensive option in full operational cost.

---

## Standard 5: Pre-Register Test Design Before Observing Results

**Rule:** For any causal inference test (A/B tests, hypothesis tests, intervention evaluations), the test design — primary metric, statistical test, alpha level, power, sample size, stopping rule, and guardrail metrics — must be documented and locked before any outcome data is examined.

**Rationale:** Post-hoc hypothesis selection ("HARKing" — Hypothesising After Results are Known) inflates false positive rates to unacceptable levels. An analyst who can see the data while choosing the test will, consciously or not, select metrics and thresholds that produce significant results. In a business context, a false positive A/B test result that leads to scaling an ineffective campaign can cost far more than the analysis itself. Pre-registration is the discipline that separates a valid causal claim from a data artefact.

**Applied in this portfolio:**

- **Project 3 (RFM and A/B Test):** Full pre-registration document (`docs/pre_registration.md`) written and assignment file locked before any outcome data was generated. Pre-specified: primary metric (conversion within 90 days), statistical test (two-proportion z-test, two-tailed), α=0.05, power=0.80, n per group=47, 90-day fixed horizon, no early stopping, Bonferroni correction if guardrails tested jointly. The stopping rule explicitly prohibits peeking — no intermediate looks at conversion rates during the 90-day window. This discipline is what makes the p=0.014 result interpretable as a real signal rather than a cherry-picked number.

---

## Standard 6: Use Time-Aware Validation for All Time-Series Problems

**Rule:** For any model trained on time-series data, validation must respect temporal ordering. Never use a random K-fold cross-validation split on time-series data. Use TimeSeriesSplit, walk-forward validation, or a fixed temporal holdout.

**Rationale:** A random K-fold split on time-series data allows the model to train on future observations and test on past observations. Because time-series data has autocorrelation, this creates data leakage that produces optimistically biased performance metrics. The model appears to perform well in validation but fails in production, where it only ever sees past data before making future predictions. The gap between validation MAPE and live MAPE will be persistently larger than expected.

**Applied in this portfolio:**

- **Project 2 (Demand Forecasting):** TimeSeriesSplit with n_splits=3 for ML model hyperparameter selection and cross-validation. Fixed temporal holdout for final evaluation: training Jan 2022 – Sep 2023, test Oct 2023 – Dec 2023 (92 days, no overlap). Features use only lag values and rolling statistics computed from historical data — no future values leak into feature construction.
- **Project 4 (Warehouse Optimization, cadence testing):** 8 rolling quarterly windows (Q1 2022 – Q4 2023) tested in sequential order. Consecutive-window correlation (Spearman ρ=0.023) and tier-change rate (74.4%) measured between adjacent periods — not across randomly selected period pairs.
- **Project 5 (WMS Anomaly Detection):** Rolling 30-day baseline (not static baseline from training period) for SPC control limits. Robustness test using an 18-month training window (Jan 2022 – Jun 2023) to predict anomalies in the held-out test period (Jul 2023 – Dec 2023).

---

## Standard 7: State Known Limitations Explicitly

**Rule:** Every model card must include at least three known limitations, stated honestly. Do not oversell. Limitations should describe conditions under which the model will fail or perform significantly worse than reported.

**Rationale:** A model card that lists only strengths is not useful to the operations team, the data engineering team implementing the pipeline, or the next analyst maintaining the system. Limitations that are known at training time are predictable failure modes — if they are documented, downstream teams can monitor for them. Limitations that are hidden become production incidents. Honest limitation disclosure is also a signal of analytical maturity: it demonstrates the analyst understands where their model works and where it does not.

**Applied in this portfolio:**

- **Project 1 (SKU Segmentation):** 5 limitations documented: feature scaling dependency, spherical cluster assumption (explains lower silhouette on Clusters 0/3), <90 days history for new SKUs, static cluster assignments (requires quarterly refresh), no soft assignment for boundary SKUs.
- **Project 2 (Demand Forecasting):** 5 limitations: cold-start for new SKUs (<28 days), no promotional/event features (expected 2–3× worse MAPE on promotional weeks), horizon degradation beyond 14 days, non-stationarity with regime changes, no cross-SKU substitution effects.
- **Project 3 (RFM and A/B Test):** 6 limitations: simulated outcome (real test required), no seasonal adjustment, new customer cold-start, single-channel assumption, no network effects modelled, synthetic data density (all customers ordered monthly, underrepresenting natural churn).
- **Project 4 (Warehouse Optimization):** 6 limitations: simplified 2D distance model (no one-way aisles), no real SKU-to-location mapping, session definition approximation, static demand assumption, zone-level (not location-level) LP, sparse affinity pairs.
- **Project 5 (WMS Anomaly Detection):** 6 limitations: no labelled ground truth, synthetic data bias (near-perfect accuracy ~99.3% mean), CUSUM calibration needs real data validation, static contamination assumption, LOF operator baseline at edge of dataset size, missing operator data (~26% filtered out).
