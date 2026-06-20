# Model Card Template
## DHL Data Scientist Portfolio — Standard Format

This template is extracted from the consistent model card format applied across all five DS projects in this portfolio. Every deployed method — clustering, time-series forecasting, statistical tests, optimisation heuristics, and anomaly detection — was documented using this structure before a production recommendation was written.

Use this template for any new model being evaluated for production. Fields marked **[Required]** must be completed before the production recommendation section is filled. Fields marked **[Required if applicable]** apply only to specific model types.

---

## Model Overview

| Field | Value |
|---|---|
| Model name / identifier | [descriptive name, e.g. "LightGBM Global Demand Forecaster v1"] |
| Model type | [clustering / time-series forecasting / ML classification / ML regression / statistical test / combinatorial optimisation / anomaly detection / association rules] |
| Algorithm | [specific algorithm name, e.g. "K-Means (Lloyd's algorithm)", "2-Opt local search", "Two-proportion z-test"] |
| Implementation | [library, function, and version, e.g. "scikit-learn 1.3 `KMeans`", "PuLP 2.7 CBC solver", "scipy.stats `proportions_ztest`"] |
| Version | [v1.0 / date-based, e.g. "2026-06-18"] |
| Production-ready | [Yes / No / Conditional — if Conditional, state the condition] |
| Scope | [what entity is being modelled: SKU / customer / warehouse-day / operator-day / session] |

---

## Training Data

| Field | Value |
|---|---|
| Data source(s) | [table or file name(s)] |
| Time period | [start date – end date] |
| Sample size | [n records and/or n entities, e.g. "1,664 SKUs, 574,509 daily demand rows"] |
| Training / test split | [method: temporal holdout / TimeSeriesSplit n_splits=X / stratified K-fold / leave-one-out] |
| Test period | [explicit date range if temporal split] |
| Key preprocessing | [standardisation method, aggregation level, filtering criteria, exclusion rules] |
| Data quality notes | [any known gaps, imputation rules, or anomalies in the source data] |

**Note on temporal splits:** For any time-series or time-indexed data, temporal splits must be used. Random K-fold cross-validation on time-series data creates data leakage and produces optimistically biased metrics. See ds_methodology_standards.md Standard 6.

---

## Feature List / Input Specification

For ML models — table format:

| Feature Name | Data Type | Engineering Rationale | Source Table |
|---|---|---|---|
| [feature_1] | [float / int / binary / categorical] | [why this feature captures signal relevant to the prediction target] | [table.column] |
| ... | | | |

For non-ML methods:
- State the input data format (e.g. "list of (x, y) location coordinates for each pick stop in a session")
- State any preconditions on the input (e.g. "minimum 28 days of history required; fewer than 28 days falls back to SES")
- State the output format (e.g. "ordered sequence of stop indices; same length as input")

---

## Hyperparameters

| Parameter | Value Used | Selection Method | Rationale |
|---|---|---|---|
| [param_1] | [value] | [grid search / AIC / elbow method / domain convention / fixed] | [why this value was chosen over alternatives] |
| ... | | | |

If a hyperparameter search was performed, state the search space and selection criterion. If a hyperparameter was set by domain convention (e.g. CUSUM k=0.5, h=4.0 are standard logistics industry values), say so explicitly.

---

## Evaluation Metrics

**Required: include at least two independent metrics.** See ds_methodology_standards.md Standard 2.

| Metric | Formula / Definition | Value Achieved | Business Interpretation |
|---|---|---|---|
| [metric_1] | [e.g. "Mean |actual − forecast| / actual × 100"] | [X%] | [what this means in operational terms, not just the number] |
| [metric_2] | | | |
| [metric_3 if applicable] | | | |

Avoid stating metrics without their business interpretation. A silhouette score of 0.5265 is meaningless to an operations manager; "clusters are meaningfully distinct — SKUs assigned to a cluster are more similar to each other than to SKUs in other clusters" is actionable.

---

## Comparison to Baseline

**Required: include the simple baseline. This is non-negotiable.** See ds_methodology_standards.md Standard 1.

| Model | Primary Metric | vs Simple Baseline | vs Previous Best | Interpretation |
|---|---|---|---|---|
| Simple baseline | [value] | — | — | [what the baseline tells us about the problem] |
| [Model being evaluated] | [value] | [+/− Xpp / X%] | [+/− Xpp / X%] | [is this improvement meaningful? in what operational context?] |
| [Other evaluated models] | [value] | [+/− Xpp / X%] | | |

---

## Statistical Significance [Required if applicable]

For hypothesis tests, A/B tests, or any comparison where you are claiming one method is better than another:

| Test | Statistic | p-value | Threshold | Result |
|---|---|---|---|---|
| [e.g. Paired t-test] | [e.g. t=2.016] | [e.g. 0.046] | [e.g. 0.05] | [Significant / Not significant] |
| [e.g. Wilcoxon signed-rank] | | | | |

State whether the test was pre-registered or post-hoc. Post-hoc tests are valid for hypothesis generation; pre-registered tests are required for causal claims. See ds_methodology_standards.md Standard 5.

---

## Robustness Test Results

**Required: at minimum, test sensitivity to one key hyperparameter and one data window variation.** See ds_methodology_standards.md Standard 3.

| Test | Parameter / Condition | Metric | Value | Threshold | Pass / Fail |
|---|---|---|---|---|---|
| [e.g. Seed stability] | [e.g. 20 different random seeds] | [e.g. Mean pairwise ARI] | [e.g. 0.9989] | [e.g. 0.95] | [PASS] |
| [e.g. Window sensitivity] | [e.g. 18-month vs 24-month training] | [e.g. % overlap in flagged anomalies] | [e.g. 70.8%] | [e.g. 60%] | [PASS] |
| [e.g. Hyperparameter sensitivity] | [e.g. contamination 0.03 vs 0.05] | [e.g. Jaccard similarity] | [e.g. 0.600] | [e.g. 0.40] | [PASS] |
| [Feature ablation — if applicable] | [feature removed] | [metric vs full model] | | | |

If any robustness test fails, the model should not be deployed without either (a) addressing the source of instability or (b) explicit documentation of the condition under which it fails and a monitoring trigger to detect that condition in production.

---

## Cost-Benefit Summary

**Required: all four cost rows and at minimum the net annual value row.** See ds_methodology_standards.md Standard 4.

| Item | Value | Notes |
|---|---|---|
| Training time | [X seconds per entity / per batch] | [hardware and batch size context] |
| Inference / scoring time | [X ms per entity] | [relevant for real-time vs batch decision] |
| Monthly compute cost | [$X at $Y/compute-hour, Z retrains/month] | [be explicit about the rate and cadence assumption] |
| Maintenance complexity | [1–5 score] | [breakdown: 1=fully automated, 3=monthly analyst review, 5=weekly bespoke monitoring] |
| Incremental cost vs simple baseline | [$X/month] | [compute delta only; excludes operational overhead] |
| Operational overhead | [$X/year] | [FP investigation time, retraining labour, training time for ops team] |
| Incremental value vs simple baseline | [$X/year or qualitative] | [be specific about assumptions: savings rate, catch rate, unit cost] |
| Net annual value | [$X] | [incremental value minus incremental cost; must be positive for deployment] |

---

## Production Recommendation

**Decision:** [DEPLOY / NOT DEPLOY / HYBRID / VALIDATION ONLY / CONDITIONAL DEPLOY]

**In one sentence:** [plain-language recommendation that a VP could read and act on without a data science background]

**The one number that drives this:** [the single metric or value that most directly justified the decision]

**Key condition for reversal:** [the single most important condition under which this decision should be re-evaluated — be specific, not generic]

---

## Known Limitations

**Required: at least three, stated honestly.** See ds_methodology_standards.md Standard 7.

Do not oversell. Limitations should describe conditions under which performance will be materially worse than reported or where the model should not be used.

- **[Limitation 1]:** [description — be specific about what breaks and when]
- **[Limitation 2]:** [description]
- **[Limitation 3]:** [description]
- **[Limitation 4 — optional]:** [description]

---

## Retraining Cadence

| Trigger | Type | Recommended Response |
|---|---|---|
| Scheduled cadence | Routine | [e.g. Weekly, Monthly, Quarterly — state what "retrain" means: full refit or parameter update only] |
| Performance degradation | Alert-based | [e.g. If live MAPE on A-class SKUs exceeds 80% over 14-day rolling window, trigger immediate investigation] |
| Structural data change | Event-based | [e.g. New warehouse opens: reset rolling baseline for SPC; re-run full K-Means pipeline before next planning cycle] |
| Catalogue change | Event-based | [e.g. >15% of SKUs change ABC class: re-evaluate k and refit before deploying new assignments] |

**Ownership:** [state who makes the retrain decision — Data Science, Operations, automated trigger]

---

## Usage Notes for Downstream Systems

[Brief notes for the Data Engineering team implementing the pipeline or the operations team using the model output:]

- **Input requirements:** [minimum history, required columns, data types]
- **Output format:** [what the model produces — scores, cluster labels, route sequences, flags]
- **Fallback behaviour:** [what happens when input data is missing or below minimum history threshold]
- **Do not use this model for:** [explicit exclusions — what problems this model was not designed to solve]
