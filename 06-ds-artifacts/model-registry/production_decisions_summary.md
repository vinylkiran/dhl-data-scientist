# Production Decisions Summary
## DHL Data Scientist Portfolio — All Projects

This document summarises every production decision made across the five core data science projects in this portfolio. It is intended to show the evaluative thread that runs through all five projects: every decision was made after comparing the recommended method against a simpler alternative on both cost and accuracy. The decision was sometimes in favour of the complex method, sometimes in favour of the simple one, and sometimes a hybrid. What is consistent is the process, not the outcome.

---

## The Cost-Discipline Thread

Production-grade data science is not about selecting the most sophisticated model. It is about selecting the model that maximises value relative to cost, where cost includes compute, maintenance overhead, explainability effort, and operational risk. The simplest method that achieves acceptable performance at an acceptable cost is the correct production choice — and the burden of proof sits with the more complex method to demonstrate incremental value.

Across the five projects in this portfolio, this principle played out differently in each context. In two projects (SKU Segmentation and Demand Forecasting), a complex ML method was justified because it demonstrably outperformed the simple baseline and the incremental cost was negligible. In one project (RFM and A/B Test), the question was not simple vs complex but rather correct vs incorrect statistical design — the "simple" two-proportion z-test was the right method, and more complex approaches would have added noise without adding validity. In one project (Warehouse Optimization), the picture was nuanced: a moderate-complexity heuristic (2-Opt) was chosen over both the naive baseline and the theoretically optimal but computationally intractable exact method, while a custom co-occurrence method was chosen over a standard library (Apriori) because the data structure made the library inapplicable. In the final project (WMS Anomaly Detection), the conclusion was explicitly that the ML method should complement SPC rather than replace it — and the financial case showed that SPC alone was more expensive due to false-positive investigation burden, making the hybrid objectively cheaper.

This is what distinguishes production-grade data science from academic modelling: the frame of reference is always cost and operational value, never model sophistication as an end in itself.

---

## Project 1 — SKU Segmentation

**What was tested:** K-Means clustering (k=4, 9 demand-pattern features, z-scored) compared against rule-based ABC (revenue-ranked) and XYZ (CV-based) classifications on a catalogue of 1,664 SKUs.

**Production decision:** DEPLOY K-MEANS AS PRIMARY; RETAIN ABC AS AUDIT OVERLAY

**Simple vs complex:** Complex method justified. K-Means agrees strongly with ABC on the revenue dimension (ARI=0.885) but is almost completely independent of XYZ (ARI=0.002). This means K-Means captures demand-pattern information (frequency, trend, seasonality, order size) that XYZ entirely misses — it is not replacing ABC/XYZ with something noisier; it is replacing XYZ with something more informative while preserving ABC's financial signal. The incremental compute cost is $0.012/year ($0.013 − $0.001). The incremental value is $3,904,833/year from correctly reclassifying 70 over-rated A-class SKUs whose demand patterns do not match their revenue-implied velocity.

**The financial logic:** K-Means identifies 70 A-class SKUs (high revenue, but actually Low-Velocity in demand pattern) that are currently managed with fast-cycle replenishment misaligned to their true rhythm. These SKUs carry excess safety stock (capital lock-up) and face stockout risk when a genuine spike occurs (because replenishment timing is wrong). Correcting 70 SKUs at a conservative 10% catch rate on estimated stockout risk captures $3,904,633 in annual value. The compute delta of $0.012/year is irrelevant to this calculation.

**Robustness confirmation:** 20-seed stability test produced mean pairwise ARI=0.9989, min pairwise ARI=0.9951, and 100% of seed pairs with ARI >0.95. Feature ablation confirmed all 9 features are meaningful (min ARI=0.965 when demand_frequency removed). Outlier sensitivity confirmed ARI=0.983 after removing top 1% of SKUs. The solution is not fragile.

**Key numbers:**
- Net annual value: **+$3,904,833**
- Silhouette score: **0.5265** (threshold 0.50)
- Seed-stability ARI: **0.9989** (threshold 0.95)
- Compute cost increment: **$0.012/year**

---

## Project 2 — Demand Forecasting

**What was tested:** Eight forecasting methods — Naive, Seasonal Naive, SES, ARIMA, SARIMA, XGBoost, LightGBM, Croston/SBA — evaluated on 1,664 SKUs across three ABC classes and two demand patterns (stable and intermittent). Training: Jan 2022 – Sep 2023. Test: Oct 2023 – Dec 2023 (92-day holdout).

**Production decision:** LIGHTGBM FOR ALL ABC CLASSES; CROSTON OVERRIDE FOR INTERMITTENT SKUs

**Simple vs complex — where complex was justified:** LightGBM achieves the best MAPE (57.4% overall) and is statistically significantly better than SES, the best simple baseline (p=0.046 paired t-test, p=0.015 Wilcoxon). The mean improvement is 2.97 percentage points (62.2% → 59.2%). Monthly cost: $0.0022 — not $22, not $2.20, $0.0022. At that cost, the statistical significance of the improvement is sufficient to justify deployment.

**Simple vs complex — where complexity was penalised:** ARIMA costs $1.48/month vs LightGBM's $0.0022/month (670× more expensive) while producing a MAPE of 80.3% — worse than Naive (79.9%) and far worse than LightGBM (57.4%). SARIMA costs $2.77/month (1,250× more expensive than LightGBM) with MAPE 80.6% — the worst of all evaluated methods. Both were rejected not on philosophical grounds but because they are strictly dominated: higher cost, worse accuracy. The optimal Pareto frontier is: Naive → SES → LightGBM. Every other method falls inside this frontier.

**ARIMA result explained:** Most of the 12 ARIMA sample SKUs selected the (0,1,1) order — mathematically equivalent to an IMA(1,1) or ETS(A,N,N) model, which is equivalent to SES. ARIMA's complex machinery converged to the same structure as the simple baseline, at 670× the cost.

**Intermittent demand:** 36 SKUs (2.2% of catalogue) with >40% zero-demand days are handled by Croston, which reduces negative bias from −1.88 (Naive) to −1.39 and improves MAPE from 88.6% (Naive) to 70.2%. Croston's purpose-built structure for sparse demand series justifies its separate deployment. The Croston SBA variant introduced slightly more bias-correction than optimal on this dataset and was not deployed.

**Key numbers:**
- LightGBM total monthly cost: **$0.0025** (all 1,664 SKUs)
- ARIMA monthly cost: **$1.48** (12-SKU sample) — 670× LightGBM
- SARIMA monthly cost: **$2.77** — 1,250× LightGBM
- LightGBM vs SES significance: **p=0.046** (paired t-test)
- LightGBM MAPE efficiency: **11,265 percentage-points per dollar spent**
- Croston bias reduction: **−1.88 → −1.39** on intermittent SKUs

---

## Project 3 — RFM Segmentation and A/B Test

**What was tested:** RFM quintile scoring to identify the At Risk customer segment (n=95, 23.9% of scored customers), followed by a stratified A/B test of a retention campaign ($8 per customer). Statistical test: two-proportion z-test (pre-registered before outcome observation). Validation methods: bootstrap stability (50 iterations, 70% subsample) and power analysis.

**Production decision:** SCALE TO FULL AT RISK POPULATION

**Simple vs complex — the right framing:** This project's key methodological decision was not which model to pick but how to design the test correctly. The "complex" choices were statistical rigour choices: pre-registration, stratified randomisation, power analysis before running the test, and pre-specified guardrail metrics. The "simple" alternative would have been to run an uncontrolled before/after comparison or to inspect results early and stop when the number looked good. These rigor choices add no compute cost; they prevent false positives, which are genuinely expensive (scaling a non-effective $760 campaign that displaces attention from real initiatives costs more than the $760).

**The economic decision:** The campaign costs $8 per customer. Average order value for At Risk customers is $33,381. The break-even lift is 0.024 percentage points — essentially zero. Any statistically significant positive result justifies rollout. The test delivered a 15.2pp lift (p=0.014), with 95% CI [3.5pp, 27.0pp] excluding zero. At the observed lift, incremental revenue is $482,652 against a $760 campaign cost: ROI ≈ 63,407%. The decision criterion was entirely whether the lift was real — not whether the ROI was good (which it obviously is at any lift above zero).

**Power analysis result:** The At Risk population (n=95) is too small to detect the pre-specified 5pp MDE (would require n=766 per group). The test was redesigned for n=47 per group with a detectable MDE of 15.1pp at 80% power. The cost model confirmed that even 15.1pp is economically meaningful — the test was appropriately scoped to the available population rather than artificially powered to detect an effect the population couldn't sustain.

**Key numbers:**
- Absolute lift: **15.2pp** (treatment 97.8% vs control 82.6%)
- p-value: **0.014** (threshold 0.05)
- 95% CI: **[3.5pp, 27.0pp]** (excludes zero)
- Campaign cost (95 customers): **$760** ($8 per customer)
- Incremental revenue: **$482,652**
- Realised ROI: **~63,407%**
- At Risk segment: **95 customers** (23.9% of scored base)
- Avg order value (At Risk): **$33,381**
- Post-hoc power: **0.772**

---

## Project 4 — Warehouse Optimization

**What was tested:** Four path-planning methods (Naive, Nearest-Neighbor, 2-Opt, Exact TSP), three slotting assignment methods (Current WMS default, Greedy heuristic, LP Joint Optimization), co-occurrence affinity detection via manual counting vs standard Apriori, and temporal stability analysis for cadence determination. Data: 219,000 WMS task records, 2,766 filtered pick sessions.

**Production decision:** 2-OPT + LP JOINT SLOTTING + MONTHLY RE-OPTIMISATION

**Path planning — avoiding the complexity trap at both ends:** The naive WMS order (mean distance 112.5 units) is clearly suboptimal. The exact TSP is theoretically optimal but NP-hard at scale: 15 stops = 15! ≈ 1.3 trillion permutations, monthly cost at full deployment $450. 2-Opt achieves 49.8% distance reduction vs naive (mean distance 56.5 units) at a monthly cost of $1.55, with only a 1.75% gap from exact-optimal. The ROI calculation for Exact TSP: $450 − $1.55 = $448 incremental monthly cost for 1.75% incremental improvement on a $2.45M saving. The improvement is worth $43K/year; the incremental cost is $5,376/year — positive but extremely marginal, and the operational complexity of exact enumeration at scale invalidates it.

**Slotting — LP justified, Greedy rejected:** The Greedy heuristic achieves only 0.4% improvement over current slotting (60.13 vs 60.36 mean travel distance). LP Joint Optimization achieves 1.5% (59.44 vs 60.36). At <1 second solve time (CBC solver), LP adds no meaningful latency cost. The joint objective — minimising travel distance while co-locating high-affinity pairs — is not achievable with a greedy rank-order approach. LP is justified.

**Affinity detection — when the standard library doesn't fit:** Standard Apriori requires min_support ≥ 1% — but the maximum pair co-occurrence in this catalogue is 4 sessions out of 2,766 (0.14% support). The library produces zero pairs. Manual co-occurrence counting with permutation validation (100 iterations per pair) achieves identical results with full transparency. All 50 top pairs were validated at p<0.05 (all p=0.00), confirming that the observed lifts are not random. Top pair: AUT-001772 ↔ FSH-000914, lift=71.8.

**Key numbers:**
- 2-Opt distance reduction vs naive: **49.8%**
- 2-Opt gap from exact optimal: **1.75%**
- Annual path-planning saving: **$2,452,887**
- Annual slotting saving (LP): **$75,064**
- Annual total saving: **$2,527,951**
- Annual total cost: **$93,815** (compute $1,017 + re-slotting labour $92,798)
- Net annual benefit: **$2,434,136**
- Implementation cost: **$200,000**
- Payback period: **~1 month**
- SKU tier change rate: **74.4%/quarter** → monthly re-optimisation required

---

## Project 5 — WMS Anomaly Detection

**What was tested:** Statistical Process Control (4 Western Electric rules + CUSUM) vs Isolation Forest vs Local Outlier Factor, at both warehouse-day and operator-day granularity, evaluated on 2,190 warehouse-day KPI records and 3,662 operator-day records across Jan 2022 – Dec 2023.

**Production decision:** HYBRID — SPC + CUSUM (DAILY, LAYER 1) + ISOLATION FOREST (WEEKLY, LAYER 2) + LOF OPERATOR (WEEKLY, LAYER 3)

**Simple vs complex — both are deployed, for different reasons:** SPC is deployed because it is universally understood by operations teams, zero explanation overhead, and produces actionable daily outputs without data science involvement. IF is deployed not because SPC failed but because IF catches a different signal: 148 warehouse-days over 2 years were flagged by IF but not SPC — these are multi-feature anomalies where no single metric individually breaches its threshold but the combination is statistically unusual. Even at 20% true-positive rate, that is 30 real incidents caught earlier over 2 years.

**The financial inversion:** SPC-only costs $8,748/year in FP investigation time. IF-only (or hybrid) costs $1,320/year. The hybrid is $7,428 cheaper per year than SPC-only — not because SPC is bad but because IF's lower FP rate reduces supervisor investigation burden. The "simpler" method (SPC alone) is more expensive in total operational cost than the hybrid. This is an important lesson: the cost comparison should include operational overhead, not just compute.

**LOF at warehouse level rejected:** IF and LOF agree on only 15.2% of warehouse-day flags (Jaccard=0.152). Deploying both at warehouse level doubles alert volume without proportionate signal gain. LOF is re-deployed at operator level, where its local density approach is well-suited to identifying individual performance outliers (unusual combinations of task rate, accuracy, and duration) within a peer cohort.

**Sigma threshold decision:** At 2σ, daily flags increase by 600–673% vs 3σ. At 3σ: 11–12 flags per warehouse over 2 years (manageable). 3σ is the correct production threshold for daily operations.

**Key numbers:**
- SPC total anomaly records (2-year test window): **286**
- CUSUM additional drift events: **15**
- IF anomaly flags (contamination=0.05): **110 warehouse-days**
- IF-unique flags (not captured by SPC): **148 days over 2 years**
- LOF operator-day anomalies: **139** (~6 flags/month across 60 operators)
- Annual early-detection value (3 WH × 4 incidents × $1,750): **$21,000**
- SPC-only annual cost: **$8,748**
- Hybrid annual cost: **$1,320**
- Annual saving from hybrid vs SPC-only: **$7,428**
- Total monthly monitoring cost: **$110**

---

## Summary Table — All Projects

| Project | Simple Baseline | Complex Method | Decision | Primary Reason |
|---|---|---|---|---|
| SKU Segmentation | Rule-based ABC/XYZ ($0.001/yr compute) | K-Means k=4, 9 features ($0.013/yr) | **Complex justified** | K-Means adds demand-pattern signal XYZ cannot capture (ARI vs XYZ=0.002); net annual value=$3,904,833; seed-stability ARI=0.9989 |
| Demand Forecasting | Naive ($0.0002/mo) / SES ($0.0092/mo) | LightGBM ($0.0022/mo) | **Complex justified; ARIMA/SARIMA rejected** | LightGBM beats SES at p=0.046 at lower cost; ARIMA is 670× more expensive with worse MAPE (80.3% vs 57.4%) |
| RFM and A/B Test | Uncontrolled before/after | Two-proportion z-test (pre-registered, stratified) | **Statistical rigour justified** | ROI irrelevant at AOV $33K; p=0.014 confirms lift is real; pre-registration prevents false positive scale decision |
| Warehouse Optimization | Naive WMS path / WMS default slotting | 2-Opt + LP Joint Slotting | **Complex justified; Exact TSP rejected** | 2-Opt achieves 49.8% distance reduction at 1.75% gap from exact; Exact TSP $450/month vs $1.55 for 1.75% incremental gain |
| WMS Anomaly Detection | SPC-only ($8,748/yr total cost) | Hybrid SPC + Isolation Forest ($1,320/yr) | **Hybrid; both deployed for different signals** | Hybrid is $7,428 CHEAPER than SPC-only due to lower FP investigation burden; IF catches 148 additional anomalies SPC misses |

---

## The Pattern Across Projects

Looking across all five decisions, a consistent principle emerges: the evaluation is always bilateral. A complex method must prove it adds value over the simple baseline, and a simple method must be shown to be sufficient before the complex alternative is rejected. The evidence comes from actual numbers — p-values, cost ratios, financial estimates — not assertions.

In three of five projects, a more complex method was chosen because the evidence supported it. In every case, the justification was specific and quantified: $3.9M net value (Project 1), p=0.046 statistical significance (Project 2), $2.43M net annual benefit (Project 4). In the remaining two projects, the decision was nuanced: statistical rigour tools (Project 3) and a cost-aware hybrid (Project 5). Neither of these was a default choice — both were reached through the same evaluation process.

The method is not to start from a preference for simplicity or complexity. It is to start from the data.
