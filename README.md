# DHL Data Scientist Portfolio

Production-grade data science applied to a synthetic DHL supply chain dataset. This portfolio demonstrates what separates a data scientist who can build models from one who can deploy them: every project compares the recommended method against a simpler baseline on cost and accuracy, validates robustness before writing a production recommendation, and quantifies the business value in specific dollar terms.

This is one of three role-specific portfolios built from the same underlying DHL supply chain problem. The other two are [`dhl-business-analyst`](https://github.com/vinylkiran/dhl-business-analyst) (SQL exploration, dashboards, stakeholder reporting) and [`dhl-data-engineer`](https://github.com/vinylkiran/dhl-data-engineer) (ETL pipelines, schema design, data quality frameworks). All three operate on the same synthetic dataset, demonstrating versatility across the full analytics-to-production stack.

---

## Projects

| # | Project | Description | Production Decision |
|---|---------|-------------|---------------------|
| 1 | [SKU Segmentation](./01-sku-segmentation/) | K-Means clustering (k=4, 9 demand-pattern features) on 1,664 SKUs to replace rule-based ABC/XYZ. Silhouette=0.5265, seed-stability ARI=0.9989 across 20 seeds. K-Means and XYZ have ARI=0.002 — entirely independent signals. | **K-Means deployed as primary; ABC retained as audit overlay.** Net annual value: $3,904,833. |
| 2 | [Demand Forecasting](./02-demand-forecasting/) | 8 methods evaluated (Naive, Seasonal Naive, SES, ARIMA, SARIMA, XGBoost, LightGBM, Croston, SBA) across 1,664 SKUs. ARIMA costs 670× more than LightGBM with worse MAPE (80.3% vs 57.4%). | **LightGBM (all ABC classes) + Croston override for intermittent SKUs.** Total compute: $0.0025/month. Significant at p=0.046 vs SES. |
| 3 | [RFM and A/B Test](./03-rfm-ab-test/) | RFM quintile scoring on 398 customers; stratified A/B test on At Risk segment (n=95, 23.9% of scored base). Pre-registered design, two-proportion z-test, bootstrap stability validation. | **Scale to full At Risk population.** p=0.014, 15.2pp lift, $482K incremental revenue at $760 campaign cost (ROI ~63,407%). |
| 4 | [Warehouse Optimization](./04-warehouse-optimization/) | Pick-path sequencing (Naive → NN → 2-Opt → Exact TSP), LP joint slotting, co-occurrence affinity analysis with permutation validation, temporal stability for cadence. Exact TSP rejected: NP-hard, $450/month for 1.75% incremental gain. | **2-Opt + LP joint slotting, monthly re-optimisation.** Net annual benefit: $2,434,136. Payback: ~1 month on $200K implementation. |
| 5 | [WMS Anomaly Detection](./05-wms-anomaly-detection/) | SPC (Western Electric + CUSUM) vs Isolation Forest vs LOF at warehouse and operator level. Hybrid is $7,428/year cheaper than SPC-only due to lower false-positive investigation burden. | **Hybrid: SPC + CUSUM (daily) + Isolation Forest (weekly) + LOF operator (weekly).** Total: $110/month. |
| 6 | [DS Artifacts](./06-ds-artifacts/) | Master model registry (28 methods across 5 projects), production decisions summary with cost-discipline thread, 7-standard methodology document, reusable model card template, HTML executive summary with DHL branding. | — |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML / Gradient Boosting | scikit-learn, XGBoost, LightGBM |
| Statistical / Time-Series | statsmodels (ARIMA, SARIMA, SES, Croston), scipy |
| Anomaly Detection | sklearn IsolationForest, LocalOutlierFactor |
| Optimisation | PuLP (LP with CBC solver), itertools (exact TSP enumeration), mlxtend |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, Plotly |
| Experiment Design | scipy.stats (z-test, t-test, Wilcoxon), bootstrap resampling |
| Database | DuckDB (SQL exploration across all projects) |

---

## Synthetic Dataset

All five projects share the same dataset, generated to reflect realistic DHL supply chain structure:

| Table | Rows | Description |
|-------|------|-------------|
| `sku_master.csv` | 2,000 SKUs | SKU catalogue (1,664 with demand activity; 336 with no demand in 2022–2023) |
| `daily_demand.csv` | 574,509 | Daily demand per SKU × warehouse, Jan 2022–Dec 2023 |
| `outbound_orders.csv` | 68,941 | Order-level transactions across 500 customers |
| `customers.csv` | 500 | Customer metadata (type, region, SLA, contract tier) |
| `wms_tasks.csv` | 219,000 | WMS pick/putaway tasks with accuracy flags and duration |
| `warehouse_locations.csv` | 2,640 | Storage locations across 3 warehouses (Pick Face / Reserve / Bulk zones) |

3 warehouses (DHL-WH-NJ01, DHL-WH-IL02, DHL-WH-TX03), 8 product categories, 730 days (Jan 2022–Dec 2023), 500 customers, 60 operators.

---

## Consistent Methodology

Every project follows the same five-stage evaluation pipeline — documented in full in [`06-ds-artifacts/standards/ds_methodology_standards.md`](./06-ds-artifacts/standards/ds_methodology_standards.md):

1. **Baseline first** — establish and measure the naive/rule-based method before evaluating anything complex
2. **Model building** — evaluate all candidates with at least two independent metrics; TimeSeriesSplit for time-series (never random K-fold)
3. **Robustness testing** — re-run with different seeds, time windows, and hyperparameter values
4. **Cost-benefit analysis** — quantify compute cost, maintenance, FP investigation overhead, and incremental value vs baseline
5. **Production decision** — one clear decision, one key number, explicit conditions for reversal

In three of five projects, the complex ML method was chosen because the evidence supported it. In two cases (ARIMA for forecasting, Exact TSP for routing), a theoretically "better" method was rejected because it was strictly dominated on both cost and accuracy. This is what production-grade data science looks like.

---

## Portfolio-Wide Key Numbers

| Project | Winning Method | Headline Metric | Annual Value |
|---------|----------------|-----------------|--------------|
| SKU Segmentation | K-Means (k=4) | Silhouette=0.5265; Seed-ARI=0.9989 | $3,904,833 |
| Demand Forecasting | LightGBM + Croston | Overall MAPE=57.4%; p=0.046 vs SES | $0.0025/month compute |
| RFM and A/B Test | Pre-registered z-test | p=0.014; 15.2pp lift | $481,892 net (ROI ~63,407%) |
| Warehouse Optimization | 2-Opt + LP Slotting | 49.8% distance reduction | $2,434,136 |
| WMS Anomaly Detection | Hybrid SPC + IF | 148 additional anomaly signals vs SPC-only | $7,428 saving + $21K detection value |

---

## Portfolio Context

Three portfolios. One dataset. Three different roles:

- **[`dhl-business-analyst`](https://github.com/vinylkiran/dhl-business-analyst)** — SQL exploration, KPI dashboards (Tableau-ready), business requirement documents, stakeholder narratives. The BA layer: turning data into business language.
- **[`dhl-data-engineer`](https://github.com/vinylkiran/dhl-data-engineer)** — ETL pipelines, schema design, data quality frameworks, serving layers, pipeline monitoring. The DE layer: making data reliable and accessible.
- **[`dhl-data-scientist`](https://github.com/vinylkiran/dhl-data-scientist)** — This portfolio. Model development, evaluation, robustness testing, cost-benefit analysis, production recommendations. The DS layer: extracting decisions from data.

Each portfolio is self-contained. Together they show what it looks like to operate across the full analytics stack from a single, well-understood business problem.
