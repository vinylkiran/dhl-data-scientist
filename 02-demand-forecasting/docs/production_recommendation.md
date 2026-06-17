# Production Recommendation — Demand Forecasting

## Decision: LightGBM (Global) with Croston/SBA Override for Intermittent SKUs

---

### Per-class recommendation

| Class | Model | MAPE | Monthly Compute Cost |
|---|---|---|---|
| A-class (n=341 SKUs) | LightGBM | 66.5% | $0.0005/month (proportional) |
| B-class (n=504 SKUs) | LightGBM | 55.5% | $0.0007/month (proportional) |
| C-class (n=819 SKUs) | LightGBM | 57.4% | $0.0011/month (proportional) |
| Intermittent SKUs (n=36, B/C-class) | Croston/SBA override | 70.2% | $0.0002/month |

---

### Total monthly compute cost: $0.0025

vs. All-Naive counterfactual: $0.0002  
Net incremental cost of recommended approach: **$0.0023/month**

The recommended approach costs less than $0.01/month. The dominant cost drivers are analyst time for monitoring and retraining pipeline maintenance — not compute.

---

### The numbers that drove this

- **Best overall MAPE:** 57.4% — LightGBM (global model, n=112 eval SKUs)
- **ML vs. best baseline significance:** p=0.046 (paired t-test), p=0.015 (Wilcoxon) — **significant at p<0.05**
- **Mean improvement over SES:** 2.97 percentage points (62.2% → 59.2%)
- **Intermittent SKUs:** 32.1% of eval catalogue — handled by Croston, reducing bias from −1.88 to −1.39
- **Cost advantage of ML over ARIMA:** LightGBM costs $0.0022/month vs. $1.48/month for ARIMA — 670× cheaper, with 22.9pp better MAPE
- **Pareto-dominant models:** Naive → SES → LightGBM. All other models are strictly dominated on the cost-accuracy frontier.

---

### What would change this

1. **If compute costs drop below $0.01/hr:** ARIMA becomes viable for B-class SKUs where longer seasonal cycles exist (quarterly promotion patterns not captured by 28-day lag features). At $0.01/hr, ARIMA's $1.48/month becomes $0.15/month — still 70× more than LightGBM, but potentially justifiable for a targeted set of high-volatility B-class SKUs.

2. **If a high-value A-class SKU causes a stockout costing >$50K:** Re-evaluate SARIMA with promotional calendar features for that specific SKU. A single SKU generating $50K+ stockout costs can absorb SARIMA's $2.77/month retrain cost. The decision becomes per-SKU ROI rather than catalogue-wide average.

---

### Not deployed, and why

| Model | Reason Not Deployed |
|---|---|
| SARIMA | 3.75/5 maintenance complexity; $2.77/month; MAPE 80.3% — worse than LightGBM by 22.9pp |
| ARIMA | $1.48/month; MAPE 80.3% — worse than LightGBM by 22.9pp; accuracy no better than SES |
| Seasonal Naive | Worst MAPE (83.8%) across all classes; outperformed by every other method |
| Naive | Retained as monitoring benchmark and emergency fallback only; not production forecaster |

---

### Implementation notes

- **Retraining:** LightGBM global model retrains weekly (estimated 30 seconds total for all 1,664 SKUs). Croston parameters update monthly.
- **Feature requirements:** 28 days of history per SKU minimum to generate all lag features. New SKUs fall back to SES for the first 60 days.
- **Monitoring:** Alert if mean A-class MAPE exceeds 80% over any rolling 14-day window.
- **Intermittent detection:** SKUs with >40% zero-demand days in the trailing 90 days are routed to Croston/SBA automatically at scoring time.
