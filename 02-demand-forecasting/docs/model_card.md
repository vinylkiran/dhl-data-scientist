# Model Card — DHL Demand Forecasting
**Project:** DS Project 02 — Demand Forecasting  
**Author:** DHL Data Science Portfolio  
**Training period:** 2022-01-01 to 2023-09-30 (638 days)  
**Test period:** 2023-10-01 to 2023-12-31 (92 days)  
**Scope:** 1,664 SKUs across 8 product categories and multiple DHL warehouses

---

## 1. Models Evaluated

### Statistical Baselines
| Model | Description | Hyperparameters |
|---|---|---|
| Naive | Last observed value repeated forward | None |
| Seasonal Naive | Repeat same day-of-week from last full week | Period = 7 |
| SES | Simple Exponential Smoothing | α optimised via MLE |

### ARIMA / SARIMA (12 sample SKUs only)
| Model | Description | Search Space |
|---|---|---|
| ARIMA(p,d,q) | Autoregressive integrated moving average | p∈[0,1,2], d∈[0,1], q∈[0,1,2] — 18 combinations |
| SARIMA(p,d,q)(1,0,1,7) | Seasonal extension, weekly seasonality | Best ARIMA orders + fixed seasonal block |

Selection criterion: Akaike Information Criterion (AIC). Fit method: `innovations_mle` (statsmodels).

### ML Global Models (112 eval SKUs)
| Model | Best Hyperparameters | CV Strategy |
|---|---|---|
| XGBoost | n_estimators=100, max_depth=3, learning_rate=0.05 | TimeSeriesSplit n_splits=3 |
| LightGBM | n_estimators=100, num_leaves=31, learning_rate=0.05 | TimeSeriesSplit n_splits=3 |

**Features (15 total):**
- Lag features: lag_1, lag_7, lag_14, lag_28
- Rolling statistics: rolling_mean_7, rolling_mean_14, rolling_mean_28, rolling_std_7
- Calendar: day_of_week, month, quarter, is_weekend
- SKU metadata: abc_class_encoded (A=2, B=1, C=0), category_encoded, days_since_last_stockout

### Intermittent Demand Methods (36 intermittent SKUs)
| Model | Description |
|---|---|
| Croston | Separate estimation of demand level (α) and inter-demand interval (p); forecast = a/p |
| Croston SBA | Syntetos-Boylan Approximation: multiply Croston forecast by (1 − α/2) |

---

## 2. Evaluation Metrics

| Metric | Formula | Business Interpretation |
|---|---|---|
| MAPE | Mean \|actual − forecast\| / actual × 100 | A 10% MAPE on a 100-unit/day SKU means forecasts are typically off by 10 units/day — for an A-class SKU this could mean ~$1,000/day in misallocated safety stock or stockout risk |
| RMSE | √Mean(actual − forecast)² | Penalises large errors quadratically; especially important for high-value A-class SKUs where a single large miss is costly |
| MAE | Mean \|actual − forecast\| | Directly interpretable as average daily unit error per SKU |
| Bias | Mean(forecast − actual) | Positive = systematic over-forecast (excess inventory cost); negative = systematic under-forecast (stockout risk) |

Zeros are excluded from MAPE denominator to avoid division-by-zero on intermittent series. All metrics computed on the 92-day holdout (Oct–Dec 2023).

---

## 3. Performance Results

### Overall Mean MAPE by Model

| Model | A-class MAPE | B-class MAPE | C-class MAPE | Overall MAPE |
|---|---|---|---|---|
| LightGBM | 66.5% | 55.5% | 50.2% | 57.4% |
| XGBoost | 68.5% | 55.5% | 48.6% | 57.5% |
| SES | 67.0% | 56.9% | 75.5% | 66.5% |
| Naive | 85.4% | 71.4% | 83.0% | 79.9% |
| ARIMA/SARIMA | 86.3% | 78.2% | 76.4% | 80.3% |
| Seasonal Naive | 93.1% | 77.3% | 80.9% | 83.8% |

### Breakdown by Demand Pattern

| Model | Stable MAPE | Intermittent MAPE |
|---|---|---|
| LightGBM | 66.5% | 54.9% |
| XGBoost | 68.5% | 54.6% |
| SES | 67.0% | 59.3% |
| Croston | — | 70.2% |
| Naive | 85.4% | 72.9% |

### ARIMA/SARIMA (12-SKU sample)

| Model | Mean MAPE | Mean RMSE | Mean MAE | Avg LB p-value |
|---|---|---|---|---|
| ARIMA | 80.3% | 49.0 | 40.0 | 0.48 |
| SARIMA(1,0,1,7) | 80.6% | 48.8 | 39.9 | 0.55 |

Most SKUs selected ARIMA(0,1,1) — an IMA(1,1) / ETS(A,N,N) structure, consistent with SES being competitive. All Ljung-Box tests passed (p>0.05) except 2 SKUs where residuals showed mild autocorrelation.

### Intermittent Demand Results (36 SKUs with >40% zero days)

| Method | MAPE | RMSE | MAE | Bias |
|---|---|---|---|---|
| Croston | 70.2% | 12.72 | 10.71 | −1.39 |
| Croston SBA | 70.3% | 12.77 | 10.68 | −1.88 |
| SES | 71.4% | 12.67 | 10.71 | −1.40 |
| Naive | 88.6% | 15.18 | 11.36 | −1.88 |
| Seasonal Naive | 94.0% | 16.26 | 12.03 | −2.10 |

**Finding:** Croston achieves lowest MAPE on intermittent SKUs and reduces negative bias from −1.88 (Naive) to −1.39, indicating less systematic under-forecasting. The SBA variant introduces slightly more bias-correction than optimal on this dataset.

---

## 4. Statistical Significance

**Test:** LightGBM vs. SES (best baseline) on per-SKU MAPE differences (n=112 SKUs).

| Test | Statistic | p-value | Result |
|---|---|---|---|
| Paired t-test | t = 2.016 | p = 0.046 | Significant at p<0.05 |
| Wilcoxon signed-rank | W = 2326 | p = 0.015 | Significant at p<0.05 |

**Mean improvement:** LightGBM beats SES by 2.97 percentage points (62.2% → 59.2% MAPE). While statistically significant, the practical improvement is modest — the primary case for ML is its scalability (one global model vs. 1,664 individual SES fits) and consistent advantage across all ABC classes.

---

## 5. Cost-Benefit Summary

| Model | Monthly Compute Cost | Maintenance Complexity | MAPE Improvement vs Naive |
|---|---|---|---|
| Naive | $0.0002 | 2.0/5 | — |
| SES | $0.0092 | 2.5/5 | 13.5pp |
| LightGBM | $0.0022 | 3.0/5 | 22.5pp |
| XGBoost | $0.0033 | 3.0/5 | 22.4pp |
| ARIMA | $1.48 | 3.0/5 | −0.4pp (worse than Naive!) |
| SARIMA | $2.77 | 3.75/5 | −0.4pp |

**Key finding — diminishing returns:** ARIMA/SARIMA costs 670–1,250× more than LightGBM while producing worse MAPE. The optimal Pareto frontier runs: Naive → SES → LightGBM. ARIMA is strictly dominated.

MAPE improvement per dollar spent (vs Naive):
- LightGBM: 11,265 pp/$ (highest efficiency)
- Croston: 6,106 pp/$
- XGBoost: 7,223 pp/$
- SES: 1,496 pp/$
- ARIMA: −0.24 pp/$ (negative — costs more, performs worse)

---

## 6. Final Production Recommendation

| Segment | Model | MAPE | Monthly Cost |
|---|---|---|---|
| A-class (n=341) | LightGBM | 66.5% | ~$0.0005 (proportional share) |
| B-class (n=504) | LightGBM | 55.5% | ~$0.0007 |
| C-class (n=819) | LightGBM | 57.4% | ~$0.0011 |
| Intermittent (n=36, B/C class) | Croston/SBA override | 70.2% | ~$0.0002 |
| **Total** | | | **$0.0025/month** |

---

## 7. Known Limitations

1. **Cold-start problem:** New SKUs with <28 days of history cannot generate lag and rolling features. These should fall back to SES or category-mean until sufficient history accumulates (recommend 60-day minimum).

2. **No promotional / event features:** The model has no knowledge of promotional calendars, holidays, or DHL sales events. Demand spikes around events will appear as residuals and will not be forecasted — expected MAPE on promotional weeks is likely 2–3× higher than reported.

3. **Horizon degradation:** All models were evaluated on a 92-day horizon but are used as flat-level forecasts. Point forecast quality degrades with horizon length. For 30+ day horizons, rolling re-forecasting (re-running the model monthly) is essential. Do not extrapolate beyond 14 days without re-fitting.

4. **Non-stationarity with regime changes:** All 12 sample SKUs were stationary (ADF p<0.001), which is consistent with the demand data structure (aggregated across warehouses dampens individual trends). However, supply chain disruptions (e.g., COVID-level events) will violate the lag-feature assumptions.

5. **No cross-SKU demand linkage:** The global model captures cross-SKU patterns only through categorical features. True substitution effects (when SKU A runs out, demand for SKU B rises) are not modelled.

---

## 8. Retraining Cadence

| Model | Recommended Cadence | Rationale |
|---|---|---|
| LightGBM (global) | Weekly | Low marginal retrain cost ($0.0022/month for all SKUs); captures recent demand shifts |
| SES | Weekly | α re-estimation on rolling window; trivial compute |
| Croston/SBA | Monthly | Intermittent series change slowly; monthly sufficient |
| ARIMA (if used) | Monthly | High retrain cost; use only for specific high-value SKUs |

**Monitoring trigger:** If mean MAPE across A-class SKUs exceeds 80% on any rolling 14-day window, trigger immediate investigation and potential model refresh. Log per-SKU bias weekly — persistent bias >20% in either direction signals a structural demand change (new warehouse, product discontinuation, etc.).
