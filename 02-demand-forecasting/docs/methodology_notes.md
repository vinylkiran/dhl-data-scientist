# Methodology Notes — DHL Demand Forecasting
**Project:** DS Project 02 — Demand Forecasting  
**Purpose:** Technical rationale for all analytical decisions made in this project

---

## 1. Why Stationarity Testing Matters for ARIMA

ARIMA models assume the time series is stationary — that is, its statistical properties (mean, variance, autocorrelation structure) do not change over time. Applying ARIMA to a non-stationary series violates this assumption: parameter estimates become unreliable, confidence intervals are too narrow, and long-horizon forecasts diverge.

**The ADF (Augmented Dickey-Fuller) test** checks the null hypothesis that a unit root is present (non-stationary). A p-value < 0.05 allows us to reject the null — the series is stationary, and we set d=0. A p-value ≥ 0.05 indicates a unit root; we set d=1 (first difference).

In this project: all 12 sample SKUs were stationary at p<0.001. This is consistent with demand data aggregated across multiple warehouses — individual-warehouse intermittent demand can show structural breaks, but warehouse-aggregated demand tends to be mean-reverting around a stable level. The practical implication: d=0 is appropriate, and ARIMA(0,0,1) or ARIMA(1,0,1) structures are reasonable starting points.

**Why ADF over KPSS?** ADF tests for a unit root (null = non-stationary), while KPSS tests for stationarity (null = stationary). ADF is more widely used for ARIMA order selection. In practice, running both and checking for disagreement is safer, but given our stationarity results were clear (p<0.001), ADF alone was sufficient.

---

## 2. Why Time-Series Cross-Validation Over Random k-Fold

Standard k-fold cross-validation randomly shuffles data and validates on held-out folds. For time series, this is fundamentally wrong because:

1. **Data leakage:** Future observations end up in the training set. A model trained on, say, December demand validated on August demand has "seen the future." Error estimates will be optimistically biased.
2. **Temporal structure destroyed:** Autocorrelation and seasonality — the core signal in demand series — are eliminated when observations are shuffled.

**TimeSeriesSplit** (sklearn) respects temporal order: each fold trains on data up to time t and validates on data from t+1 to t+k. We used n_splits=3 on the Jan 2022 – Sep 2023 training period, creating three expanding training windows with sequential validation periods.

The penalty for using random k-fold on time series is not just philosophical — in our ML models, random k-fold would have inflated CV RMSE estimates and potentially led to different hyperparameter selections (overfitting to temporal patterns that are only visible with future data).

---

## 3. Croston's Method: Intuition and Implementation

Standard forecasting methods (SES, ARIMA) treat demand as a continuous process. For intermittent demand series (36 SKUs, or 32% of our eval set, had >40% zero-demand days), this creates two problems:

1. **Forecasts trend toward zero** because the running average is dominated by zero periods.
2. **Bias is negative** — the model chronically under-forecasts non-zero demand events.

**Croston's insight:** treat the demand process as two independent sub-processes:
- `a` = the **level of non-zero demand** (how much when demand occurs)
- `p` = the **inter-demand interval** (how often demand occurs)

Each sub-process is updated independently using exponential smoothing (α) only when demand is non-zero. The forecast is `a / p` — demand level divided by average interval.

**Implementation (from scratch):**
```python
def croston(demand, alpha=0.1):
    for i, d in enumerate(demand):
        if d > 0:
            a = alpha * d + (1 - alpha) * a      # update level
            p = alpha * q + (1 - alpha) * p      # update interval
            q = 1                                 # reset period counter
        else:
            q += 1                                # increment period counter
    return a / p
```

**SBA (Syntetos-Boylan Approximation):** Croston's method is theoretically biased upward (overestimates the forecast rate). SBA corrects this by multiplying by (1 − α/2), producing unbiased estimates asymptotically. In practice, SBA is recommended for most intermittent demand applications; in our data, both methods performed similarly (MAPE 70.2% vs 70.3%).

**Result:** Croston reduced negative bias from −1.88 (Naive) to −1.39 — a meaningful reduction in systematic under-forecasting on intermittent SKUs.

---

## 4. AIC for Model Selection

In the ARIMA grid search (18 combinations per SKU), we selected the best model by **Akaike Information Criterion (AIC):**

```
AIC = 2k − 2 ln(L)
```

where k = number of parameters and L = the maximised likelihood. Lower AIC is better.

**Why AIC over in-sample R² or log-likelihood?**
- R² always improves when adding parameters — an ARIMA(2,1,2) will always fit better in-sample than ARIMA(0,1,1), even if the extra parameters are pure noise.
- The `−2k` term in AIC **penalises model complexity** — each additional parameter must justify itself by improving fit by at least 2 log-likelihood units.
- AIC is asymptotically equivalent to leave-one-out cross-validation for time series, making it a principled out-of-sample proxy.

**Practical note:** AIC assumes large samples. With 638 training days, sample size is adequate. For shorter series (<50 observations), AICc (corrected AIC) would be preferable.

**Result in this project:** Most SKUs selected ARIMA(0,1,1) by AIC — equivalent to the IMA(1,1) model that SES optimally estimates. This is consistent with the EDA finding that all series are stationary and the SES baseline performing similarly to ARIMA.

---

## 5. Why Statistical Significance Testing

Looking at mean MAPE across 112 SKUs, LightGBM (59.2%) beats SES (62.2%) by 2.97 percentage points. But is this a reliable finding, or could it be driven by a handful of SKUs where LightGBM happened to do well?

**The problem with point estimates:** If one model has better average MAPE, it might be because it had lucky results on a few outlier SKUs while performing identically on most. On new data (a different 92-day window, different SKU mix), the relationship might reverse.

**Paired tests** control for SKU-level variance by computing per-SKU MAPE differences and testing whether those differences are consistently non-zero:

- **Paired t-test** (parametric): Assumes differences are approximately normally distributed. Result: t=2.016, p=0.046. Significant.
- **Wilcoxon signed-rank** (non-parametric): No normality assumption; more robust to outlier SKUs. Result: W=2326, p=0.015. Also significant.

Both tests agree: LightGBM's improvement over SES is statistically reliable at the p<0.05 level — it's not driven by a few lucky SKUs. However, the practical effect size (2.97pp improvement) is modest. This informs our recommendation: the ML model is recommended not primarily because it's dramatically more accurate, but because its global training structure makes it equally cheap to run across 1,664 SKUs while providing consistent, statistically verified improvement.

---

## 6. Why Cost and Maintenance Are First-Class Criteria

The technically best model (lowest MAPE) is not always the right model for production. In a logistics demand forecasting context with 1,664 SKUs:

**ARIMA/SARIMA provides no accuracy advantage over LightGBM** in our data (ARIMA MAPE: 80.3% vs LightGBM 57.4%), yet costs **670× more** to retrain weekly ($1.48/month vs $0.0022/month). The extra cost buys nothing in accuracy.

**SARIMA specifically** (maintenance complexity 3.75/5) requires:
- Manual order selection or grid search per SKU (18+ fits per retrain)
- Convergence monitoring and fallback logic in production
- Separate seasonal model for each SKU, not shareable across the catalogue
- Significantly higher inference latency (0.5s/SKU vs 0.001s for ML)

**The core trade-off in large-catalogue logistics:**  
Statistical models (ARIMA, SARIMA) are theoretically principled for individual series but scale linearly with catalogue size — 1,664 SKUs × 8 seconds/SKU = 3.7 hours per ARIMA retrain cycle. Global ML models train once in 30 seconds regardless of catalogue size. As catalogues grow (DHL adds new SKUs, expands warehouses), the ML architecture requires no additional compute, while ARIMA costs grow proportionally.

**The recommendation hierarchy:**
1. If cost savings from better forecasting justify ML complexity → deploy ML (true here for A- and B-class)
2. If compute cost savings outweigh the small accuracy gain → use SES (relevant for C-class, but LightGBM is so cheap it wins anyway)
3. If accuracy difference is negligible and costs are much lower → use the simpler model (why Naive is retained as a benchmark and fallback, not production)
