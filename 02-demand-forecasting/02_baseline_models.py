"""
02_baseline_models.py
Baseline forecasting models: Naive, Seasonal Naive (period=7), SES.
Evaluated on 12 sample SKUs + 100 stratified-random active SKUs.
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])

daily = (demand.groupby(["Date", "SKU_ID", "ABC_Class", "Category"])["Quantity_Demanded"]
         .sum().reset_index())

all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
n_days = len(all_dates)

TRAIN_END = pd.Timestamp("2023-09-30")
TEST_START = pd.Timestamp("2023-10-01")

train_dates = all_dates[all_dates <= TRAIN_END]
test_dates = all_dates[all_dates >= TEST_START]
n_train_days = len(train_dates)
print(f"Train days: {n_train_days} | Test days: {len(test_dates)}")

# ── Identify active SKUs (demand on >= 30% of training days) ─────────────────
train_data = daily[daily["Date"] <= TRAIN_END].copy()
sku_train_stats = train_data.groupby("SKU_ID").agg(
    nonzero_days=("Quantity_Demanded", lambda x: (x > 0).sum())
).reset_index()
sku_train_stats["nonzero_pct"] = sku_train_stats["nonzero_days"] / n_train_days
active_skus = sku_train_stats[sku_train_stats["nonzero_pct"] >= 0.30]["SKU_ID"].tolist()
print(f"Active SKUs (>=30% nonzero in train): {len(active_skus)}")

sku_abc = daily.groupby("SKU_ID")[["ABC_Class", "Category"]].first().reset_index()

# ── Load 12 sample SKUs ───────────────────────────────────────────────────────
sample_df = pd.read_csv(OUTPUTS_DIR / "sample_skus.csv")
sample_skus = sample_df["sku_id"].tolist()
print(f"Sample SKUs loaded: {len(sample_skus)}")

# ── Stratified random 100 additional SKUs ────────────────────────────────────
np.random.seed(42)
active_with_abc = pd.DataFrame({"SKU_ID": active_skus}).merge(sku_abc, on="SKU_ID")
active_excl_sample = active_with_abc[~active_with_abc["SKU_ID"].isin(sample_skus)]

abc_counts = active_excl_sample["ABC_Class"].value_counts()
total_active = len(active_excl_sample)
n_extra = 100

extra_skus = []
for cls, cnt in abc_counts.items():
    n_cls = round(n_extra * cnt / total_active)
    pool = active_excl_sample[active_excl_sample["ABC_Class"] == cls]["SKU_ID"].tolist()
    chosen = np.random.choice(pool, min(n_cls, len(pool)), replace=False).tolist()
    extra_skus.extend(chosen)
extra_skus = extra_skus[:100]
print(f"Extra SKUs selected: {len(extra_skus)}")

eval_skus = list(dict.fromkeys(sample_skus + extra_skus))
print(f"Total eval SKUs (deduplicated): {len(eval_skus)}")

eval_abc = pd.DataFrame({"SKU_ID": eval_skus}).merge(sku_abc, on="SKU_ID")
eval_abc.rename(columns={"SKU_ID": "sku_id", "ABC_Class": "abc_class", "Category": "category"},
                inplace=True)
eval_abc.to_csv(OUTPUTS_DIR / "eval_skus.csv", index=False)
print("Saved outputs/eval_skus.csv")
print(eval_abc["abc_class"].value_counts())

# ── Helper ────────────────────────────────────────────────────────────────────
def get_sku_series(sku_id):
    s = daily[daily["SKU_ID"] == sku_id].set_index("Date")["Quantity_Demanded"]
    s = s.reindex(all_dates, fill_value=0)
    s.index.name = "Date"
    return s

def compute_metrics(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mae = np.mean(np.abs(actual - forecast))
    bias = np.mean(forecast - actual)
    return mape, rmse, mae, bias

# ── Models ────────────────────────────────────────────────────────────────────
def naive_forecast(train_series, n_steps):
    return np.full(n_steps, float(train_series.iloc[-1]))

def seasonal_naive_forecast(train_series, n_steps, period=7):
    last_season = train_series.iloc[-period:].values.astype(float)
    return np.array([last_season[i % period] for i in range(n_steps)])

def ses_forecast(train_series, n_steps):
    try:
        model = SimpleExpSmoothing(train_series.values.astype(float),
                                   initialization_method="estimated")
        fit = model.fit(optimized=True)
        return fit.forecast(n_steps)
    except Exception:
        return naive_forecast(train_series, n_steps)

# ── Evaluate ─────────────────────────────────────────────────────────────────
print(f"\nEvaluating {len(eval_skus)} SKUs on 3 baseline models...")
records = []
n_test = len(test_dates)

for i, sku_id in enumerate(eval_skus):
    if (i + 1) % 25 == 0:
        print(f"  Progress: {i+1}/{len(eval_skus)}")

    series = get_sku_series(sku_id)
    abc_cls_vals = eval_abc[eval_abc["sku_id"] == sku_id]["abc_class"].values
    abc_cls = abc_cls_vals[0] if len(abc_cls_vals) > 0 else "C"

    train_s = series[series.index <= TRAIN_END]
    test_s = series[series.index >= TEST_START]
    actual = test_s.values

    for model_name, fc_fn in [
        ("Naive", lambda ts, n: naive_forecast(ts, n)),
        ("SeasonalNaive", lambda ts, n: seasonal_naive_forecast(ts, n)),
        ("SES", lambda ts, n: ses_forecast(ts, n)),
    ]:
        fc = fc_fn(train_s, n_test)
        m, r, a, b = compute_metrics(actual, fc)
        records.append({"sku_id": sku_id, "abc_class": abc_cls, "model": model_name,
                        "mape": m, "rmse": r, "mae": a, "bias": b})

results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUTS_DIR / "baseline_results.csv", index=False)
print(f"\nSaved outputs/baseline_results.csv ({len(results_df)} rows)")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE RESULTS — OVERALL MEAN METRICS")
print("=" * 70)
overall = results_df.groupby("model")[["mape", "rmse", "mae", "bias"]].mean().round(3)
print(overall.to_string())

print("\n" + "=" * 70)
print("BASELINE RESULTS — BY MODEL AND ABC CLASS")
print("=" * 70)
by_class = results_df.groupby(["model", "abc_class"])[["mape", "rmse", "mae", "bias"]].mean().round(3)
print(by_class.to_string())
print("=" * 70)

print("\nDone — 02_baseline_models.py complete.")
