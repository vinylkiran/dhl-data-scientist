"""
04_ml_models.py
XGBoost and LightGBM global demand forecasting models.
Train on all eval SKUs, global model with SKU features.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["Date"])

# Per-SKU per-date: aggregate across warehouses
daily = (demand.groupby(["Date", "SKU_ID", "ABC_Class", "Category"])
         .agg(Quantity_Demanded=("Quantity_Demanded", "sum"),
              Stockout_Flag=("Stockout_Flag", "max"))
         .reset_index())

all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
n_days = len(all_dates)
TRAIN_END = pd.Timestamp("2023-09-30")
TEST_START = pd.Timestamp("2023-10-01")

# ── Load eval SKUs ────────────────────────────────────────────────────────────
eval_df = pd.read_csv(OUTPUTS_DIR / "eval_skus.csv")
eval_skus = eval_df["sku_id"].tolist()
print(f"Eval SKUs: {len(eval_skus)}")

# ── Build full SKU × date panel (vectorized) ─────────────────────────────────
print("Building SKU × date panel...")
# Filter demand to eval SKUs
daily_eval = daily[daily["SKU_ID"].isin(eval_skus)].copy()

# Create a complete cross product: each SKU × all dates
sku_list = eval_skus
date_range_df = pd.DataFrame({"Date": all_dates})
sku_df = pd.DataFrame({"SKU_ID": sku_list})
full_idx = date_range_df.assign(key=1).merge(sku_df.assign(key=1), on="key").drop("key", axis=1)
print(f"Full panel shape before merge: {full_idx.shape}")

# Merge demand
panel = full_idx.merge(
    daily_eval[["Date", "SKU_ID", "ABC_Class", "Category", "Quantity_Demanded", "Stockout_Flag"]],
    on=["Date", "SKU_ID"], how="left"
)
panel["Quantity_Demanded"] = panel["Quantity_Demanded"].fillna(0)
panel["Stockout_Flag"] = panel["Stockout_Flag"].fillna(0)

# Fill ABC_Class and Category (constant per SKU)
sku_meta = daily_eval.groupby("SKU_ID")[["ABC_Class", "Category"]].first().reset_index()
panel = panel.drop(columns=["ABC_Class", "Category"]).merge(sku_meta, on="SKU_ID", how="left")
panel = panel.sort_values(["SKU_ID", "Date"]).reset_index(drop=True)
print(f"Panel shape: {panel.shape}")

# ── Feature engineering (vectorized by SKU) ───────────────────────────────────
print("Engineering features...")

def add_features(df):
    df = df.sort_values(["SKU_ID", "Date"]).copy()
    g = df.groupby("SKU_ID")["Quantity_Demanded"]

    # Lag features
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)

    # Rolling means
    df["rolling_mean_7"] = g.shift(1).transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["rolling_mean_14"] = g.shift(1).transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["rolling_mean_28"] = g.shift(1).transform(lambda x: x.rolling(28, min_periods=1).mean())

    # Rolling std
    df["rolling_std_7"] = g.shift(1).transform(lambda x: x.rolling(7, min_periods=2).std().fillna(0))

    # Calendar
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)

    # Days since last stockout (per SKU)
    def days_since_stockout(grp):
        result = []
        last_so = -999
        for i, (idx, row) in enumerate(grp.iterrows()):
            if row["Stockout_Flag"] == 1:
                last_so = i
            result.append(i - last_so if last_so >= 0 else 999)
        return pd.Series(result, index=grp.index)

    df["days_since_stockout"] = (df.groupby("SKU_ID")
                                   .apply(days_since_stockout)
                                   .reset_index(level=0, drop=True))

    return df

panel = add_features(panel)

# Encode categorical
abc_enc = {"A": 2, "B": 1, "C": 0}
panel["abc_class_encoded"] = panel["ABC_Class"].map(abc_enc).fillna(0).astype(int)

le = LabelEncoder()
panel["category_encoded"] = le.fit_transform(panel["Category"].fillna("Unknown"))

# Drop rows with NaN features (first 28 days per SKU)
FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7",
    "day_of_week", "month", "quarter", "is_weekend",
    "abc_class_encoded", "category_encoded", "days_since_stockout"
]
TARGET = "Quantity_Demanded"

panel_clean = panel.dropna(subset=FEATURE_COLS).copy()
print(f"Panel after dropping NaN rows: {panel_clean.shape}")

# ── Train / test split ────────────────────────────────────────────────────────
train = panel_clean[panel_clean["Date"] <= TRAIN_END].copy()
test = panel_clean[panel_clean["Date"] >= TEST_START].copy()
print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values
X_test = test[FEATURE_COLS].values
y_test = test[TARGET].values

# ── TimeSeriesSplit CV ────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=3)

def cv_rmse(model_fn, params, X, y):
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_fn(params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - pred) ** 2))
        scores.append(rmse)
    return np.mean(scores)

# ── XGBoost hyperparameter tuning ────────────────────────────────────────────
print("\nTuning XGBoost...")
xgb_grid = [
    {"n_estimators": ne, "max_depth": md, "learning_rate": lr}
    for ne in [100, 300]
    for md in [3, 5]
    for lr in [0.05, 0.1]
]

def make_xgb(params):
    return xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )

best_xgb_params = None
best_xgb_rmse = np.inf
for i, params in enumerate(xgb_grid):
    rmse = cv_rmse(make_xgb, params, X_train, y_train)
    print(f"  XGB {i+1}/{len(xgb_grid)}: {params} -> CV RMSE={rmse:.3f}")
    if rmse < best_xgb_rmse:
        best_xgb_rmse = rmse
        best_xgb_params = params

print(f"Best XGB params: {best_xgb_params} (CV RMSE={best_xgb_rmse:.3f})")

# Train final XGBoost
xgb_model = make_xgb(best_xgb_params)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_preds = np.maximum(xgb_preds, 0)

# ── LightGBM hyperparameter tuning ───────────────────────────────────────────
print("\nTuning LightGBM...")
lgb_grid = [
    {"n_estimators": ne, "num_leaves": nl, "learning_rate": lr}
    for ne in [100, 300]
    for nl in [31, 63]
    for lr in [0.05, 0.1]
]

def make_lgb(params):
    return lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"],
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )

best_lgb_params = None
best_lgb_rmse = np.inf
for i, params in enumerate(lgb_grid):
    rmse = cv_rmse(make_lgb, params, X_train, y_train)
    print(f"  LGB {i+1}/{len(lgb_grid)}: {params} -> CV RMSE={rmse:.3f}")
    if rmse < best_lgb_rmse:
        best_lgb_rmse = rmse
        best_lgb_params = params

print(f"Best LGB params: {best_lgb_params} (CV RMSE={best_lgb_rmse:.3f})")

# Train final LightGBM
lgb_model = make_lgb(best_lgb_params)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_preds = np.maximum(lgb_preds, 0)

# ── Per-SKU metrics ───────────────────────────────────────────────────────────
def compute_metrics(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mae = np.mean(np.abs(actual - forecast))
    return mape, rmse, mae

test_with_preds = test.copy()
test_with_preds["xgb_pred"] = xgb_preds
test_with_preds["lgb_pred"] = lgb_preds

records = []
for sku_id in eval_skus:
    sku_test = test_with_preds[test_with_preds["SKU_ID"] == sku_id]
    if len(sku_test) == 0:
        continue
    abc_cls = sku_test["ABC_Class"].iloc[0]
    actual = sku_test[TARGET].values

    for model_name, pred_col in [("xgboost", "xgb_pred"), ("lightgbm", "lgb_pred")]:
        preds = sku_test[pred_col].values
        m, r, a = compute_metrics(actual, preds)
        records.append({
            "sku_id": sku_id, "abc_class": abc_cls,
            "model": model_name, "mape": round(m, 3), "rmse": round(r, 3), "mae": round(a, 3)
        })

results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUTS_DIR / "ml_results.csv", index=False)
print(f"\nSaved outputs/ml_results.csv ({len(results_df)} rows)")

# ── Feature importance plots ──────────────────────────────────────────────────
print("\nGenerating feature importance plots...")

# XGBoost
xgb_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(xgb_imp["feature"], xgb_imp["importance"], color="steelblue", alpha=0.8)
ax.set_title("XGBoost — Top 15 Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (F-score)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ml_xgb_importance.png", dpi=120, bbox_inches="tight")
plt.close()

# LightGBM
lgb_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(lgb_imp["feature"], lgb_imp["importance"], color="darkorange", alpha=0.8)
ax.set_title("LightGBM — Top 15 Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (split count)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ml_lgb_importance.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved feature importance figures.")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ML RESULTS — OVERALL MEAN METRICS")
print("=" * 60)
overall = results_df.groupby("model")[["mape", "rmse", "mae"]].mean().round(3)
print(overall.to_string())

print("\n" + "=" * 60)
print("ML RESULTS — BY MODEL AND ABC CLASS")
print("=" * 60)
by_class = results_df.groupby(["model", "abc_class"])[["mape", "rmse", "mae"]].mean().round(3)
print(by_class.to_string())
print("=" * 60)

print(f"\nBest XGBoost params: {best_xgb_params}")
print(f"Best LightGBM params: {best_lgb_params}")
print("\nDone — 04_ml_models.py complete.")
