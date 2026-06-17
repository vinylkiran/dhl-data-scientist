"""
01_feature_engineering.py — SKU-Level Feature Engineering for Clustering
DHL Data Scientist Portfolio — DS Project 01 (Rebuild)

Engineers 9 SKU-level features from daily_demand.csv.
Each feature captures a distinct dimension of demand behaviour relevant
to inventory management decisions.

Features:
  1. mean_daily_demand    — velocity (average units/day across all calendar days)
  2. std_demand           — absolute variability
  3. cv_demand            — relative variability (std/mean); the XYZ signal
  4. total_revenue        — financial scale of the SKU
  5. revenue_rank_pct     — percentile rank on revenue [0,1]; compresses right tail
  6. demand_frequency     — % of calendar days with demand > 0 (intermittency)
  7. avg_order_size       — mean qty on active days; large-infrequent vs small-regular
  8. demand_trend         — OLS slope of demand over time (units/day); growing vs declining
  9. seasonality_strength — seasonal variance / total variance (monthly decomposition proxy)

All 9 features are z-score standardised before export.

Outputs:
  outputs/sku_features.csv
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMAND_FILE = DATA_DIR / "daily_demand.csv"
SKU_FILE    = DATA_DIR / "sku_master.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "mean_daily_demand", "std_demand", "cv_demand",
    "total_revenue", "revenue_rank_pct",
    "demand_frequency", "avg_order_size",
    "demand_trend", "seasonality_strength",
]


def _demand_trend(series: pd.Series) -> float:
    if len(series) < 10:
        return 0.0
    slope, *_ = stats.linregress(np.arange(len(series), dtype=float), series.values)
    return float(slope)


def _seasonality_strength(series: pd.Series, date_index: pd.DatetimeIndex) -> float:
    if len(series) < 24 or series.var() == 0:
        return 0.0
    monthly_mean  = series.groupby(date_index.month).transform("mean")
    seasonal_var  = (monthly_mean - series.mean()).var()
    return float(np.clip(seasonal_var / series.var(), 0.0, 1.0))


def engineer_features() -> pd.DataFrame:
    log.info("Loading daily_demand.csv …")
    df = pd.read_csv(DEMAND_FILE, parse_dates=["Date"])
    df = df.sort_values(["SKU_ID", "Date"])
    log.info(f"  {len(df):,} rows | {df['SKU_ID'].nunique():,} SKUs | "
             f"{df['Date'].min().date()} – {df['Date'].max().date()}")

    log.info("Aggregating to SKU × date …")
    daily = (df.groupby(["SKU_ID", "Date"])
             .agg(qty=("Quantity_Demanded", "sum"),
                  revenue=("Revenue", "sum"),
                  stockout=("Stockout_Flag", "max"))
             .reset_index())

    date_spine = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    n_days     = len(date_spine)

    sku_meta = (df.groupby("SKU_ID")
                .agg(abc_class    =("ABC_Class",     lambda x: x.mode().iloc[0]),
                     xyz_class    =("XYZ_Class",     lambda x: x.mode().iloc[0]),
                     category     =("Category",      lambda x: x.mode().iloc[0]),
                     stockout_rate=("Stockout_Flag", "mean"))
                .reset_index().rename(columns={"SKU_ID": "sku_id"}))

    all_dates = pd.DataFrame({"Date": date_spine})
    records   = []

    sku_ids = daily["SKU_ID"].unique()
    log.info(f"  Engineering features for {len(sku_ids):,} SKUs …")

    for sku_id in sku_ids:
        sub  = daily[daily["SKU_ID"] == sku_id]
        full = all_dates.merge(sub, on="Date", how="left").fillna(
                   {"qty": 0.0, "revenue": 0.0, "stockout": 0.0})
        q    = full["qty"]
        di   = pd.DatetimeIndex(full["Date"])

        mean_d  = float(q.mean())
        std_d   = float(q.std())
        cv_d    = (std_d / mean_d) if mean_d > 0 else 0.0
        tot_rev = float(sub["revenue"].sum())
        freq    = float((q > 0).sum() / n_days)
        nz      = q[q > 0]
        avg_ord = float(nz.mean()) if len(nz) > 0 else 0.0
        trend   = _demand_trend(q)
        seas    = _seasonality_strength(q.reset_index(drop=True), di)

        records.append(dict(sku_id=sku_id,
                            mean_daily_demand=mean_d, std_demand=std_d, cv_demand=cv_d,
                            total_revenue=tot_rev, demand_frequency=freq,
                            avg_order_size=avg_ord, demand_trend=trend,
                            seasonality_strength=seas))

    feat = pd.DataFrame(records)
    feat["revenue_rank_pct"] = feat["total_revenue"].rank(pct=True)
    feat = feat.merge(sku_meta, on="sku_id", how="left")

    log.info("Z-score standardising …")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat[FEATURE_COLS])
    for i, col in enumerate(FEATURE_COLS):
        feat[col + "_z"] = X_scaled[:, i]

    out = OUTPUT_DIR / "sku_features.csv"
    feat.to_csv(out, index=False)
    log.info(f"  Saved → {out}")
    return feat


def print_summary(feat: pd.DataFrame) -> None:
    print("\n" + "═" * 68)
    print("FEATURE SUMMARY STATISTICS")
    print("═" * 68)
    desc = feat[FEATURE_COLS].describe().T
    desc["cv"] = (desc["std"] / desc["mean"].abs()).round(4)
    print(desc[["count","mean","std","min","50%","max","cv"]].round(4).to_string())

    print("\n" + "═" * 68)
    print("FEATURE CORRELATION MATRIX (raw features)")
    print("═" * 68)
    print(feat[FEATURE_COLS].corr().round(3).to_string())

    print("\n" + "═" * 68)
    print("SKU COUNT BY ABC × XYZ CLASS")
    print("═" * 68)
    print(feat.groupby(["abc_class","xyz_class"]).size().unstack(fill_value=0).to_string())

    print("\n" + "═" * 68)
    print("STOCKOUT RATE BY ABC CLASS  (baseline for later comparison)")
    print("═" * 68)
    print(feat.groupby("abc_class")["stockout_rate"]
          .agg(["mean","std","min","max"]).round(4).to_string())

    print("\n" + "═" * 68)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"  SKUs: {len(feat):,}   Features: {len(FEATURE_COLS)}   Output: outputs/sku_features.csv")
    print("═" * 68)


if __name__ == "__main__":
    feat = engineer_features()
    print_summary(feat)
