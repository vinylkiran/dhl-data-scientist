"""
01_feature_engineering.py — SKU-Level Feature Engineering for Clustering
DHL Data Scientist Portfolio — Project 01

Engineers 9 SKU-level features from daily_demand.csv:
  1. mean_daily_demand       — average daily demand across all days and warehouses
  2. std_demand              — standard deviation of daily demand
  3. cv_demand               — coefficient of variation (std / mean)
  4. total_revenue           — total revenue over the study period
  5. revenue_rank_pct        — revenue rank percentile (0=lowest, 1=highest)
  6. demand_frequency        — % of days with non-zero demand
  7. avg_order_size          — average demand quantity on days when demand > 0
  8. demand_trend            — linear regression slope of demand over time (units/day)
  9. seasonality_strength    — ratio of seasonal variance to total variance (STL decomposition)

Outputs:
  outputs/sku_features.csv  — raw + standardised features
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR.parent.parent / "shared" / "data" / "dhl-synthetic"
OUTPUT_DIR   = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEMAND_FILE = DATA_DIR / "daily_demand.csv"
SKU_FILE    = DATA_DIR / "sku_master.csv"


# ---------------------------------------------------------------------------
# Seasonality helper: simple decomposition using monthly means
# ---------------------------------------------------------------------------

def seasonality_strength_sku(series: pd.Series) -> float:
    """
    Estimate seasonality strength as the ratio of seasonal variance to total variance.
    Seasonal component estimated by monthly mean deviations (classical decomposition proxy).
    Returns a value in [0, 1]; higher = more seasonal.
    """
    if len(series) < 24:
        return 0.0
    s = series.copy().reset_index(drop=True)
    s.index = pd.date_range("2022-01-01", periods=len(s), freq="D")
    # monthly means (seasonal signal)
    monthly_mean = s.groupby(s.index.month).transform("mean")
    overall_mean = s.mean()
    seasonal_component = monthly_mean - overall_mean
    seasonal_var = seasonal_component.var()
    total_var    = s.var()
    if total_var == 0:
        return 0.0
    return float(np.clip(seasonal_var / total_var, 0, 1))


def demand_trend(series: pd.Series) -> float:
    """Linear regression slope of demand over time (days from start)."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope, _, _, _, _ = stats.linregress(x, series.values)
    return float(slope)


# ---------------------------------------------------------------------------
# Main feature engineering
# ---------------------------------------------------------------------------

def engineer_features(demand_path: Path, sku_path: Path) -> pd.DataFrame:
    log.info("Loading daily_demand.csv ...")
    df = pd.read_csv(demand_path, parse_dates=["Date"])
    df = df.sort_values(["SKU_ID", "Date"])

    log.info(f"  {len(df):,} rows | {df['SKU_ID'].nunique():,} SKUs | "
             f"{df['Date'].min().date()} – {df['Date'].max().date()}")

    # Aggregate across warehouses to SKU level (sum demand per day per SKU)
    log.info("Aggregating to SKU × date level ...")
    daily = (
        df.groupby(["SKU_ID", "Date"])
        .agg(qty=("Quantity_Demanded", "sum"), revenue=("Revenue", "sum"),
             stockout=("Stockout_Flag", "max"))
        .reset_index()
    )

    # Date spine: every calendar date for the full study period
    date_spine = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    n_days = len(date_spine)
    log.info(f"  Date spine: {n_days} days")

    # Fill missing dates per SKU with 0 demand
    all_dates = pd.DataFrame({"Date": date_spine})
    records = []

    sku_ids = daily["SKU_ID"].unique()
    log.info(f"  Engineering features for {len(sku_ids):,} SKUs ...")

    for sku_id in sku_ids:
        sku_df = daily[daily["SKU_ID"] == sku_id].copy()
        full = all_dates.merge(sku_df, on="Date", how="left").fillna(
            {"qty": 0, "revenue": 0, "stockout": 0}
        )
        q = full["qty"]

        mean_d    = q.mean()
        std_d     = q.std()
        cv_d      = (std_d / mean_d) if mean_d > 0 else 0.0
        total_rev = sku_df["revenue"].sum()
        freq      = (q > 0).sum() / n_days
        nonzero   = q[q > 0]
        avg_ord   = nonzero.mean() if len(nonzero) > 0 else 0.0
        trend     = demand_trend(q)
        seas      = seasonality_strength_sku(q)

        records.append({
            "sku_id":           sku_id,
            "mean_daily_demand": mean_d,
            "std_demand":        std_d,
            "cv_demand":         cv_d,
            "total_revenue":     total_rev,
            "demand_frequency":  freq,
            "avg_order_size":    avg_ord,
            "demand_trend":      trend,
            "seasonality_strength": seas,
        })

    feat = pd.DataFrame(records)

    # Revenue rank percentile
    feat["revenue_rank_pct"] = feat["total_revenue"].rank(pct=True)

    # Merge in ABC/XYZ class and category from demand data (mode per SKU)
    meta = (
        df.groupby("SKU_ID")
        .agg(abc_class=("ABC_Class", lambda x: x.mode()[0]),
             xyz_class=("XYZ_Class", lambda x: x.mode()[0]),
             category=("Category",  lambda x: x.mode()[0]),
             stockout_rate=("Stockout_Flag", "mean"))
        .reset_index()
        .rename(columns={"SKU_ID": "sku_id"})
    )
    feat = feat.merge(meta, on="sku_id", how="left")

    # --- Feature matrix for scaling ---
    FEATURE_COLS = [
        "mean_daily_demand", "std_demand", "cv_demand",
        "total_revenue", "revenue_rank_pct",
        "demand_frequency", "avg_order_size",
        "demand_trend", "seasonality_strength",
    ]

    log.info("Standardising features (z-score) ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat[FEATURE_COLS])

    scaled_cols = {f + "_z": X_scaled[:, i] for i, f in enumerate(FEATURE_COLS)}
    feat_out = pd.concat([feat, pd.DataFrame(scaled_cols, index=feat.index)], axis=1)

    out_path = OUTPUT_DIR / "sku_features.csv"
    feat_out.to_csv(out_path, index=False)
    log.info(f"  Saved: {out_path}")

    return feat_out, FEATURE_COLS


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(feat: pd.DataFrame, feature_cols: list):
    print("\n" + "=" * 65)
    print("FEATURE SUMMARY STATISTICS")
    print("=" * 65)
    summary = feat[feature_cols].describe().T
    summary["cv"] = summary["std"] / summary["mean"].abs()
    print(summary[["count", "mean", "std", "min", "50%", "max", "cv"]].round(4).to_string())

    print("\n" + "=" * 65)
    print("FEATURE CORRELATION MATRIX (raw features)")
    print("=" * 65)
    corr = feat[feature_cols].corr().round(3)
    print(corr.to_string())

    print("\n" + "=" * 65)
    print("SKU COUNT BY ABC / XYZ CLASS")
    print("=" * 65)
    print(feat.groupby(["abc_class", "xyz_class"]).size().unstack(fill_value=0).to_string())

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"  SKUs with features : {len(feat):,}")
    print(f"  Features engineered: {len(feature_cols)}")
    print(f"  Output             : outputs/sku_features.csv")
    print("=" * 65)


if __name__ == "__main__":
    feat, feature_cols = engineer_features(DEMAND_FILE, SKU_FILE)
    print_summary(feat, feature_cols)
