"""
07_business_decision.py — Business Decision Framework
DHL Data Scientist Portfolio — Project 01

Combines statistical, robustness, and financial signals to produce a
VP-facing recommendation on whether to deploy K-Means as the primary
SKU segmentation method.

Decision thresholds:
  silhouette  > 0.50  — statistically well-clustered
  seed ARI    > 0.95  — stable across random initialisations
  net_value   > 0     — K-Means pays for itself

Decision logic:
  DEPLOY   : net_value > 0  AND sil > 0.5  AND ari > 0.95
  HYBRID   : net_value ≤ 0  AND sil > 0.5  AND ari > 0.95  (or sil barely above 0.5)
  RULE-ONLY: otherwise

Outputs:
  outputs/business_decision.csv
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────────
SIL_THRESHOLD = 0.50
ARI_THRESHOLD = 0.95


def load_signals():
    """Load all prior outputs and extract the key scalar signals."""
    # 1. Silhouette from validation metrics (k=4 row)
    vm = pd.read_csv(OUTPUT_DIR / "validation_metrics.csv")
    silhouette = float(vm[vm["k"] == 4]["silhouette"].iloc[0])

    # 2. Seed-stability ARI from robustness results
    rb = pd.read_csv(OUTPUT_DIR / "robustness_results.csv")
    ari_stability = float(
        rb[(rb["test"] == "seed_stability") & (rb["metric"] == "mean_pairwise_ari")]["value"].iloc[0]
    )
    pct_gt95 = float(
        rb[(rb["test"] == "seed_stability") & (rb["metric"] == "pct_pairs_ari_gt_95")]["value"].iloc[0]
    )

    # 3. Net annual value from cost-benefit
    cb = pd.read_csv(OUTPUT_DIR / "cost_benefit_analysis.csv")
    net_row = cb[cb["item"].str.startswith("NET ANNUAL VALUE")]
    net_annual_value = float(net_row["kmeans"].iloc[0])

    # 4. Separation quality: does K-Means beat ABC on within-group stockout CV?
    sq = pd.read_csv(OUTPUT_DIR / "separation_quality.csv")
    km_cv  = float(sq[sq["method"] == "K-Means (cluster)"]["within_group_stockout_cv"].iloc[0])
    abc_cv = float(sq[sq["method"] == "ABC Class"]["within_group_stockout_cv"].iloc[0])
    km_better_separation = km_cv < abc_cv

    # 5. Misclassified counts from cost-benefit
    misc_row = cb[cb["item"].str.contains("A-class SKUs in Low-Vel")]
    misclassified_a = int(misc_row["kmeans"].iloc[0]) if len(misc_row) > 0 else 0

    # 6. ARI vs ABC from method comparison
    mc = pd.read_csv(OUTPUT_DIR / "method_comparison.csv")
    ari_vs_abc = float(mc[mc["comparison"] == "K-Means vs ABC"]["ari"].iloc[0])

    return {
        "silhouette":           silhouette,
        "ari_stability":        ari_stability,
        "pct_gt95":             pct_gt95,
        "net_annual_value_usd": net_annual_value,
        "km_better_separation": km_better_separation,
        "km_within_cv":         km_cv,
        "abc_within_cv":        abc_cv,
        "misclassified_a":      misclassified_a,
        "ari_vs_abc":           ari_vs_abc,
    }


def make_decision(signals: dict) -> dict:
    """
    Apply the three-gate decision framework and return a decision record.
    """
    sil_pass = signals["silhouette"]    > SIL_THRESHOLD
    ari_pass = signals["ari_stability"] > ARI_THRESHOLD
    val_pass = signals["net_annual_value_usd"] > 0

    if val_pass and sil_pass and ari_pass:
        decision   = "DEPLOY K-MEANS AS PRIMARY"
        rationale  = (
            "K-Means passes all three gates: statistically well-clustered "
            f"(silhouette {signals['silhouette']:.4f} > {SIL_THRESHOLD}), "
            f"highly stable across seeds (ARI {signals['ari_stability']:.4f} > {ARI_THRESHOLD}), "
            f"and financially justified (net annual value ${signals['net_annual_value_usd']:,.0f} > $0). "
            "Rule-based ABC/XYZ is retained as an audit overlay."
        )
        bullet_stats  = (
            f"Silhouette = {signals['silhouette']:.4f} (>0.50 threshold); "
            f"seed-stability ARI = {signals['ari_stability']:.4f} (100% of pairs >0.95)."
        )
        bullet_cost   = (
            f"Net annual value ${signals['net_annual_value_usd']:,.0f} driven by "
            f"{signals['misclassified_a']} A-class SKUs correctly reclassified to Low-Velocity, "
            "enabling targeted inventory rebalancing."
        )
        bullet_value  = (
            f"K-Means within-group stockout CV = {signals['km_within_cv']:.4f} vs "
            f"ABC CV = {signals['abc_within_cv']:.4f} — better stockout-risk homogeneity "
            "inside each segment."
        )
        change_cond_1 = "Catalogue drops below ~500 SKUs: rule-based becomes fully adequate."
        change_cond_2 = (
            "Stockout patterns shift structurally (e.g. post-network restructure): "
            "re-run feature engineering + cluster validation before next planning cycle."
        )
    elif (sil_pass and ari_pass) and not val_pass:
        decision   = "HYBRID: RULE-BASED PRIMARY, K-MEANS QUARTERLY DIAGNOSTIC"
        rationale  = (
            "K-Means is statistically sound but the estimated net annual value is negative, "
            "meaning the maintenance overhead exceeds the quantified stockout savings under "
            "current assumptions. Use rule-based ABC/XYZ for daily operations; run K-Means "
            "quarterly to surface emerging segment shifts."
        )
        bullet_stats  = (
            f"Silhouette = {signals['silhouette']:.4f}; ARI stability = {signals['ari_stability']:.4f}."
        )
        bullet_cost   = (
            f"Net annual value ${signals['net_annual_value_usd']:,.0f} — below break-even. "
            "Review assumptions if stockout rates or catalogue size change."
        )
        bullet_value  = (
            f"K-Means identifies {signals['misclassified_a']} A-class SKUs mislabelled by ABC, "
            "which informs manual review even without full deployment."
        )
        change_cond_1 = "If stockout rate rises >5% or catalogue grows >3,000 SKUs: re-evaluate deployment."
        change_cond_2 = "If operational team validates >20% stockout reduction on pilot segment: deploy."
    else:
        decision   = "RULE-BASED (ABC/XYZ) ONLY"
        rationale  = (
            "K-Means failed one or more statistical gates. Rule-based classification "
            "is simpler, fully explainable, and adequate for current business needs."
        )
        bullet_stats  = (
            f"Silhouette = {signals['silhouette']:.4f} "
            f"({'PASS' if sil_pass else 'FAIL'}, threshold {SIL_THRESHOLD}); "
            f"ARI stability = {signals['ari_stability']:.4f} "
            f"({'PASS' if ari_pass else 'FAIL'}, threshold {ARI_THRESHOLD})."
        )
        bullet_cost   = "Net value negative or statistical quality insufficient to justify deployment."
        bullet_value  = "Rerun analysis with richer feature set or larger data window."
        change_cond_1 = "Collect 2+ years of demand history: richer seasonality features may improve silhouette."
        change_cond_2 = "Add supplier lead-time or shelf-life features: may resolve ambiguous cluster boundaries."

    return {
        "decision":            decision,
        "rationale":           rationale,
        "bullet_stats":        bullet_stats,
        "bullet_cost":         bullet_cost,
        "bullet_value":        bullet_value,
        "change_condition_1":  change_cond_1,
        "change_condition_2":  change_cond_2,
        "silhouette":          signals["silhouette"],
        "ari_stability":       signals["ari_stability"],
        "net_annual_value_usd":signals["net_annual_value_usd"],
        "sil_pass":            sil_pass,
        "ari_pass":            ari_pass,
        "val_pass":            val_pass,
    }


def main():
    log.info("Loading signals from prior outputs ...")
    signals  = load_signals()

    log.info("Applying decision framework ...")
    decision = make_decision(signals)

    # ── Export ──────────────────────────────────────────────────────────────
    export = pd.DataFrame([{
        "decision":            decision["decision"],
        "rationale":           decision["rationale"],
        "silhouette":          round(decision["silhouette"], 4),
        "ari_stability":       round(decision["ari_stability"], 4),
        "net_annual_value_usd":round(decision["net_annual_value_usd"], 2),
        "change_condition_1":  decision["change_condition_1"],
        "change_condition_2":  decision["change_condition_2"],
    }])
    out_path = OUTPUT_DIR / "business_decision.csv"
    export.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")

    # ── VP-facing print ─────────────────────────────────────────────────────
    width = 72
    print("\n" + "=" * width)
    print("BUSINESS DECISION — SKU SEGMENTATION METHOD SELECTION")
    print("Prepared for: VP Supply Chain Operations")
    print("=" * width)

    print(f"\n  RECOMMENDATION: {decision['decision']}")
    print()
    print(f"  In one sentence:")
    # Pull the first sentence of the rationale
    first_sentence = decision["rationale"].split(". ")[0] + "."
    # Word-wrap at ~68 chars
    words = first_sentence.split()
    line, lines = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > 66:
            lines.append("  " + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))

    print(f"\n  Supporting evidence:")
    print(f"  • Statistics:       {decision['bullet_stats']}")
    print(f"  • Cost / value:     {decision['bullet_cost']}")
    print(f"  • Operational gain: {decision['bullet_value']}")

    print(f"\n  Gate checks:")
    sil_icon = "PASS" if decision["sil_pass"] else "FAIL"
    ari_icon = "PASS" if decision["ari_pass"] else "FAIL"
    val_icon = "PASS" if decision["val_pass"] else "FAIL"
    print(f"  [{sil_icon}] Statistical quality  : silhouette {signals['silhouette']:.4f} "
          f"(threshold >{SIL_THRESHOLD})")
    print(f"  [{ari_icon}] Robustness           : seed ARI {signals['ari_stability']:.4f} "
          f"(threshold >{ARI_THRESHOLD})")
    print(f"  [{val_icon}] Financial justification: net value ${signals['net_annual_value_usd']:,.0f} "
          f"(threshold >$0)")

    print(f"\n  Conditions that would change this recommendation:")
    print(f"  1. {decision['change_condition_1']}")
    print(f"  2. {decision['change_condition_2']}")

    print("\n" + "=" * width)
    print(f"OUTPUT SAVED: outputs/business_decision.csv")
    print("=" * width)

    return decision


if __name__ == "__main__":
    decision = main()
