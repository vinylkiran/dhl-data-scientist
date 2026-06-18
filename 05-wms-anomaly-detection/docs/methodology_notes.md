# Methodology Notes — WMS Anomaly Detection

**Project:** DS Project 5  
**Audience:** Technical reviewers, data science peers, ops analytics team  

---

## 1. Why Both SPC and ML Rather Than Just ML

The instinct to "go straight to ML" is understandable but misguided in a warehouse operations context.

Statistical Process Control is the established international standard for manufacturing and logistics quality monitoring — ISO 8258, the Shewhart chart framework, and Western Electric's run rules have been deployed in industrial settings for 80+ years. When a floor supervisor sees a control chart, they understand it intuitively: the line went above the red boundary, here is the exact metric, here is the date. They can act on it without a data scientist in the room.

Isolation Forest produces an anomaly score. To a warehouse supervisor, that means nothing without a substantial trust-building exercise and training programme. Jumping straight to ML sacrifices explainability and the operational trust that makes monitoring systems actually used.

The correct approach is additive: SPC as the daily operational layer that the ops team owns and acts on, ML as a weekly diagnostic that the data science team reviews and escalates selectively. Over time, as the team builds intuition for what IF flags look like on real operational data, the ML layer can be promoted to daily cadence.

---

## 2. Why Western Electric Run Rules Over Simple Thresholds

A naive implementation would flag any day where pick_accuracy_rate drops below some fixed threshold — say 97%. This approach has two problems.

First, the threshold is arbitrary and context-free. A warehouse running at 99.5% baseline accuracy is in a very different control state than one running at 97.5%, even though both might occasionally breach a fixed 97% threshold.

Second, and more importantly, a single random point beyond any threshold happens by chance. Under a normal distribution, a measurement will exceed the 3σ boundary 0.27% of the time purely by chance — roughly once per year for a daily metric. A single such breach is not operationally meaningful.

Western Electric run rules detect patterns that are individually normal but collectively suspicious:
- Rule 2 (2 of 3 beyond 2σ) has a 0.004% false alarm rate vs 0.27% for a single point
- Rule 3 (8 consecutive same side) detects process shifts that may never breach even a 2σ limit
- Rule 4 (6-point trend) detects slow deterioration invisible to any threshold approach

The run rules dramatically reduce false alarm rates while detecting systematic process changes that matter operationally — a workforce running consistently 0.3% below their usual accuracy is a training problem, not a random bad day.

---

## 3. Why CUSUM for Gradual Drift

Consider a pick accuracy rate drifting downward by 0.05 percentage points per week due to, say, a training issue with a new cohort of operators. Starting from a 99.3% baseline:

- Week 1: 99.25% — well within 3σ
- Week 2: 99.20% — still normal
- Week 3: 99.15% — still normal
- Week 4: 99.10% — still normal
- Week 6: 99.0% — might barely breach a 2σ limit

A CUSUM chart with k=0.5, h=4.0 would flag this pattern by week 3–4. It does this by accumulating small deviations: each day that the metric is slightly below the reference value contributes a positive increment to C+. The sum grows even when individual deviations are sub-threshold.

This early detection is critical for accuracy-related issues, which are typically training- or process-driven. A two-week earlier detection of a systematic accuracy degradation at 500 tasks/day saves approximately 1,000 additional pick errors, each of which costs ~$25 to resolve. That is $25,000 in recoverable losses per early-detection event.

CUSUM parameters k and h require calibration on real data. The standard values (k=0.5σ, h=4-5σ) are appropriate starting points, but should be tuned against actual incident history within the first 6 months of production deployment.

---

## 4. Why Isolation Forest and LOF

Isolation Forest and Local Outlier Factor are philosophically different algorithms that complement each other.

**Isolation Forest** works by randomly partitioning the feature space into trees and measuring how many splits are needed to isolate a point. Anomalous points, being sparse and extreme, require fewer splits and receive higher anomaly scores. IF is efficient at scale and handles high-dimensional feature spaces well. It is a global method — it identifies points that are outliers relative to the entire dataset.

**Local Outlier Factor** compares the local density of a point to the densities of its k nearest neighbours. A point that is less dense than its neighbours (i.e., requires more space to find k neighbours) gets a higher outlier score. LOF is a local method — it detects points that are unusual relative to their immediate neighbourhood, even if they would appear normal in a global view.

For warehouse monitoring, this distinction matters. IF is well suited to detecting days that are extreme on the absolute scale — very low accuracy, unusual volume spikes. LOF is better suited to detecting unusual combinations — a day where accuracy is normal and volume is normal but the specific combination of those two with task duration is unusual for this warehouse at this time of year.

Requiring agreement between IF and LOF (33 days out of 2,190) dramatically increases confidence in a flag. Two algorithms using different mathematical foundations, independently agreeing that a day is anomalous, is strong evidence that something genuinely unusual occurred.

---

## 5. Why Cost and Explainability Are First-Class Criteria

A technically superior anomaly detector that is never acted upon delivers zero business value.

This is the central failure mode of data science deployments in operations: the team builds a sophisticated model, it gets deployed, operations supervisors don't understand the outputs, they stop trusting it, they stop reading the alerts, and the system is quietly decommissioned after 6 months. The warehouse is no safer than before, and the organisation becomes more sceptical of future analytics investments.

Cost matters for a different reason. Every false positive costs approximately $10 in supervisor investigation time (20 minutes at $30/hr loaded rate). SPC with its high false positive rate on stable data generates $8,748/year in investigation costs across 3 warehouses — almost entirely from Rule 3 firing on benign consecutive sequences. IF at 20% FP rate generates $1,320/year. This is not a trivial consideration in a cost-sensitive logistics environment.

The right framework for evaluating anomaly detectors in industrial settings is:

1. Does an ops supervisor understand the output and know what to do with it? (Explainability)
2. How often will it send them on a chase for nothing? (FP rate × investigation cost)
3. Will it still work if the underlying process shifts slightly? (Robustness)
4. Can we afford to run it every day at production scale? (Compute cost)
5. Does it actually catch real problems earlier than we would find them without it? (Recall on true anomalies)

ML optimised purely on (5) while ignoring (1)–(4) is not a deployable solution.

The hybrid architecture recommended here is designed to score well on all five dimensions simultaneously: SPC handles (1)–(3) at near-zero cost; ML handles (5) as a weekly diagnostic layer with acceptable FP rates and minimal compute burden.
