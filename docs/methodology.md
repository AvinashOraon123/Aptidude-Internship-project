# Methodology — AptiDude Learning Analytics Extension

## 1. Background & Scope

Prior work on this project delivered:
- Cleaning and preprocessing of 10K+ student interaction records (Python, Pandas, NumPy)
- Exploratory analysis of accuracy and rating trends across section, difficulty, and time filters
- An interactive Power BI dashboard surfacing performance trends and weak areas

This phase extends that foundation from **descriptive** analytics into
**predictive and prescriptive** analytics, adding four components that
share a single feature/data pipeline:

1. Elo-based difficulty recalibration
2. K-Means student segmentation
3. XGBoost-based drop-off (48h inactivity) prediction, with SHAP explainability
4. Next-best-question recommender

## 2. Data

**Source table: `attempts`**
| Column | Type | Description |
|---|---|---|
| user_id | int | Student identifier |
| question_id | int | Question identifier |
| correct | int (0/1) | Attempt outcome |
| time_taken_sec | float | Time spent on the question |
| attempted_at | timestamp | Attempt time |
| section | string | e.g. Quant, Verbal, DI, Reasoning |
| difficulty | string | Platform's static label: Easy/Medium/Hard |

**Source table: `users`**
| Column | Type | Description |
|---|---|---|
| user_id | int | Student identifier |
| signup_date | date | Account creation date |

Cleaning steps (carried over from prior phase): deduplication of
repeated submissions, timestamp normalization to a single timezone,
removal of records with null `user_id`/`question_id`, and outlier
capping on `time_taken_sec` (values >99th percentile treated as
likely idle-tab artifacts rather than genuine solve time).

## 3. Component 1 — Elo-Based Difficulty Recalibration

**Motivation:** static Easy/Medium/Hard labels are set once and never
revisited, but the EDA phase showed accuracy varying substantially
*within* a single labeled tier — suggesting labels don't track actual
empirical difficulty.

**Method:** sequential Elo updates over chronologically ordered
attempts (see `src/elo.py`):

```
E_student = 1 / (1 + 10^((R_question - R_student) / 400))
R_student_new  = R_student  + K_student  * (correct - E_student)
R_question_new = R_question + K_question * ((1-correct) - (1-E_student))
```

- `K_student = 32`, `K_question = 16` (questions move slower than
  students; more attempts are needed to stabilize a question's rating)
- Ratings initialized at 1200, or seeded from the static label
  (Easy=1000, Medium=1300, Hard=1600) to reduce early-attempt noise

**Validation:** final question Elo ratings are bucketed into tiers
using empirical quantiles (`calibrate_bins_from_quantiles` in
`elo.py`) rather than arbitrary fixed cutoffs, then compared against
the static label to flag mislabeled questions.

**Limitation:** Elo assumes attempt order reflects a meaningful
learning signal; bulk-imported historical attempts with unreliable
timestamps would distort convergence. A 2PL IRT model (`girth`
library) was considered as a more statistically rigorous alternative
and is left as a documented next step.

## 4. Component 2 — Student Segmentation

**Motivation:** the existing dashboard surfaces weak areas in
aggregate; it doesn't distinguish *why* a student is weak (rushing?
avoiding hard questions? plateaued?), which limits how actionable the
insight is.

**Method:** K-Means clustering (`src/segmentation.py`) on 7
standardized behavioral features: overall accuracy, normalized average
time per question, cross-section accuracy variance, Elo rating (from
Component 1), percentage of hard questions attempted, section coverage
breadth, and 14-day accuracy trend slope.

- k selected via elbow (inertia) + silhouette score across k=2–8
- Clusters manually profiled and labeled based on centroid values,
  then spot-checked against individual student trajectories to sanity
  check the automated labels

**Limitation:** cluster stability across different time windows
(e.g. would clusters formed on last month's data agree with this
month's?) was not formally tested — a re-clustering stability check is
a natural follow-up before using segments for automated interventions.

## 5. Component 3 — Drop-off Risk Prediction

**Motivation:** most tracked accounts show minimal or zero streak
length, and the community forum has explicit user complaints that
progress/engagement isn't visible to them. A forward-looking model can
flag at-risk users before they disengage, rather than only reporting
disengagement after the fact.

**Label:** 1 if a user has zero attempts in the 48 hours following a
given reference ("snapshot") timestamp, else 0.

**Leakage prevention (`src/dropoff_features.py`):** rather than
computing features once "as of today," multiple historical snapshot
dates are generated per user, each requiring:
- at least 7 days of prior history (so features aren't computed on
  near-empty activity)
- at least a full label horizon of *possible* future observation
  after the snapshot (so the label is well-defined, not
  cut off by the edge of the dataset)

All features for a given snapshot are computed strictly from
`attempted_at <= snapshot_ts` — verified in `tests/test_dropoff_features.py`.

**Features:** recency (days since last activity, days since signup),
volume trend (3d/7d/14d attempt counts, 7-day slope), accuracy trend
(rolling 7-day accuracy and week-over-week change), difficulty
exposure (average Elo rating of recently attempted questions), time
behavior (average time per question, time variance), section breadth,
and session-timing entropy (irregular hours as a disengagement
signal).

**Model:** XGBoost classifier, with a logistic regression baseline
(recency + accuracy-change only) reported alongside it for comparison.
Class imbalance handled via `scale_pos_weight`.

**Evaluation:** given the expected class imbalance (most users are
active on any given day), PR-AUC and **recall at a fixed precision
threshold** (70%) are reported as the primary metrics, since these map
more directly to an intervention use case ("if we act only on
high-confidence predictions, what fraction of true churners do we
catch?") than raw accuracy.

**Explainability:** SHAP values computed on the held-out test set to
identify the top global drivers of risk and to support per-user
explanations (e.g. "flagged primarily due to declining 7-day
accuracy, not just inactivity").

## 6. Component 4 — Next-Best-Question Recommender

**Motivation:** with Elo-calibrated difficulty available (Component 1),
the platform can move from static-label-based question selection to
targeting each student's ideal challenge level.

**Method:** for each student, candidate questions are scored by how
close their Elo-derived expected success probability falls to a target
band (default 60–75%) — "hard but achievable" rather than a difficulty
tier match. Falls back gracefully for unseen students (average-skill
assumption) and when no in-band candidates exist (returns the
closest-to-band candidates instead of nothing).

**Offline evaluation:** rather than requiring a live A/B test,
historical attempts are re-scored under the Elo model using each
student's rating *before* that attempt (to avoid leaking the attempt's
own outcome into its evaluation), to estimate what fraction of
previously served questions would have fallen inside the target band
versus what Elo-based targeting would achieve.

**Limitation:** this offline comparison estimates hypothetical
improvement, not measured outcomes — a live intervention test remains
the natural validation step (see Section 7).

## 7. Integration

Elo ratings, segment labels, risk scores, and recommendations are
written back to dedicated output tables (see `dashboard/sql/`) and
joined into the existing Power BI data model, adding:
- A page comparing static vs. Elo-derived question difficulty
- A segment filter applied to existing accuracy/weak-area visuals
- A weekly "at-risk users" table with top SHAP-driven risk factors
- A "recommended next question" field per student

## 8. Limitations & Future Work

- Elo/IRT: 2PL IRT left as future work for more rigorous difficulty
  and discrimination estimation
- Segmentation: no formal stability testing across time windows
- Drop-off model: trained and evaluated on historical labels only; no
  live intervention (e.g. push notification) A/B test was conducted to
  confirm a causal reduction in drop-off — this would be the natural
  next phase to validate real-world impact
- Recommender: offline-evaluated only; a live intervention test would
  be needed to confirm actual engagement/accuracy improvements
