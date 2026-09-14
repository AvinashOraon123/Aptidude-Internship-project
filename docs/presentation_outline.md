# Final Internship Presentation — Slide Outline

Aim for ~10–12 slides, 10-minute talk. Structure: problem -> what you
built -> what you found -> what it's worth to the business.

**Slide 1 — Title**
Project name, your name, dates, mentor/team name.

**Slide 2 — Context & Problem**
- AptiDude platform overview (1 sentence)
- What existed before: cleaning + EDA + descriptive Power BI dashboard
- Gap identified: static difficulty labels untested against real
  performance; no personalization; no forward-looking signal on
  disengagement

**Slide 3 — Project Scope**
Four components, one sentence each (difficulty recalibration,
segmentation, drop-off prediction, recommender). This is your map for
the rest of the talk — refer back to it at each section transition.

**Slide 4 — Data**
- 10K+ interaction records, schema summary (table/columns)
- Cleaning steps (brief — this was prior work, don't over-spend time here)

**Slide 5 — Component 1: Difficulty Recalibration**
- One-line explanation of Elo update mechanic (simple diagram: two
  ratings, one update rule)
- Headline result: "% of questions mislabeled" chart (bar or table,
  static label vs. Elo tier)

**Slide 6 — Component 1: Impact**
- Example: 2–3 specific mislabeled questions (name them if possible)
  with before/after tier
- What this means for the product: informs question bank curation,
  fairer test difficulty balancing

**Slide 7 — Component 2: Student Segmentation**
- Elbow/silhouette chart (small, just to show rigor)
- Segment profile table with your 5 named segments + one defining
  stat each

**Slide 8 — Component 2: Impact**
- Segment sizes (pie or bar — "40% of active users fall into
  'Plateaued'")
- Recommended intervention per segment (this is the "so what")

**Slide 9 — Component 3: Drop-off Prediction**
- Problem framing: 48h inactivity prediction
- Model choice + why (XGBoost, handling imbalance)
- Headline metric: AUC + "at 70% precision, catches X% of at-risk
  users"

**Slide 10 — Component 3: Explainability**
- SHAP summary plot (this single visual usually lands best with
  non-technical stakeholders)
- Top 3 drivers translated into plain language

**Slide 11 — Component 4 & Dashboard Demo**
- Recommender logic in one sentence + offline evaluation result
  ("X% of served questions were in the ideal difficulty band vs. Y%
  under Elo-based targeting")
- Screenshot(s) of extended Power BI page(s)
- If live demo possible, do it here instead of slides

**Slide 12 — Limitations, Next Steps & Business Recommendation**
- Key caveats (from methodology.md Section 8)
- Concrete next step: e.g. "run a 2-week A/B test sending at-risk
  users a targeted drill nudge"
- One sentence tying back to business value: retention, question bank
  quality, personalization

Keep an appendix (not counted in the 12) with your Elo formula,
feature list tables, and model hyperparameters in case of technical
questions.
