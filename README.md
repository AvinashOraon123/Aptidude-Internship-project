# AptiDude Learning Analytics — Difficulty Recalibration, Student Segmentation & Drop-off Prediction

## Overview
This project extends a prior descriptive analytics pipeline (data cleaning, EDA,
Power BI dashboard) built on 10K+ student interaction records from AptiDude, an
online aptitude practice platform. The goal was to move from descriptive reporting
("what happened") to predictive and prescriptive analytics ("what will happen,
and what should we do about it").

Four components were added:
1. **Elo-based difficulty recalibration** — testing whether static difficulty
   labels (Easy/Medium/Hard) match empirical student performance.
2. **Student segmentation** — clustering students into behavioral archetypes
   to personalize weak-area recommendations.
3. **Drop-off risk prediction** — an XGBoost classifier predicting which
   students are likely to disengage within 48 hours, with SHAP-based
   explanations to support targeted re-engagement.
4. **Next-best-question recommender** — using Elo-calibrated difficulty to
   target each student's ideal success-probability band.

See `docs/methodology.md` for full methodology and `docs/presentation_outline.md`
for the final presentation structure.

## Repo Structure

```
aptidude-learning-analytics/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                       # original exports (gitignored)
│   ├── interim/                   # cleaned but not feature-engineered
│   └── processed/                 # final feature tables ready for modeling
│
├── notebooks/                     # exploratory notebooks (one per component)
│
├── src/
│   ├── elo.py                     # Elo difficulty recalibration
│   ├── segmentation.py            # K-Means student segmentation
│   ├── dropoff_features.py        # leakage-safe feature/label builder
│   ├── dropoff_model.py           # XGBoost training/eval/SHAP
│   └── recommender.py             # next-best-question recommender
│
├── models/                        # saved model artifacts (gitignored)
│
├── outputs/
│   ├── figures/                   # exported plots for deck/report
│   └── tables/                    # exported CSVs (mislabeled questions, at-risk users)
│
├── dashboard/
│   ├── powerbi/                   # .pbix file
│   └── sql/
│       └── student_features.sql   # feature table feeding Power BI + model
│
├── tests/
│   └── test_dropoff_features.py   # leakage-guard + pipeline unit tests
│
└── docs/
    ├── methodology.md
    └── presentation_outline.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline (example order)

```bash
# 1. Difficulty recalibration
python src/elo.py

# 2. Student segmentation
python src/segmentation.py

# 3. Build leakage-safe training set for drop-off model
python src/dropoff_features.py

# 4. Train + evaluate drop-off model
python src/dropoff_model.py

# 5. Generate recommendations
python src/recommender.py

# Run tests
pytest tests/ -v
```

Each module also runs as a standalone smoke test with synthetic data when
executed directly (`if __name__ == "__main__"` blocks), so you can verify
the pipeline works before plugging in real AptiDude exports.

## Tech Stack
Python (Pandas, NumPy, scikit-learn, XGBoost, SHAP), SQL, Power BI

## Limitations & Next Steps
See `docs/methodology.md` Section 7 for full details:
- 2PL IRT considered as a more rigorous alternative to Elo — left as future work
- Segmentation not yet validated for stability across time windows
- Drop-off model trained on historical labels only — no live A/B test yet run
- Recommender scoped and built, but not yet validated with a live intervention test
