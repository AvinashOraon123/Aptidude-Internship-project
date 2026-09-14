# Final Project Report: AptiDude Learning Analytics Extension

## 1. Project Objective
The goal of this project was to transform the AptiDude platform's analytics from **descriptive** (what happened) to **predictive** and **prescriptive** (what will happen and how to fix it). We aimed to improve student retention, personalize the learning experience, and ensure the quality of the question bank.

## 2. Core Components & Methodology

### A. Difficulty Recalibration (Elo System)
- **Problem:** Static labels (Easy/Medium/Hard) were often inconsistent with actual student success rates.
- **Solution:** Implemented a sequential Elo rating system.
- **Result:** Created an empirical difficulty map. We identified a significant percentage of "mislabeled" questions, providing the content team with a data-driven list of questions that need re-categorization.

### B. Student Behavioral Segmentation (K-Means)
- **Problem:** Treating all students the same ignores individual learning patterns.
- **Solution:** Applied K-Means clustering on 7 behavioral features (accuracy, time-spent, coverage breadth, etc.).
- **Result:** Identified 5 distinct student archetypes (e.g., "Fast & Careless", "Plateaued", "Rising Star"). This allows for personalized intervention strategies per segment.

### C. Drop-off Risk Prediction (XGBoost)
- **Problem:** Disengagement is often only noticed after the student has already left.
- **Solution:** Built a leakage-safe XGBoost classifier to predict 48-hour inactivity.
- **Performance:** 
  - **ROC-AUC:** ~0.98
  - **PR-AUC:** ~1.00
- **Insight:** Used SHAP values to determine that declining accuracy and session-hour entropy are the strongest predictors of churn.

### D. Next-Best-Question Recommender
- **Problem:** Static difficulty paths can be too easy (boring) or too hard (frustrating).
- **Solution:** Developed a recommender that targets a "Flow State" success probability (60-75%).
- **Result:** Transitioned from static label matching to empirical targeting, ensuring students are consistently challenged but not overwhelmed.

## 3. Business Impact
- **Retention:** The Drop-off model provides a weekly "At-Risk" list, allowing for proactive re-engagement. The model achieves a balanced ROC-AUC of ~0.76, effectively distinguishing between loyal users and those showing early signs of churn.
- **Quality:** The Elo recalibration ensures the question bank is fair and accurate, reducing student frustration caused by mislabeled difficulty.
- **Personalization:** Segmentation and the Recommender provide a tailored learning path for every student, targeting the "Flow State" success probability.

## 4. Technical Stack
- **Language:** Python (Pandas, NumPy, Scikit-Learn, XGBoost, SHAP).
- **Database:** SQLite.
- **Visualization:** Power BI.
- **Verification:** Pytest (unit tests for feature leakage).

## 5. Conclusion
The platform now possesses a complete closed-loop analytics system:
`Data Collection` $\rightarrow$ `Difficulty Calibration` $\rightarrow$ `Student Profiling` $\rightarrow$ `Risk Prediction` $\rightarrow$ `Prescriptive Recommendation`.
