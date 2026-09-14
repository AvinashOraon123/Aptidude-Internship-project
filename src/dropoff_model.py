"""
Drop-off / 48h-inactivity risk prediction model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report
)
from xgboost import XGBClassifier
import shap

FEATURE_COLS = [
    "days_since_last_activity",
    "days_since_signup",
    "attempts_last_3d",
    "attempts_last_7d",
    "attempts_last_14d",
    "attempts_trend_slope_7d",
    "rolling_accuracy_7d",
    "rolling_accuracy_change",
    "avg_elo_difficulty_attempted_7d",
    "avg_time_per_question",
    "time_variance",
    "num_distinct_sections_attempted_7d",
    "session_hour_entropy",
]


def train_baseline(X_train, y_train):
    """Simple logistic regression baseline for comparison."""
    baseline_features = ["days_since_last_activity", "rolling_accuracy_change"]
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train[baseline_features], y_train)
    return model, baseline_features


def train_xgboost(X_train, y_train, params=None):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    default_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, feature_subset=None):
    X_eval = X_test[feature_subset] if feature_subset else X_test
    proba = model.predict_proba(X_eval)[:, 1]

    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    # Recall at fixed precision (e.g. 70%)
    target_precision = 0.70
    valid_idx = np.where(precision[:-1] >= target_precision)[0]
    recall_at_target = recall[valid_idx].max() if len(valid_idx) else None

    print(f"ROC-AUC: {auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")
    print(f"Recall at {target_precision:.0%} precision: {recall_at_target}")
    print(classification_report(y_test, (proba >= 0.5).astype(int)))

    return {"roc_auc": auc, "pr_auc": pr_auc, "recall_at_target_precision": recall_at_target,
            "proba": proba}


def explain(model, X_sample):
    """Returns SHAP values and can render a summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def score_all_users(model, X_current, id_col_series):
    """For a daily scoring job: returns a dataframe of user_id + risk_score, sorted descending."""
    proba = model.predict_proba(X_current)[:, 1]
    return pd.DataFrame({
        "user_id": id_col_series,
        "risk_score": proba
    }).sort_values("risk_score", ascending=False)


def run_pipeline(features_df: pd.DataFrame, label_col="target", feature_cols=FEATURE_COLS):
    X = features_df[feature_cols]
    y = features_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    baseline_model, baseline_features = train_baseline(X_train, y_train)
    print("=== Baseline (Logistic Regression) ===")
    evaluate(baseline_model, X_test, y_test, feature_subset=baseline_features)

    xgb_model = train_xgboost(X_train, y_train)
    print("\n=== XGBoost ===")
    results = evaluate(xgb_model, X_test, y_test)

    explainer, shap_values = explain(xgb_model, X_test)

    return {
        "baseline_model": baseline_model,
        "xgb_model": xgb_model,
        "results": results,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 3000
    synth = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLS})
    synth["target"] = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    out = run_pipeline(synth)
