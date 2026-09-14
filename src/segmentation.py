"""
K-Means student segmentation for personalized weak-area recommendations.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DEFAULT_FEATURES = [
    "overall_accuracy",
    "avg_time_per_question_norm",
    "accuracy_variance_across_sections",
    "elo_rating",
    "pct_hard_questions_attempted",
    "section_coverage_breadth",
    "accuracy_trend_slope",
]


def prepare_features(student_df: pd.DataFrame, features=DEFAULT_FEATURES):
    X = student_df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def select_k(X_scaled, k_range=range(2, 9), random_state=42):
    """Returns a DataFrame of inertia + silhouette score per k, for elbow/silhouette plots."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
        })
    return pd.DataFrame(results)


def fit_segments(student_df: pd.DataFrame, k: int, features=DEFAULT_FEATURES, random_state=42):
    """
    Fits KMeans and returns the student_df with a `segment` column,
    plus the fitted model, scaler, and a profile table of cluster means.
    """
    X_scaled, scaler = prepare_features(student_df, features)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    student_df = student_df.copy()
    student_df["segment"] = km.fit_predict(X_scaled)

    profile = student_df.groupby("segment")[features].mean().round(2)
    profile["n_students"] = student_df.groupby("segment").size()

    return student_df, km, scaler, profile


def assign_new_students(new_df: pd.DataFrame, km: KMeans, scaler: StandardScaler,
                         features=DEFAULT_FEATURES):
    """Score new/unseen students against an already-fit model (for daily scoring jobs)."""
    X = new_df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    X_scaled = scaler.transform(X)
    new_df = new_df.copy()
    new_df["segment"] = km.predict(X_scaled)
    return new_df


SEGMENT_LABELS_TEMPLATE = {
    # Fill in after inspecting `profile` from fit_segments() -- example mapping only
    0: "Fast & Careless",
    1: "Slow & Accurate",
    2: "Plateaued",
    3: "Rising Star",
    4: "Avoider",
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    synth = pd.DataFrame({
        "overall_accuracy": rng.uniform(0.3, 0.95, n),
        "avg_time_per_question_norm": rng.uniform(0.5, 2.0, n),
        "accuracy_variance_across_sections": rng.uniform(0, 0.3, n),
        "elo_rating": rng.normal(1200, 150, n),
        "pct_hard_questions_attempted": rng.uniform(0, 0.5, n),
        "section_coverage_breadth": rng.uniform(0.2, 1.0, n),
        "accuracy_trend_slope": rng.normal(0, 0.02, n),
    })

    k_scores = select_k(prepare_features(synth)[0])
    print(k_scores)

    labeled_df, km, scaler, profile = fit_segments(synth, k=5)
    print(profile)
