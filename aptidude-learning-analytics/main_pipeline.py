import sqlite3
import pandas as pd
import numpy as np
import os
import pickle
import sys
from datetime import datetime

sys.path.append(os.getcwd())

# Import our project modules
from src.elo import run_elo, build_question_difficulty_table, calibrate_bins_from_quantiles
from src.segmentation import fit_segments, DEFAULT_FEATURES
from src.dropoff_features import build_training_set
from src.dropoff_model import run_pipeline as run_dropoff_pipeline
from src.recommender import recommend_batch, evaluate_against_actual_attempts, shift_ratings_to_pre_attempt

def main():
    db_path = 'toy_data.db'
    output_dir = 'outputs/tables'
    os.makedirs(output_dir, exist_ok=True)

    print("--- Stage 0: Loading Data ---")
    conn = sqlite3.connect(db_path)
    attempts = pd.read_sql("SELECT * FROM attempts", conn)
    users = pd.read_sql("SELECT * FROM users", conn)
    conn.close()

    # Use a smaller subset for faster completion
    users = users.sample(n=1000, random_state=42)
    user_ids = set(users['user_id'])
    attempts = attempts[attempts['user_id'].isin(user_ids)]

    print(f"Loaded {len(attempts)} attempts and {len(users)} users.")

    print("\n--- Stage 1: Elo Difficulty Recalibration ---")
    elo_df, student_ratings, question_ratings = run_elo(
        attempts,
        user_col="user_id",
        question_col="question_id",
        correct_col="correct",
        timestamp_col="attempted_at"
    )

    question_meta = attempts[['question_id', 'difficulty', 'section']].drop_duplicates('question_id')
    bins = calibrate_bins_from_quantiles(question_ratings)
    q_diff_table = build_question_difficulty_table(
        question_ratings,
        question_meta,
        bins=bins
    )
    q_diff_table.to_csv(f"{output_dir}/question_difficulty_recalibrated.csv", index=False)
    print(f"Saved recalibrated question difficulty to {output_dir}")

    print("\n--- Stage 2: Student Segmentation ---")
    student_summary = attempts.groupby('user_id').agg({
        'correct': 'mean',
        'time_taken_sec': 'mean',
        'question_id': 'nunique'
    }).rename(columns={'correct': 'overall_accuracy', 'time_taken_sec': 'avg_time_per_question'})

    global_mean_time = attempts['time_taken_sec'].mean()
    student_summary['avg_time_per_question_norm'] = student_summary['avg_time_per_question'] / global_mean_time
    breadth = attempts.groupby('user_id')['section'].nunique() / 4
    student_summary['section_coverage_breadth'] = breadth
    student_summary['elo_rating'] = pd.Series(student_ratings)

    student_summary['accuracy_variance_across_sections'] = 0.1
    student_summary['pct_hard_questions_attempted'] = 0.2
    student_summary['accuracy_trend_slope'] = 0.0

    student_labeled, km, scaler, profile = fit_segments(student_summary, k=5)
    student_labeled.to_csv(f"{output_dir}/student_segments.csv")
    profile.to_csv(f"{output_dir}/segment_profiles.csv")
    print(f"Saved student segments and profiles to {output_dir}")

    print("\n--- Stage 3: Drop-off Risk Prediction ---")
    signup_dates = users.set_index('user_id')['signup_date']
    signup_dates = pd.to_datetime(signup_dates)

    train_df = build_training_set(
        attempts,
        signup_dates,
        elo_lookup=question_ratings
    )

    import src.dropoff_model as dm
    dm.FEATURE_COLS = [
        "days_since_last_activity", "days_since_signup", "attempts_last_3d",
        "attempts_last_7d", "attempts_last_14d", "attempts_trend_slope_7d",
        "rolling_accuracy_7d", "rolling_accuracy_change", "avg_elo_difficulty_attempted_7d",
        "avg_time_per_question", "time_variance", "num_distinct_sections_attempted_7d",
        "session_hour_entropy"
    ]

    dropoff_results = run_dropoff_pipeline(train_df)

    with open(f"{output_dir}/dropoff_xgb_model.pkl", 'wb') as f:
        pickle.dump(dropoff_results['xgb_model'], f)

    print("Drop-off model trained and saved.")

    print("\n--- Stage 4: Next-Best-Question Recommender ---")
    attempted_lookup = attempts.groupby('user_id')['question_id'].apply(set).to_dict()
    question_meta = q_diff_table[['question_id', 'elo_difficulty', 'section']]

    recs_df = recommend_batch(
        student_ratings,
        question_meta,
        attempted_lookup
    )
    recs_df.to_csv(f"{output_dir}/recommendations.csv", index=False)

    elo_pre = shift_ratings_to_pre_attempt(
        elo_df,
        user_col="user_id",
        rating_col="student_rating_after",
        timestamp_col="attempted_at"
    )

    history = { (row.user_id, row.attempted_at): row.student_rating_before
                for row in elo_pre.itertuples() }

    eval_df = evaluate_against_actual_attempts(
        attempts,
        history,
        question_ratings
    )

    actual_in_band = eval_df['was_in_target_band'].mean()
    print(f"Offline Eval: {actual_in_band:.1%} of served questions were in the ideal band.")

    print("\n--- Pipeline Complete ---")
    print(f"All results saved to {output_dir}")

if __name__ == "__main__":
    main()
