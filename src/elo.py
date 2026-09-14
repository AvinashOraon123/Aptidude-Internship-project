"""
Elo-based question difficulty recalibration for AptiDude interaction data.
"""
import pandas as pd
import numpy as np


def run_elo(
    df: pd.DataFrame,
    k_student: float = 32,
    k_question: float = 16,
    init_rating: float = 1200,
    user_col: str = "user_id",
    question_col: str = "question_id",
    correct_col: str = "correct",
    timestamp_col: str = "timestamp",
):
    """
    Sequentially updates Elo ratings for students and questions based on
    chronologically ordered attempts.

    Returns
    -------
    df_out : DataFrame with two new columns: student_rating_after, question_rating_after
    student_ratings : dict {user_id: final_rating}
    question_ratings : dict {question_id: final_rating}
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    student_r = {}
    question_r = {}
    student_hist = np.empty(len(df))
    question_hist = np.empty(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        u = getattr(row, user_col)
        q = getattr(row, question_col)
        s = getattr(row, correct_col)

        ru = student_r.get(u, init_rating)
        rq = question_r.get(q, init_rating)

        expected = 1 / (1 + 10 ** ((rq - ru) / 400))

        ru_new = ru + k_student * (s - expected)
        rq_new = rq + k_question * ((1 - s) - (1 - expected))

        student_r[u] = ru_new
        question_r[q] = rq_new

        student_hist[i] = ru_new
        question_hist[i] = rq_new

    df_out = df.copy()
    df_out["student_rating_after"] = student_hist
    df_out["question_rating_after"] = question_hist

    return df_out, student_r, question_r


def label_from_elo(rating: float, bins=(0, 1150, 1450, np.inf),
                    labels=("Easy", "Medium", "Hard")) -> str:
    """Map a single Elo rating to a difficulty tier. Calibrate bins from your data's quantiles."""
    return pd.cut([rating], bins=bins, labels=labels)[0]


def build_question_difficulty_table(
    question_ratings: dict,
    question_meta: pd.DataFrame,
    question_col: str = "question_id",
    static_label_col: str = "difficulty",
    bins=(0, 1150, 1450, np.inf),
    labels=("Easy", "Medium", "Hard"),
) -> pd.DataFrame:
    """
    Joins Elo ratings onto question metadata and flags mislabeled questions.
    """
    elo_series = pd.Series(question_ratings, name="elo_difficulty")
    out = question_meta.merge(elo_series, left_on=question_col, right_index=True, how="left")

    out["elo_label"] = pd.cut(out["elo_difficulty"], bins=bins, labels=labels)
    out["mislabeled"] = out["elo_label"].astype(str) != out[static_label_col].astype(str)

    return out


def calibrate_bins_from_quantiles(question_ratings: dict, labels=("Easy", "Medium", "Hard")):
    """
    Helper to pick bin edges from the empirical distribution instead of hardcoding.
    Splits ratings into len(labels) equal-frequency buckets.
    """
    ratings = pd.Series(list(question_ratings.values()))
    quantiles = np.linspace(0, 1, len(labels) + 1)
    bins = ratings.quantile(quantiles).values.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    return bins


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 2000
    synth = pd.DataFrame({
        "user_id": rng.integers(0, 50, n),
        "question_id": rng.integers(0, 100, n),
        "correct": rng.integers(0, 2, n),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
    })
    df_out, s_r, q_r = run_elo(synth)
    print("Sample student ratings:", dict(list(s_r.items())[:5]))
    print("Sample question ratings:", dict(list(q_r.items())[:5]))
