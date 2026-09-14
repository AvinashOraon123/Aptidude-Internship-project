"""
Next-best-question recommender built on top of Elo-calibrated difficulty
(src/elo.py) and, optionally, student segments (src/segmentation.py).

Core idea: target a success probability band (e.g. ~60-75%) per student,
using the same Elo expected-score formula used to fit the ratings. This
keeps questions "hard but achievable" rather than relying on the static
Easy/Medium/Hard label -- which Component 1 already showed disagrees with
empirical difficulty for a meaningful fraction of questions.
"""
import pandas as pd
import numpy as np


DEFAULT_TARGET_PROB = (0.60, 0.75)  # desired success-probability band


def expected_success_prob(student_rating: float, question_rating: float) -> float:
    """Same Elo expectation formula used during rating fitting."""
    return 1 / (1 + 10 ** ((question_rating - student_rating) / 400))


def score_candidates(
    student_rating: float,
    candidate_questions: pd.DataFrame,
    question_rating_col: str = "elo_difficulty",
    target_band: tuple = DEFAULT_TARGET_PROB,
) -> pd.DataFrame:
    """
    Scores a pool of candidate questions for one student by how close
    their expected success probability is to the target band's midpoint.

    Returns candidate_questions with two new columns:
      - expected_prob: P(student solves this question)
      - band_distance:  how far outside the target band (0 if inside)
    sorted by best fit first.
    """
    out = candidate_questions.copy()
    out["expected_prob"] = out[question_rating_col].apply(
        lambda rq: expected_success_prob(student_rating, rq)
    )

    lo, hi = target_band
    midpoint = (lo + hi) / 2

    def band_distance(p):
        if lo <= p <= hi:
            return 0.0
        return min(abs(p - lo), abs(p - hi))

    out["band_distance"] = out["expected_prob"].apply(band_distance)
    out["midpoint_distance"] = (out["expected_prob"] - midpoint).abs()

    # Primary sort: inside-band candidates first (band_distance == 0),
    # then closest to midpoint as tiebreaker among in-band candidates,
    # then closest to the band edges among out-of-band candidates.
    out = out.sort_values(["band_distance", "midpoint_distance"])
    return out


def recommend_next_question(
    student_id,
    student_ratings: dict,
    question_meta: pd.DataFrame,
    attempted_question_ids: set,
    question_id_col: str = "question_id",
    question_rating_col: str = "elo_difficulty",
    section_filter: str = None,
    exclude_attempted: bool = True,
    target_band: tuple = DEFAULT_TARGET_PROB,
    n_recommendations: int = 1,
    fallback_rating: float = 1200,
):
    """
    Returns the top `n_recommendations` question(s) for a student.

    Falls back gracefully:
      - unseen student -> uses fallback_rating (assume average skill)
      - no in-band candidates -> returns closest-to-band candidates instead
        of returning nothing
    """
    student_rating = student_ratings.get(student_id, fallback_rating)

    pool = question_meta.copy()
    if section_filter is not None:
        pool = pool[pool["section"] == section_filter]
    if exclude_attempted:
        pool = pool[~pool[question_id_col].isin(attempted_question_ids)]

    if pool.empty:
        return pd.DataFrame()  # nothing left to recommend (e.g. exhausted a section)

    scored = score_candidates(student_rating, pool, question_rating_col, target_band)
    return scored.head(n_recommendations)


def recommend_batch(
    student_ratings: dict,
    question_meta: pd.DataFrame,
    attempted_lookup: dict,       # {student_id: set(question_ids attempted)}
    question_id_col: str = "question_id",
    question_rating_col: str = "elo_difficulty",
    target_band: tuple = DEFAULT_TARGET_PROB,
    n_recommendations: int = 1,
):
    """
    Generates recommendations for all students in `student_ratings` in one
    pass -- this is what a daily/batch scoring job would call, writing
    results back into a `recommendations` table for the app/dashboard to read.
    """
    rows = []
    for student_id, rating in student_ratings.items():
        attempted = attempted_lookup.get(student_id, set())
        recs = recommend_next_question(
            student_id, student_ratings, question_meta, attempted,
            question_id_col=question_id_col,
            question_rating_col=question_rating_col,
            target_band=target_band,
            n_recommendations=n_recommendations,
        )
        for _, r in recs.iterrows():
            rows.append({
                "user_id": student_id,
                "recommended_question_id": r[question_id_col],
                "expected_success_prob": round(r["expected_prob"], 3),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Offline evaluation: how would recommendations have differed from what
# the static system actually showed? This is the evidence for the
# "recalibrated targeting improves on static difficulty" claim.
# ---------------------------------------------------------------------

def evaluate_against_actual_attempts(
    attempts_df: pd.DataFrame,
    student_ratings_history: dict,   # {(user_id, timestamp): rating_at_that_time (BEFORE the attempt)}
    question_ratings: dict,          # {question_id: final elo_difficulty}
    target_band: tuple = DEFAULT_TARGET_PROB,
):
    """
    For each historical attempt, computes what the expected success
    probability *would have been* under the Elo model, and compares it
    to the actual outcome. This lets you report, e.g.:
      "Of the questions the static system actually served, only X% fell
       within the ideal 60-75% success-probability band; under Elo-based
       targeting, that rises to Y%."

    IMPORTANT: student_ratings_history must use the rating BEFORE each
    attempt (not after), or you leak the attempt's own outcome into its
    evaluation. If using run_elo()'s output, shift `student_rating_after`
    by one row per user to get the pre-attempt rating.

    Returns the attempts_df with an added `expected_prob_at_attempt` and
    `was_in_target_band` column for aggregate reporting.
    """
    df = attempts_df.copy()
    lo, hi = target_band

    def compute_row(row):
        rating_key = (row["user_id"], row["attempted_at"])
        student_r = student_ratings_history.get(rating_key, 1200)
        question_r = question_ratings.get(row["question_id"], 1200)
        return expected_success_prob(student_r, question_r)

    df["expected_prob_at_attempt"] = df.apply(compute_row, axis=1)
    df["was_in_target_band"] = df["expected_prob_at_attempt"].between(lo, hi)

    return df


def shift_ratings_to_pre_attempt(elo_output_df: pd.DataFrame, user_col: str = "user_id",
                                  rating_col: str = "student_rating_after",
                                  timestamp_col: str = "timestamp",
                                  init_rating: float = 1200):
    """
    Helper: converts run_elo()'s post-attempt rating column into a
    pre-attempt rating column per user (shifts by one row within each
    user's chronological attempt sequence). Use this before building
    student_ratings_history for evaluate_against_actual_attempts.
    """
    df = elo_output_df.sort_values([user_col, timestamp_col]).copy()
    df["student_rating_before"] = (
        df.groupby(user_col)[rating_col].shift(1).fillna(init_rating)
    )
    return df


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(3)

    student_ratings = {i: rng.normal(1200, 150) for i in range(20)}
    question_meta = pd.DataFrame({
        "question_id": range(100),
        "elo_difficulty": rng.normal(1200, 200, 100),
        "section": rng.choice(["Quant", "Verbal", "DI"], 100),
    })
    attempted_lookup = {i: set(rng.choice(100, size=10, replace=False)) for i in range(20)}

    recs = recommend_batch(student_ratings, question_meta, attempted_lookup, n_recommendations=2)
    print(recs.head(10))
