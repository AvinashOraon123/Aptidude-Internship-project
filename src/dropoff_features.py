"""
Builds a leakage-safe, multi-snapshot training set for the drop-off model.

Instead of computing features only "as of today" (which gives one row per
user and risks the model just memorizing current state), this module picks
several historical reference dates per user and computes point-in-time
features + a forward-looking label for each. This multiplies the number of
training rows and forces the model to generalize across different points
in a user's lifecycle, not just their current snapshot.
"""
import pandas as pd
import numpy as np


def generate_snapshot_dates(
    attempts_df: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "attempted_at",
    n_snapshots_per_user: int = 5,
    min_history_days: int = 7,
    label_horizon_days: int = 2,
):
    """
    For each user, picks up to `n_snapshots_per_user` reference dates
    (the "as-of" point at which we pretend we're making the prediction),
    spaced across their activity history.

    Requires at least `min_history_days` of prior activity before a
    snapshot (so features aren't computed on near-zero history) and at
    least `label_horizon_days` of *possible* future observation after
    the snapshot (so the label itself is well-defined -- a snapshot too
    close to "today" in the raw data has no way to know if the user
    would have gone inactive, since we simply haven't observed far
    enough forward yet).

    Returns
    -------
    DataFrame with columns [user_id, snapshot_ts]
    """
    df = attempts_df[[user_col, timestamp_col]].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    max_observed_ts = df[timestamp_col].max()
    snapshots = []

    for user_id, group in df.groupby(user_col):
        first_ts = group[timestamp_col].min()
        last_possible_snapshot = max_observed_ts - pd.Timedelta(days=label_horizon_days)
        earliest_possible_snapshot = first_ts + pd.Timedelta(days=min_history_days)

        if earliest_possible_snapshot >= last_possible_snapshot:
            # Not enough history/future window for this user -- skip
            continue

        candidate_range = pd.date_range(
            earliest_possible_snapshot, last_possible_snapshot,
            periods=min(n_snapshots_per_user, max(2, (last_possible_snapshot - earliest_possible_snapshot).days))
        )

        for snap_ts in candidate_range:
            snapshots.append({user_col: user_id, "snapshot_ts": snap_ts})

    return pd.DataFrame(snapshots)


def compute_label(
    attempts_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "attempted_at",
    horizon_days: int = 2,
):
    """
    For each (user, snapshot_ts), label = 1 if the user has ZERO attempts
    in (snapshot_ts, snapshot_ts + horizon_days], else 0.
    """
    df = attempts_df[[user_col, timestamp_col]].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    labels = []
    # Group attempts per user once for faster repeated lookups
    grouped = {uid: g[timestamp_col].sort_values().to_numpy() for uid, g in df.groupby(user_col)}

    for row in snapshots_df.itertuples(index=False):
        uid = getattr(row, user_col)
        snap_ts = row.snapshot_ts
        window_end = snap_ts + pd.Timedelta(days=horizon_days)

        user_ts = grouped.get(uid, np.array([]))
        has_future_activity = np.any((user_ts > np.datetime64(snap_ts)) & (user_ts <= np.datetime64(window_end)))

        labels.append(0 if has_future_activity else 1)

    out = snapshots_df.copy()
    out["target"] = labels
    return out


def compute_features_for_snapshot(
    attempts_df: pd.DataFrame,
    user_id,
    snapshot_ts: pd.Timestamp,
    signup_date: pd.Timestamp,
    elo_lookup: dict = None,
):
    """
    Computes ALL features using only data with attempted_at <= snapshot_ts.
    This is the core leakage guard: nothing after snapshot_ts is touched.

    `elo_lookup`: optional {question_id: elo_rating} dict from elo.py,
    used to compute avg_elo_difficulty_attempted_7d.
    """
    hist = attempts_df[
        (attempts_df["user_id"] == user_id) &
        (attempts_df["attempted_at"] <= snapshot_ts)
    ]

    if hist.empty:
        return None  # shouldn't happen given min_history_days filter, but guard anyway

    def window(days):
        cutoff = snapshot_ts - pd.Timedelta(days=days)
        return hist[hist["attempted_at"] >= cutoff]

    w3, w7, w14 = window(3), window(7), window(14)
    w7_prior = hist[
        (hist["attempted_at"] >= snapshot_ts - pd.Timedelta(days=14)) &
        (hist["attempted_at"] < snapshot_ts - pd.Timedelta(days=7))
    ]

    # Attempts trend: linear slope of daily attempt counts over last 7 days
    daily_counts = w7.groupby(w7["attempted_at"].dt.date).size()
    if len(daily_counts) >= 2:
        x = np.arange(len(daily_counts))
        slope = np.polyfit(x, daily_counts.values, 1)[0]
    else:
        slope = 0.0

    rolling_acc_7d = w7["correct"].mean() if len(w7) else np.nan
    rolling_acc_prior_7d = w7_prior["correct"].mean() if len(w7_prior) else np.nan
    rolling_acc_change = (
        rolling_acc_7d - rolling_acc_prior_7d
        if pd.notna(rolling_acc_7d) and pd.notna(rolling_acc_prior_7d) else 0.0
    )

    avg_elo_7d = np.nan
    if elo_lookup is not None and len(w7):
        elo_vals = w7["question_id"].map(elo_lookup)
        avg_elo_7d = elo_vals.mean()

    # Session hour entropy (all history, not just recent window)
    hour_counts = hist["attempted_at"].dt.hour.value_counts(normalize=True)
    session_entropy = -(hour_counts * np.log2(hour_counts)).sum() if len(hour_counts) else 0.0

    features = {
        "days_since_last_activity": (snapshot_ts - hist["attempted_at"].max()).days,
        "days_since_signup": (snapshot_ts - signup_date).days,
        "attempts_last_3d": len(w3),
        "attempts_last_7d": len(w7),
        "attempts_last_14d": len(w14),
        "attempts_trend_slope_7d": slope,
        "rolling_accuracy_7d": rolling_acc_7d,
        "rolling_accuracy_change": rolling_acc_change,
        "avg_elo_difficulty_attempted_7d": avg_elo_7d,
        "avg_time_per_question": hist["time_taken_sec"].mean(),
        "time_variance": hist["time_taken_sec"].var(),
        "num_distinct_sections_attempted_7d": w7["section"].nunique(),
        "session_hour_entropy": session_entropy,
    }
    return features


def build_training_set(
    attempts_df: pd.DataFrame,
    signup_dates: pd.Series,      # indexed by user_id
    elo_lookup: dict = None,
    n_snapshots_per_user: int = 5,
    min_history_days: int = 7,
    label_horizon_days: int = 2,
):
    """
    Full pipeline: generate snapshots -> compute labels -> compute features
    -> return one row per (user, snapshot) ready for train/test split.
    """
    attempts_df = attempts_df.copy()
    attempts_df["attempted_at"] = pd.to_datetime(attempts_df["attempted_at"])

    snapshots = generate_snapshot_dates(
        attempts_df,
        n_snapshots_per_user=n_snapshots_per_user,
        min_history_days=min_history_days,
        label_horizon_days=label_horizon_days,
    )

    labeled = compute_label(attempts_df, snapshots, horizon_days=label_horizon_days)

    rows = []
    for row in labeled.itertuples(index=False):
        uid, snap_ts, target = row.user_id, row.snapshot_ts, row.target
        signup = signup_dates.get(uid, attempts_df["attempted_at"].min())

        feats = compute_features_for_snapshot(attempts_df, uid, snap_ts, signup, elo_lookup)
        if feats is None:
            continue
        feats["user_id"] = uid
        feats["snapshot_ts"] = snap_ts
        feats["target"] = target
        rows.append(feats)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(7)
    n_users = 30
    rows = []
    for uid in range(n_users):
        n_attempts = rng.integers(20, 150)
        start = pd.Timestamp("2026-06-01") + pd.Timedelta(days=int(rng.integers(0, 20)))
        # simulate irregular, sometimes-tapering activity
        ts = start + pd.to_timedelta(np.sort(rng.exponential(2, n_attempts).cumsum()), unit="D")
        for t in ts:
            rows.append({
                "user_id": uid,
                "question_id": int(rng.integers(0, 200)),
                "correct": int(rng.integers(0, 2)),
                "time_taken_sec": float(rng.uniform(10, 120)),
                "attempted_at": t,
                "section": rng.choice(["Quant", "Verbal", "DI", "Reasoning"]),
            })

    attempts_df = pd.DataFrame(rows)
    signup_dates = attempts_df.groupby("user_id")["attempted_at"].min() - pd.Timedelta(days=1)

    training_set = build_training_set(attempts_df, signup_dates, n_snapshots_per_user=4)
    print(training_set.shape)
    print(training_set.head())
    print("Target balance:\n", training_set["target"].value_counts(normalize=True))
