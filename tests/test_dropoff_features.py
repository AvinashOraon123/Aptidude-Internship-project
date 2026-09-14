"""
Unit tests for src/dropoff_features.py

Focus areas:
1. No feature computation uses data beyond the snapshot timestamp (the
   core leakage guard).
2. Labels correctly reflect future activity/inactivity.
3. Snapshot generation respects history/horizon constraints.
4. Edge cases (sparse users, single-attempt users) don't crash the pipeline.
"""
import pandas as pd
import numpy as np
import pytest

from src.dropoff_features import (
    generate_snapshot_dates,
    compute_label,
    compute_features_for_snapshot,
    build_training_set,
)


@pytest.fixture
def sample_attempts():
    """
    A small, hand-constructed attempts table where we know the ground
    truth: user 1 keeps a steady daily cadence, user 2 goes quiet after
    day 10 (simulating a genuine drop-off), user 3 has very sparse activity.
    """
    rows = []

    # User 1: active every day for 20 days
    for day in range(20):
        rows.append({
            "user_id": 1,
            "question_id": day % 5,
            "correct": 1 if day % 3 != 0 else 0,
            "time_taken_sec": 30 + day,
            "attempted_at": pd.Timestamp("2026-06-01") + pd.Timedelta(days=day),
            "section": "Quant",
        })

    # User 2: active days 0-9, then nothing (drop-off case)
    for day in range(10):
        rows.append({
            "user_id": 2,
            "question_id": day % 5,
            "correct": 1,
            "time_taken_sec": 45,
            "attempted_at": pd.Timestamp("2026-06-01") + pd.Timedelta(days=day),
            "section": "Verbal",
        })

    # User 3: only 2 attempts total, far apart (sparse / insufficient history)
    rows.append({
        "user_id": 3, "question_id": 1, "correct": 1, "time_taken_sec": 60,
        "attempted_at": pd.Timestamp("2026-06-01"), "section": "DI",
    })
    rows.append({
        "user_id": 3, "question_id": 2, "correct": 0, "time_taken_sec": 90,
        "attempted_at": pd.Timestamp("2026-06-25"), "section": "DI",
    })

    return pd.DataFrame(rows)


@pytest.fixture
def signup_dates(sample_attempts):
    return sample_attempts.groupby("user_id")["attempted_at"].min() - pd.Timedelta(days=1)


# ---------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------

def test_features_never_use_future_data(sample_attempts, signup_dates):
    """
    For an early snapshot (day 5), the computed `attempts_last_14d` etc.
    must only reflect attempts up to and including that snapshot -- never
    the later attempts that exist in the full dataset.
    """
    snapshot_ts = pd.Timestamp("2026-06-06")  # day 5, 0-indexed
    feats = compute_features_for_snapshot(
        sample_attempts, user_id=1, snapshot_ts=snapshot_ts,
        signup_date=signup_dates[1],
    )

    # Only days 0-5 (6 attempts) should be visible at this snapshot
    assert feats["attempts_last_14d"] == 6
    assert feats["days_since_last_activity"] == 0  # last attempt was on the snapshot day itself


def test_features_manual_recompute_matches_filtered_history(sample_attempts, signup_dates):
    """
    Cross-check: manually filter attempts <= snapshot_ts and recompute a
    couple of features by hand, confirming the function matches.
    """
    snapshot_ts = pd.Timestamp("2026-06-10")
    feats = compute_features_for_snapshot(
        sample_attempts, user_id=1, snapshot_ts=snapshot_ts,
        signup_date=signup_dates[1],
    )

    manual_hist = sample_attempts[
        (sample_attempts["user_id"] == 1) &
        (sample_attempts["attempted_at"] <= snapshot_ts)
    ]
    expected_avg_time = manual_hist["time_taken_sec"].mean()

    assert feats["avg_time_per_question"] == pytest.approx(expected_avg_time)


# ---------------------------------------------------------------------
# Label correctness
# ---------------------------------------------------------------------

def test_label_marks_dropoff_user_correctly(sample_attempts):
    """
    User 2's last attempt is day 9 (2026-06-10). A snapshot taken on day 9
    with a 2-day horizon should be labeled target=1 (no activity in the
    following 2 days), since we know from the fixture they never return.
    """
    snapshots = pd.DataFrame({
        "user_id": [2],
        "snapshot_ts": [pd.Timestamp("2026-06-10")],
    })
    labeled = compute_label(sample_attempts, snapshots, horizon_days=2)

    assert labeled.loc[0, "target"] == 1


def test_label_marks_active_user_correctly(sample_attempts):
    """
    User 1 is active every day through day 19. A snapshot on day 5 with a
    2-day horizon should be labeled target=0, since they have attempts on
    day 6 and day 7.
    """
    snapshots = pd.DataFrame({
        "user_id": [1],
        "snapshot_ts": [pd.Timestamp("2026-06-06")],
    })
    labeled = compute_label(sample_attempts, snapshots, horizon_days=2)

    assert labeled.loc[0, "target"] == 0


def test_label_handles_user_with_no_future_rows_in_data(sample_attempts):
    """
    A snapshot placed exactly at a user's last-ever attempt with no data
    beyond it should be labeled 1 (no observed future activity), not error out.
    """
    snapshots = pd.DataFrame({
        "user_id": [3],
        "snapshot_ts": [pd.Timestamp("2026-06-25")],
    })
    labeled = compute_label(sample_attempts, snapshots, horizon_days=2)

    assert labeled.loc[0, "target"] == 1


# ---------------------------------------------------------------------
# Snapshot generation constraints
# ---------------------------------------------------------------------

def test_snapshot_generation_respects_min_history(sample_attempts):
    """
    User 3 has only 2 attempts, 24 days apart. With min_history_days=7,
    the earliest valid snapshot should be at least 7 days after their
    first attempt -- never before.
    """
    snapshots = generate_snapshot_dates(
        sample_attempts, min_history_days=7, label_horizon_days=2, n_snapshots_per_user=3
    )
    user3_snaps = snapshots[snapshots["user_id"] == 3]

    first_attempt = sample_attempts.loc[sample_attempts["user_id"] == 3, "attempted_at"].min()
    if not user3_snaps.empty:
        assert (user3_snaps["snapshot_ts"] >= first_attempt + pd.Timedelta(days=7)).all()


def test_snapshot_generation_respects_label_horizon(sample_attempts):
    """
    No generated snapshot should sit within `label_horizon_days` of the
    dataset's max observed timestamp -- otherwise the label would be
    computed on a truncated, unreliable future window.
    """
    horizon = 2
    snapshots = generate_snapshot_dates(
        sample_attempts, min_history_days=5, label_horizon_days=horizon, n_snapshots_per_user=5
    )
    max_ts = sample_attempts["attempted_at"].max()

    assert (snapshots["snapshot_ts"] <= max_ts - pd.Timedelta(days=horizon)).all()


def test_sparse_user_may_be_excluded_without_crashing(sample_attempts):
    """
    User 3 has insufficient history/horizon margin for most settings --
    the pipeline should simply skip them, not raise.
    """
    snapshots = generate_snapshot_dates(
        sample_attempts, min_history_days=10, label_horizon_days=5, n_snapshots_per_user=3
    )
    # Should not raise; user 3 may or may not appear depending on margin
    assert isinstance(snapshots, pd.DataFrame)


# ---------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------

def test_build_training_set_runs_end_to_end(sample_attempts, signup_dates):
    """
    Full pipeline should produce a non-empty DataFrame with the expected
    feature columns and a binary target, with no NaNs in the target column.
    """
    training_set = build_training_set(
        sample_attempts, signup_dates,
        n_snapshots_per_user=3, min_history_days=5, label_horizon_days=2,
    )

    assert not training_set.empty
    assert set(training_set["target"].unique()).issubset({0, 1})
    assert training_set["target"].isna().sum() == 0
    assert "days_since_last_activity" in training_set.columns


def test_build_training_set_produces_multiple_rows_per_user(sample_attempts, signup_dates):
    """
    Confirms the multi-snapshot design actually multiplies rows per user
    (the whole point of avoiding single-snapshot leakage), not just one
    row per user.
    """
    training_set = build_training_set(
        sample_attempts, signup_dates,
        n_snapshots_per_user=4, min_history_days=3, label_horizon_days=2,
    )
    user1_rows = training_set[training_set["user_id"] == 1]

    assert len(user1_rows) > 1
