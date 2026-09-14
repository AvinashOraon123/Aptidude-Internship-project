-- =========================================================
-- student_features.sql
-- Builds the per-user feature table consumed by Power BI
-- and by the drop-off risk model. Assumes a base
-- `attempts` table: (user_id, question_id, correct,
-- time_taken_sec, attempted_at, section, difficulty)
-- and a `users` table: (user_id, signup_date).
--
-- NOTE: written for Postgres/SQL Server-style syntax
-- (DATEDIFF, INTERVAL, EXTRACT). Adjust for your actual
-- dialect if different.
--
-- NOTE: streak fields (current_streak_length, best_streak,
-- streak_break_count_past) and avg_elo_difficulty_attempted_7d
-- depend on your streak-tracking table and the Elo output
-- table (from src/elo.py) respectively -- join those in once
-- those schemas are confirmed.
-- =========================================================

WITH base AS (
    SELECT
        a.user_id,
        a.question_id,
        a.correct,
        a.time_taken_sec,
        a.attempted_at,
        a.section,
        a.difficulty,
        DATE(a.attempted_at) AS attempt_date
    FROM attempts a
),

-- Reference point: "now" for recency calculations (swap for a
-- fixed as-of date when generating historical training rows)
ref AS (
    SELECT CURRENT_TIMESTAMP AS ref_ts
),

recency AS (
    SELECT
        b.user_id,
        MAX(b.attempted_at) AS last_activity_ts,
        DATEDIFF(day, MAX(b.attempted_at), r.ref_ts) AS days_since_last_activity
    FROM base b CROSS JOIN ref r
    GROUP BY b.user_id, r.ref_ts
),

tenure AS (
    SELECT
        u.user_id,
        DATEDIFF(day, u.signup_date, r.ref_ts) AS days_since_signup
    FROM users u CROSS JOIN ref r
),

volume AS (
    SELECT
        b.user_id,
        SUM(CASE WHEN b.attempted_at >= r.ref_ts - INTERVAL '3 day'  THEN 1 ELSE 0 END) AS attempts_last_3d,
        SUM(CASE WHEN b.attempted_at >= r.ref_ts - INTERVAL '7 day'  THEN 1 ELSE 0 END) AS attempts_last_7d,
        SUM(CASE WHEN b.attempted_at >= r.ref_ts - INTERVAL '14 day' THEN 1 ELSE 0 END) AS attempts_last_14d
    FROM base b CROSS JOIN ref r
    GROUP BY b.user_id
),

accuracy AS (
    SELECT
        b.user_id,
        AVG(CASE WHEN b.attempted_at >= r.ref_ts - INTERVAL '7 day'  THEN b.correct * 1.0 END) AS rolling_accuracy_7d,
        AVG(CASE WHEN b.attempted_at >= r.ref_ts - INTERVAL '14 day'
                  AND b.attempted_at <  r.ref_ts - INTERVAL '7 day' THEN b.correct * 1.0 END) AS rolling_accuracy_prior_7d,
        AVG(b.time_taken_sec) AS avg_time_per_question,
        VAR_SAMP(b.time_taken_sec) AS time_variance
    FROM base b CROSS JOIN ref r
    GROUP BY b.user_id
),

sections AS (
    SELECT
        b.user_id,
        COUNT(DISTINCT CASE WHEN b.attempted_at >= r.ref_ts - INTERVAL '7 day' THEN b.section END) AS num_distinct_sections_attempted_7d
    FROM base b CROSS JOIN ref r
    GROUP BY b.user_id
),

hour_entropy AS (
    -- Shannon entropy of session hour-of-day distribution: irregular timing -> lower commitment
    SELECT
        user_id,
        -SUM(p * LOG(2, p)) AS session_hour_entropy
    FROM (
        SELECT
            user_id,
            EXTRACT(HOUR FROM attempted_at) AS hr,
            COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY user_id) AS p
        FROM base
        GROUP BY user_id, EXTRACT(HOUR FROM attempted_at)
    ) hourly
    GROUP BY user_id
)

SELECT
    rec.user_id,
    rec.days_since_last_activity,
    ten.days_since_signup,
    vol.attempts_last_3d,
    vol.attempts_last_7d,
    vol.attempts_last_14d,
    acc.rolling_accuracy_7d,
    (acc.rolling_accuracy_7d - acc.rolling_accuracy_prior_7d) AS rolling_accuracy_change,
    acc.avg_time_per_question,
    acc.time_variance,
    sec.num_distinct_sections_attempted_7d,
    he.session_hour_entropy
FROM recency rec
LEFT JOIN tenure ten     ON rec.user_id = ten.user_id
LEFT JOIN volume vol     ON rec.user_id = vol.user_id
LEFT JOIN accuracy acc   ON rec.user_id = acc.user_id
LEFT JOIN sections sec   ON rec.user_id = sec.user_id
LEFT JOIN hour_entropy he ON rec.user_id = he.user_id;
