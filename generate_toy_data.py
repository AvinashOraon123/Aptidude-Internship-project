import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_toy_data(db_path='toy_data.db', num_users=10060, num_attempts=100000):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS attempts")
    cursor.execute("DROP TABLE IF EXISTS users")

    cursor.execute("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            signup_date DATE
        )
    """)

    cursor.execute("""
        CREATE TABLE attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question_id INTEGER,
            correct INTEGER,
            time_taken_sec REAL,
            attempted_at TIMESTAMP,
            section TEXT,
            difficulty TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # Global timeline
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 9, 14)
    total_days = (end_date - start_date).days

    # Generate Users
    users = []
    for uid in range(1, num_users + 1):
        signup_date = start_date + timedelta(days=random.randint(0, 300))
        users.append((uid, signup_date.date()))

    cursor.executemany("INSERT INTO users (user_id, signup_date) VALUES (?, ?)", users)

    # Setup Questions
    sections = ['Quant', 'Verbal', 'DI', 'Reasoning']
    difficulties = ['Easy', 'Medium', 'Hard']
    questions = []
    for qid in range(1, 501):
        sec = random.choice(sections)
        diff = random.choice(difficulties)
        if diff == 'Easy': actual_diff = random.uniform(0.1, 0.4)
        elif diff == 'Medium': actual_diff = random.uniform(0.4, 0.7)
        else: actual_diff = random.uniform(0.7, 0.9)
        if random.random() < 0.1: actual_diff = 1.0 - actual_diff
        questions.append({'id': qid, 'sec': sec, 'diff': diff, 'actual_diff': actual_diff})

    attempts = []

    # User Profiles
    for uid in range(1, num_users + 1):
        # 60% are "Loyalists" (active until the end), 40% are "Churners"
        is_loyalist = random.random() < 0.6
        skill = random.uniform(0.2, 0.8)
        rushing = random.random() < 0.2

        signup_date = start_date + timedelta(days=(users[uid-1][1] - start_date.date()).days)

        # Determine attempt count
        count = int(num_attempts / num_users * random.uniform(0.5, 1.5))

        # Distribute attempts
        if is_loyalist:
            # Spread attempts across the whole window up to end_date
            times = np.sort(np.random.uniform(
                signup_date.timestamp(),
                end_date.timestamp(),
                count
            ))
        else:
            # Stop activity well before end_date
            stop_date = signup_date + timedelta(days=random.randint(10, 200))
            if stop_date > end_date: stop_date = end_date
            times = np.sort(np.random.uniform(
                signup_date.timestamp(),
                stop_date.timestamp(),
                count
            ))

        for t in times:
            attempt_time = datetime.fromtimestamp(t)
            q = random.choice(questions)
            prob_correct = 1 / (1 + np.exp(-(skill - q['actual_diff']) * 5))
            correct = 1 if random.random() < prob_correct else 0

            base_time = 60 if q['diff'] == 'Easy' else (120 if q['diff'] == 'Medium' else 240)
            time_mult = 0.5 if rushing else random.uniform(0.8, 1.2)
            time_taken = max(5, base_time * time_mult + random.uniform(-20, 20))

            attempts.append((uid, q['id'], correct, time_taken, attempt_time, q['sec'], q['diff']))

    cursor.executemany("""
        INSERT INTO attempts (user_id, question_id, correct, time_taken_sec, attempted_at, section, difficulty)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, attempts)

    conn.commit()
    conn.close()
    print(f"Successfully created {db_path} with {num_users} users and {len(attempts)} attempts.")

if __name__ == "__main__":
    generate_toy_data()
