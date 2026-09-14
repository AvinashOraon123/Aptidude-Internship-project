import sqlite3
import pandas as pd

def create_questions_table(db_path='toy_data.db'):
    conn = sqlite3.connect(db_path)

    print("Extracting static labels from attempts...")
    # Extract unique questions and their static labels from the attempts table
    # Since our toy data assigned one static label per question_id, we just take the first occurrence
    df_questions = pd.read_sql("""
        SELECT question_id, section, difficulty
        FROM attempts
        GROUP BY question_id
    """, conn)

    # Rename 'difficulty' to 'static_label' to make it crystal clear for the user and Power BI
    df_questions = df_questions.rename(columns={'difficulty': 'static_label'})

    # Write to a new dedicated 'questions' table
    df_questions.to_sql('questions', conn, if_exists='replace', index=False)

    conn.close()
    print(f"Successfully created 'questions' table in {db_path} with static labels.")
    print(df_questions.head())

if __name__ == "__main__":
    create_questions_table()
