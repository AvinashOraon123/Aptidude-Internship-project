import sqlite3
import pandas as pd
import os

def export_raw_data_to_csv(db_path='toy_data.db', output_dir='outputs/tables'):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Export users
    print("Exporting users table...")
    users_df = pd.read_sql("SELECT * FROM users", conn)
    users_df.to_csv(f"{output_dir}/users.csv", index=False)

    # Export attempts
    print("Exporting attempts table...")
    attempts_df = pd.read_sql("SELECT * FROM attempts", conn)
    attempts_df.to_csv(f"{output_dir}/attempts.csv", index=False)

    # Export drop off risk
    print("Exporting dropoff_risk table...")
    risk_df = pd.read_sql("SELECT * FROM dropoff_risk", conn)
    risk_df.to_csv(f"{output_dir}/dropoff_risk.csv", index=False)

    conn.close()
    print(f"Successfully exported raw data and risk table to {output_dir}")

if __name__ == "__main__":
    export_raw_data_to_csv()
