# src/load_to_db.py
import os
import pandas as pd
from sqlalchemy import create_engine
import config

PROCESSED = config.PROCESSED_DIR
engine = create_engine(config.SQLALCHEMY_DATABASE_URI)

def load_table(filename, table_name):
    df = pd.read_csv(os.path.join(PROCESSED, filename),
                     parse_dates=["timestamp"] if "interactions" in filename else None)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"✅ Loaded {filename} → {table_name}")

if __name__ == "__main__":
    load_table("users_final.csv", "users")
    load_table("courses_final.csv", "courses")
    load_table("interactions_final.csv", "interactions")
