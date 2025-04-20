# src/preprocessing.py
import os
import pandas as pd
from sqlalchemy import create_engine
import config

PROCESSED = config.PROCESSED_DIR

def load_processed():
    users        = pd.read_csv(os.path.join(PROCESSED, "users.csv"))
    courses      = pd.read_csv(os.path.join(PROCESSED, "courses_processed.csv"))
    interactions = pd.read_csv(os.path.join(PROCESSED, "interactions.csv"),
                               parse_dates=["timestamp"])
    return users, courses, interactions

def filter_activity(interactions, min_user=5, min_item=5):
    u_counts = interactions["student_id"].value_counts()
    i_counts = interactions["course_id"].value_counts()
    keep_users = u_counts[u_counts >= min_user].index
    keep_items = i_counts[i_counts >= min_item].index
    filtered = interactions[
        interactions["student_id"].isin(keep_users) &
        interactions["course_id"].isin(keep_items)
    ]
    return filtered

def save_final(users, courses, interactions):
    users.to_csv(os.path.join(PROCESSED, "users_final.csv"), index=False)
    courses.to_csv(os.path.join(PROCESSED, "courses_final.csv"), index=False)
    interactions.to_csv(os.path.join(PROCESSED, "interactions_final.csv"), index=False)

if __name__ == "__main__":
    users_df, courses_df, interactions_df = load_processed()
    interactions_f = filter_activity(interactions_df)
    save_final(users_df, courses_df, interactions_f)
    print("✅ Filtered data saved to data/processed/")
