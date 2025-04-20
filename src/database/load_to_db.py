# src/database/load_to_db.py

import pandas as pd
import sys
from pathlib import Path
import time
from sqlalchemy.exc import IntegrityError
import numpy as np

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
# Import session_scope, engine, check_db_connection
from src.database.db_utils import session_scope, engine, check_db_connection
# Import the required models
from src.database.schema import User, Course, Presentation, AggregatedInteraction

# Define paths to processed Parquet files
USERS_PATH = config.PROCESSED_DATA_DIR / "users_final.parquet"
ITEMS_PATH = config.PROCESSED_DATA_DIR / "items_final.parquet"
INTERACTIONS_PATH = config.PROCESSED_DATA_DIR / "interactions_final.parquet"


# --- load_users function (Revert to simpler version, remove debug check) ---
def load_users(session, file_path=USERS_PATH):
    """Loads user data from Parquet file into the users table."""
    print(f"Loading users from {file_path}...")
    if not file_path.exists(): print(f"Error: File not found: {file_path}"); return 0, set()
    users_df = pd.read_parquet(file_path)
    print(f"Loaded {users_df.shape[0]} user feature records from Parquet.")
    if users_df.empty: print("Users Parquet file is empty."); return 0, set()

    users_data = users_df.to_dict(orient='records')
    try:
        session.bulk_insert_mappings(User, users_data)
        session.flush() # Flush inside session_scope ensures it's part of the transaction
        loaded_count = len(users_data)
        loaded_user_ids = set(users_df['id_student'])
        print(f"Successfully flushed {loaded_count} records for 'users' table.")
        return loaded_count, loaded_user_ids
    except IntegrityError as e: print(f"IntegrityError loading users: {e}"); session.rollback(); return 0, set()
    except Exception as e: print(f"An unexpected error occurred loading users: {e}"); session.rollback(); return 0, set()

# --- load_presentations function (Revert to simpler version) ---
def load_presentations(session, file_path=ITEMS_PATH):
    """Loads presentation (item) data and ensures courses exist."""
    print(f"Loading presentations (items) from {file_path}...")
    if not file_path.exists(): print(f"Error: File not found: {file_path}"); return 0, 0, set()
    items_df = pd.read_parquet(file_path)
    print(f"Loaded {items_df.shape[0]} presentation feature records from Parquet.")
    if items_df.empty: print("Items Parquet file is empty."); return 0, 0, set()
    if 'presentation_id' not in items_df.columns: print("Error: 'presentation_id' column missing"); return 0, 0, set()

    items_df[['module_id', 'presentation_code']] = items_df['presentation_id'].str.split('_', expand=True)
    # Populate Courses
    unique_modules = items_df['module_id'].unique(); courses_data = [{'module_id': mod} for mod in unique_modules]
    existing_courses = {c.module_id for c in session.query(Course.module_id).all()}
    new_courses_data = [c for c in courses_data if c['module_id'] not in existing_courses]; num_courses_added = 0
    if new_courses_data:
        try:
            session.bulk_insert_mappings(Course, new_courses_data); session.flush()
            num_courses_added = len(new_courses_data); print(f"Successfully flushed {num_courses_added} new records for 'courses' table.")
        except Exception as e: session.rollback(); print(f"An error occurred loading courses: {e}"); return 0, 0, set()
    else: print("No new courses to add.")
    # Populate Presentations
    presentation_cols = ['module_id', 'presentation_code', 'module_presentation_length']
    presentations_df = items_df[presentation_cols].copy(); presentations_data = presentations_df.to_dict(orient='records')
    try:
        session.bulk_insert_mappings(Presentation, presentations_data); session.flush()
        loaded_count = len(presentations_data); loaded_presentation_ids = set(items_df['presentation_id'])
        print(f"Successfully flushed {loaded_count} records for 'presentations' table.")
        return num_courses_added, loaded_count, loaded_presentation_ids
    except IntegrityError as e: session.rollback(); print(f"IntegrityError loading presentations: {e}"); return num_courses_added, 0, set()
    except Exception as e: session.rollback(); print(f"An unexpected error occurred loading presentations: {e}"); return num_courses_added, 0, set()

# --- load_aggregated_interactions function (Revert to bulk insert) ---
def load_aggregated_interactions(session, file_path=INTERACTIONS_PATH):
    """Loads aggregated interaction data using bulk insert, but only for students already loaded."""
    print(f"Loading aggregated interactions from {file_path}...")
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 0

    # 1) Read the Parquet
    interactions_df = pd.read_parquet(file_path)
    print(f"Loaded {interactions_df.shape[0]} aggregated interaction records from Parquet.")
    if interactions_df.empty:
        print("Interactions Parquet file is empty.")
        return 0

    # 2) Pull the set of student_ids we just loaded into users
    existing_users = {u.student_id for u in session.query(User.student_id).all()}
    # 3) Filter out any interactions for students we didn’t load
    before = interactions_df.shape[0]
    interactions_df = interactions_df[interactions_df['student_id'].isin(existing_users)]
    dropped = before - interactions_df.shape[0]
    if dropped:
        print(f"Dropped {dropped} aggregated rows for unknown student_id(s).")

    # 4) Split presentation_id back into module_id and presentation_code
    interactions_df[['module_id', 'presentation_code']] = (
        interactions_df['presentation_id']
        .str.split('_', expand=True)
    )

    # 5) Prepare the dict records
    load_df = interactions_df[[
        'student_id', 'module_id', 'presentation_code',
        'total_clicks', 'interaction_days',
        'first_interaction_date', 'last_interaction_date',
        'implicit_feedback'
    ]].copy()
    load_df.rename(columns={'student_id': 'student_id'}, inplace=True)
    data = load_df.to_dict(orient='records')

    # 6) Bulk‐insert
    try:
        session.bulk_insert_mappings(AggregatedInteraction, data)
        session.flush()
        print(f"Successfully loaded {len(data)} aggregated interactions.")
        return len(data)
    except IntegrityError as e:
        session.rollback()
        print(f"IntegrityError loading aggregated interactions: {e}")
        return 0

# --- main function with explicit check between transactions ---
def main():
    """Loads all processed data from Parquet files into the PostgreSQL database using separate transactions."""
    start_time = time.time()
    print("--- Starting Database Loading Script ---")
    if not check_db_connection(): print("Database connection failed. Exiting."); sys.exit(1)

    total_users = 0; total_courses = 0; total_presentations = 0; total_agg_interactions = 0

    # Transaction 1: Load Users, Courses, Presentations
    print("\n--- Starting Transaction 1: Users, Courses, Presentations ---")
    try:
        with session_scope() as session1:
            total_users, loaded_user_ids = load_users(session1)
            if total_users == 0 and USERS_PATH.exists(): raise Exception("User loading failed...")
            total_courses, total_presentations, loaded_presentation_ids = load_presentations(session1)
            if total_presentations == 0 and ITEMS_PATH.exists(): raise Exception("Presentation loading failed...")
        print("--- Transaction 1 Committed Successfully ---")
    except Exception as e:
        print(f"\n--- FATAL ERROR during Transaction 1 ---"); print(e); sys.exit(1)

    # --- Explicit check AFTER commit of Transaction 1 ---
    print("\n--- Performing check after Transaction 1 commit ---")
    problematic_user_id = 6516 # Check the ID from the latest error
    try:
        with session_scope() as check_session:
            user_check = check_session.query(User).filter(User.student_id == problematic_user_id).first()
            if user_check:
                 print(f"SUCCESS: User {problematic_user_id} FOUND in DB after Transaction 1 commit.")
            else:
                 print(f"FAILURE: User {problematic_user_id} NOT FOUND in DB after Transaction 1 commit. Aborting.")
                 # If this happens, the problem is very deep (DB config, driver, silent failure?)
                 sys.exit(1)
    except Exception as e:
         print(f"ERROR during post-commit check: {e}")
         sys.exit(1)
    # --- End explicit check ---

    # Transaction 2: Load Aggregated Interactions
    if total_users > 0 and total_presentations > 0:
        print("\n--- Starting Transaction 2: Aggregated Interactions ---")
        try:
            with session_scope() as session2:
                 total_agg_interactions = load_aggregated_interactions(session2) # Use bulk insert version
                 if total_agg_interactions == 0 and INTERACTIONS_PATH.exists():
                     print("Warning: Aggregated interaction loading finished but loaded 0 records.")
            print("--- Transaction 2 Committed Successfully ---")
        except Exception as e:
            print(f"\n--- ERROR during Transaction 2 ---"); print(e)
    else:
         print("\nSkipping Transaction 2 due to issues in Transaction 1.")

    # Final Summary
    print("\n--- Database Loading Summary ---")
    print(f" Users loaded: {total_users}"); print(f" Courses added: {total_courses}"); print(f" Presentations loaded: {total_presentations}"); print(f" Aggregated Interactions loaded: {total_agg_interactions}")
    end_time = time.time(); print(f"\n--- Database Loading Script Finished ---"); print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    print("INFO: To clear tables before loading, manually drop/recreate them using psql or modify this script.")
    print("--- Run TRUNCATE TABLE aggregated_interactions, presentations, users, courses RESTART IDENTITY CASCADE; in psql before executing ---")
    main()