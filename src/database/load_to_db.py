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
# Import session_scope AND the engine directly for checking connection
from src.database.db_utils import session_scope, engine, check_db_connection
from src.database.schema import User, Course, Presentation, AggregatedInteraction

# Define paths to processed Parquet files
USERS_PATH = config.PROCESSED_DATA_DIR / "users_final.parquet"
ITEMS_PATH = config.PROCESSED_DATA_DIR / "items_final.parquet"
INTERACTIONS_PATH = config.PROCESSED_DATA_DIR / "interactions_final.parquet"


def load_users(session, file_path=USERS_PATH):
    """Loads user data from Parquet file into the users table."""
    print(f"Loading users from {file_path}...")
    if not file_path.exists(): print(f"Error: File not found: {file_path}"); return 0, set()
    users_df = pd.read_parquet(file_path)
    print(f"Loaded {users_df.shape[0]} user feature records from Parquet.")
    if users_df.empty: print("Users Parquet file is empty."); return 0, set()

    problematic_id = 25572 # Keep check for info
    if problematic_id in users_df['id_student'].values:
        print(f"INFO: Problematic user ID {problematic_id} IS present in users_final.parquet.")
    else:
        print(f"WARNING: Problematic user ID {problematic_id} is NOT present in users_final.parquet.")

    users_data = users_df.to_dict(orient='records')
    try:
        session.bulk_insert_mappings(User, users_data)
        # Flush inside session_scope ensures it's part of the transaction
        session.flush()
        loaded_count = len(users_data)
        loaded_user_ids = set(users_df['id_student'])
        print(f"Successfully flushed {loaded_count} records for 'users' table.")

        # Optional: Keep the debug check if you want confirmation
        # print(f"DEBUG: Querying DB for user {problematic_id} immediately after flush...")
        # user_check = session.query(User).filter(User.student_id == problematic_id).first()
        # if user_check: print(f"DEBUG: User {problematic_id} FOUND in DB within session after flush.")
        # else: print(f"DEBUG: User {problematic_id} NOT FOUND in DB within session after flush!")

        return loaded_count, loaded_user_ids
    except IntegrityError as e: print(f"IntegrityError loading users: {e}"); session.rollback(); return 0, set()
    except Exception as e: print(f"An unexpected error occurred loading users: {e}"); session.rollback(); return 0, set()


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
    presentations_df = items_df[presentation_cols].copy()
    presentations_data = presentations_df.to_dict(orient='records')
    try:
        session.bulk_insert_mappings(Presentation, presentations_data); session.flush()
        loaded_count = len(presentations_data); loaded_presentation_ids = set(items_df['presentation_id'])
        print(f"Successfully flushed {loaded_count} records for 'presentations' table.")
        return num_courses_added, loaded_count, loaded_presentation_ids
    except IntegrityError as e: session.rollback(); print(f"IntegrityError loading presentations: {e}"); return num_courses_added, 0, set()
    except Exception as e: session.rollback(); print(f"An unexpected error occurred loading presentations: {e}"); return num_courses_added, 0, set()


def load_aggregated_interactions(session, file_path=INTERACTIONS_PATH):
    """Loads aggregated interaction data."""
    print(f"Loading aggregated interactions from {file_path}...")
    if not file_path.exists(): print(f"Error: File not found: {file_path}"); return 0
    interactions_df = pd.read_parquet(file_path)
    print(f"Loaded {interactions_df.shape[0]} aggregated interaction records from Parquet.")
    if interactions_df.empty: print("Interactions Parquet file is empty."); return 0
    if 'presentation_id' not in interactions_df.columns: print("Error: 'presentation_id' column missing"); return 0
    if 'id_student' not in interactions_df.columns: print("Error: 'id_student' column missing"); return 0

    interactions_df[['module_id', 'presentation_code']] = interactions_df['presentation_id'].str.split('_', expand=True)
    interactions_load_df = interactions_df[['id_student', 'module_id', 'presentation_code', 'total_clicks','interaction_days', 'first_interaction_date', 'last_interaction_date','implicit_feedback']].copy()
    interactions_load_df.rename(columns={'id_student': 'student_id'}, inplace=True)
    interactions_data = interactions_load_df.to_dict(orient='records')
    try:
        session.bulk_insert_mappings(AggregatedInteraction, interactions_data); session.flush()
        loaded_count = len(interactions_data)
        print(f"Successfully loaded {loaded_count} records into 'aggregated_interactions' table.")
        return loaded_count
    except IntegrityError as e: session.rollback(); print(f"IntegrityError loading aggregated interactions: {e}"); return 0
    except Exception as e: session.rollback(); print(f"An unexpected error occurred loading aggregated interactions: {e}"); return 0


def main():
    """Loads all processed data from Parquet files into the PostgreSQL database using separate transactions."""
    start_time = time.time()
    print("--- Starting Database Loading Script ---")
    if not check_db_connection(): print("Database connection failed. Exiting."); sys.exit(1)

    total_users = 0; total_courses = 0; total_presentations = 0; total_agg_interactions = 0

    # --- Transaction 1: Load Users, Courses, Presentations ---
    print("\n--- Starting Transaction 1: Users, Courses, Presentations ---")
    try:
        with session_scope() as session1:
            # Load users first
            total_users, loaded_user_ids = load_users(session1)
            if total_users == 0 and USERS_PATH.exists():
                raise Exception("User loading failed or produced no users, aborting.")

            # Load presentations (and courses) next
            total_courses, total_presentations, loaded_presentation_ids = load_presentations(session1)
            if total_presentations == 0 and ITEMS_PATH.exists():
                raise Exception("Presentation loading failed or produced no presentations, aborting.")

        print("--- Transaction 1 Committed Successfully ---")

    except Exception as e:
        # Error occurred in the first transaction, session_scope handles rollback
        print(f"\n--- FATAL ERROR during Transaction 1 (Users/Presentations) ---"); print(e)
        print("Aborting further loading.")
        sys.exit(1)

    # --- Transaction 2: Load Aggregated Interactions ---
    # Only proceed if users and presentations were loaded successfully
    if total_users > 0 and total_presentations > 0:
        print("\n--- Starting Transaction 2: Aggregated Interactions ---")
        try:
            with session_scope() as session2:
                 total_agg_interactions = load_aggregated_interactions(session2)
                 if total_agg_interactions == 0 and INTERACTIONS_PATH.exists():
                     # This might happen if FK constraints still fail for some reason, even after commit
                     print("Warning: Aggregated interaction loading finished but loaded 0 records.")

            print("--- Transaction 2 Committed Successfully ---")

        except Exception as e:
            # Error occurred in the second transaction
            print(f"\n--- ERROR during Transaction 2 (Aggregated Interactions) ---"); print(e)
            # Continue to summary, but interaction count will be 0
    else:
         print("\nSkipping Transaction 2 (Aggregated Interactions) due to issues in Transaction 1.")


    # --- Final Summary ---
    print("\n--- Database Loading Summary ---")
    print(f" Users loaded: {total_users}")
    print(f" Courses added: {total_courses}")
    print(f" Presentations loaded: {total_presentations}")
    print(f" Aggregated Interactions loaded: {total_agg_interactions}") # Will be 0 if Transaction 2 failed

    end_time = time.time()
    print(f"\n--- Database Loading Script Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    print("INFO: To clear tables before loading, manually drop/recreate them using psql or modify this script.")
    print("--- Run TRUNCATE TABLE aggregated_interactions, presentations, users, courses RESTART IDENTITY CASCADE; in psql before executing ---")
    main()