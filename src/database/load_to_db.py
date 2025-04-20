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
# Import the required models (schema now has updated User model)
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

    # --- Define renames: Parquet Column -> Model Attribute ---
    rename_map = {
        'id_student': 'student_id',
        'num_of_prev_attempts': 'num_prev_attempts' # Rename this specific column
        # Add other renames here if needed, e.g., if parquet used 'gender' and model used 'gender_mapped'
    }
    columns_to_rename = {k: v for k, v in rename_map.items() if k in users_df.columns}
    if columns_to_rename:
        print(f"Renaming Parquet columns for User model compatibility: {columns_to_rename}")
        users_df.rename(columns=columns_to_rename, inplace=True)

    # --- Define target model columns based on the NEW schema ---
    # These names must exactly match the attributes in the User class in schema.py
    user_model_col_names = [
        'student_id', 'region', 'studied_credits',
        'gender_mapped', 'highest_education_mapped', 'imd_band_mapped',
        'age_band_mapped', 'num_prev_attempts', 'disability_mapped'
    ]

    # Select columns from the dataframe that exist in our target model columns
    # This handles cases where parquet might have extra cols, or schema expects cols parquet doesn't have
    cols_to_load = [col for col in user_model_col_names if col in users_df.columns]
    print(f"Columns selected for User loading: {cols_to_load}")
    if set(cols_to_load) != set(user_model_col_names):
         # This is now just a warning, as missing columns in parquet might be acceptable (will be NULL in DB)
         print("Warning: Not all expected user model columns were found in the Parquet file!")
         print(f"Expected Schema Columns: {sorted(user_model_col_names)}")
         print(f"Found in Parquet & Loading: {sorted(cols_to_load)}")
         missing_cols = set(user_model_col_names) - set(cols_to_load)
         if missing_cols: print(f"Columns Missing in Parquet (will be NULL): {sorted(list(missing_cols))}")

    # Create the final DataFrame to load, containing only the selected columns
    users_load_df = users_df[cols_to_load]
    users_data = users_load_df.to_dict(orient='records')

    # --- Check dictionary keys ---
    if users_data:
        first_record_keys = set(users_data[0].keys())
        print(f"Sample keys in dict for User bulk insert: {first_record_keys}")
        # Check if the essential ID key is present after renaming
        if 'student_id' not in first_record_keys:
            print(f"CRITICAL ERROR: 'student_id' key missing in dictionary for User insertion!")
            return 0, set()

    try:
        session.bulk_insert_mappings(User, users_data)
        session.flush()
        loaded_count = len(users_data)
        # Get IDs from the correct column name ('student_id' after rename)
        loaded_user_ids = set(users_df['student_id'])
        print(f"Successfully flushed {loaded_count} records for 'users' table.")
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

    parquet_id_col = 'presentation_id'
    if parquet_id_col not in items_df.columns:
        print(f"Error: '{parquet_id_col}' column missing in items Parquet.")
        return 0, 0, set()

    # Ensure correct column names for splitting and model attributes
    items_df[['module_id', 'presentation_code']] = items_df[parquet_id_col].str.split('_', expand=True)

    # Populate Courses (unchanged)
    unique_modules = items_df['module_id'].unique()
    courses_data = [{'module_id': mod} for mod in unique_modules]
    existing_courses = {c.module_id for c in session.query(Course.module_id).all()}
    new_courses_data = [c for c in courses_data if c['module_id'] not in existing_courses]
    num_courses_added = 0
    if new_courses_data:
        try:
            session.bulk_insert_mappings(Course, new_courses_data)
            session.flush()
            num_courses_added = len(new_courses_data)
            print(f"Successfully flushed {num_courses_added} new records for 'courses' table.")
        except Exception as e: session.rollback(); print(f"An error occurred loading courses: {e}"); return 0, 0, set()
    else: print("No new courses to add.")

    # Populate Presentations (unchanged from previous corrected version)
    presentation_model_cols = ['module_id', 'presentation_code', 'module_presentation_length']
    if 'module_presentation_length' not in items_df.columns:
        print(f"Warning: 'module_presentation_length' missing in items_df. Ensure it exists or handle default.")
    cols_present = [col for col in presentation_model_cols if col in items_df.columns]
    print(f"Columns selected for Presentation loading: {cols_present}")
    presentations_load_df = items_df[cols_present].copy()
    presentations_data = presentations_load_df.to_dict(orient='records')

    try:
        session.bulk_insert_mappings(Presentation, presentations_data)
        session.flush()
        loaded_count = len(presentations_data)
        loaded_presentation_ids = set(items_df[parquet_id_col])
        print(f"Successfully flushed {loaded_count} records for 'presentations' table.")
        return num_courses_added, loaded_count, loaded_presentation_ids
    except IntegrityError as e: session.rollback(); print(f"IntegrityError loading presentations: {e}"); return num_courses_added, 0, set()
    except Exception as e: session.rollback(); print(f"An unexpected error occurred loading presentations: {e}"); return num_courses_added, 0, set()


def load_aggregated_interactions(session, file_path=INTERACTIONS_PATH):
    """Loads aggregated interaction data using bulk insert, ensuring student IDs exist."""
    print(f"Loading aggregated interactions from {file_path}...")
    if not file_path.exists(): print(f"Error: File not found: {file_path}"); return 0

    interactions_df = pd.read_parquet(file_path)
    print(f"Loaded {interactions_df.shape[0]} aggregated interaction records from Parquet.")
    if interactions_df.empty: print("Interactions Parquet file is empty."); return 0

    # Define column names (unchanged)
    parquet_student_id_col = 'id_student'
    parquet_item_id_col = 'presentation_id'
    model_student_id_col = 'student_id'

    if parquet_student_id_col not in interactions_df.columns:
        print(f"Error: Expected column '{parquet_student_id_col}' not found in interactions Parquet. Columns: {interactions_df.columns}")
        return 0
    if parquet_item_id_col not in interactions_df.columns:
        print(f"Error: Expected column '{parquet_item_id_col}' not found in interactions Parquet. Columns: {interactions_df.columns}")
        return 0

    # Query existing users from DB (User model uses 'student_id')
    existing_users = {u.student_id for u in session.query(User.student_id).all()}
    print(f"Found {len(existing_users)} existing users in database session.")

    # Filter DataFrame using the correct Parquet column name
    interactions_df_filtered = interactions_df[interactions_df[parquet_student_id_col].isin(existing_users)].copy()
    dropped = interactions_df.shape[0] - interactions_df_filtered.shape[0]
    if dropped > 0: print(f"Dropped {dropped} aggregated rows for student IDs not found in the database.")
    if interactions_df_filtered.empty: print("No interactions remaining after filtering against existing users."); return 0

    # Split presentation_id
    interactions_df_filtered[['module_id', 'presentation_code']] = (
        interactions_df_filtered[parquet_item_id_col]
        .str.split('_', expand=True)
    )

    # Prepare the dict records, mapping to model columns (unchanged logic, relies on correct column names)
    agg_interaction_model_cols = [col.name for col in AggregatedInteraction.__table__.columns if col.name != 'agg_interaction_id']
    rename_map = {parquet_student_id_col: model_student_id_col}
    current_cols_after_split = list(interactions_df_filtered.columns) # Get all available columns now
    cols_for_load_df = []
    for col in agg_interaction_model_cols:
        parquet_col_name = col # Default: model name matches intermediate df name
        if col == model_student_id_col:
            parquet_col_name = parquet_student_id_col # Special case for student ID

        if parquet_col_name in current_cols_after_split:
            cols_for_load_df.append(parquet_col_name)
        elif col in interactions_df_filtered.columns: # Check original name too, just in case
             cols_for_load_df.append(col)
        else:
            print(f"Warning: Model column '{col}' not found in source DataFrame for AggregatedInteraction. Skipping.")

    print(f"Columns selected for AggregatedInteraction loading: {cols_for_load_df}")
    load_df = interactions_df_filtered[cols_for_load_df].copy()
    load_df.rename(columns=rename_map, inplace=True) # Rename student ID to match model

    data = load_df.to_dict(orient='records')

    if data:
        first_record_keys = set(data[0].keys())
        print(f"Sample keys in dict for AggregatedInteraction bulk insert: {first_record_keys}")
        if model_student_id_col not in first_record_keys:
             print(f"CRITICAL ERROR: '{model_student_id_col}' key missing in dictionary for AggregatedInteraction insertion!")
             return 0

    # Bulk insert (unchanged)
    try:
        session.bulk_insert_mappings(AggregatedInteraction, data)
        session.flush()
        loaded_count = len(data)
        print(f"Successfully flushed {loaded_count} aggregated interactions.")
        return loaded_count
    except IntegrityError as e: session.rollback(); print(f"IntegrityError loading aggregated interactions: {e}"); return 0
    except Exception as e: session.rollback(); print(f"An unexpected error occurred loading aggregated interactions: {e}"); return 0


def main():
    """Loads all processed data from Parquet files into the PostgreSQL database using separate transactions."""
    start_time = time.time()
    print("--- Starting Database Loading Script ---")
    if not check_db_connection(): print("Database connection failed. Exiting."); sys.exit(1)

    total_users = 0; total_courses = 0; total_presentations = 0; total_agg_interactions = 0

    # Transaction 1: Load Users, Courses, Presentations (unchanged logic, uses updated functions)
    print("\n--- Starting Transaction 1: Users, Courses, Presentations ---")
    try:
        with session_scope() as session1:
            total_users, loaded_user_ids = load_users(session1)
            expected_users = pd.read_parquet(USERS_PATH).shape[0] if USERS_PATH.exists() else 0
            if total_users != expected_users:
                 print(f"Warning: Expected {expected_users} users, but loaded {total_users}.")
                 if total_users == 0 and expected_users > 0: raise Exception("User loading failed (loaded 0).")

            total_courses, total_presentations, loaded_presentation_ids = load_presentations(session1)
            expected_presentations = pd.read_parquet(ITEMS_PATH).shape[0] if ITEMS_PATH.exists() else 0
            if total_presentations != expected_presentations:
                print(f"Warning: Expected {expected_presentations} presentations, but loaded {total_presentations}.")
                if total_presentations == 0 and expected_presentations > 0: raise Exception("Presentation loading failed (loaded 0).")

        print("--- Transaction 1 Committed Successfully ---")
    except Exception as e:
        print(f"\n--- FATAL ERROR during Transaction 1 ---"); print(e); sys.exit(1)

    # --- Post-Commit Check (unchanged) ---
    print("\n--- Performing check after Transaction 1 commit ---")
    try:
        with session_scope() as check_session:
            user_count_in_db = check_session.query(User).count()
            print(f"Found {user_count_in_db} users in DB after Transaction 1 commit (Expected: {total_users}).")
            if user_count_in_db != total_users:
                 print(f"CRITICAL FAILURE: Mismatch in user count after commit! Aborting.")
                 sys.exit(1)
            else:
                 print("SUCCESS: User count matches expected count after Transaction 1 commit.")
    except Exception as e:
         print(f"ERROR during post-commit check: {e}")
         sys.exit(1)

    # Transaction 2: Load Aggregated Interactions (unchanged logic, uses updated function)
    if total_users > 0 and total_presentations > 0:
        print("\n--- Starting Transaction 2: Aggregated Interactions ---")
        try:
            with session_scope() as session2:
                 total_agg_interactions = load_aggregated_interactions(session2)
                 expected_interactions = 0
                 interactions_df_temp = None
                 if INTERACTIONS_PATH.exists():
                     interactions_df_temp = pd.read_parquet(INTERACTIONS_PATH)
                     expected_interactions = interactions_df_temp.shape[0]

                 # Refined warning: Check if loaded count matches expected AFTER potential filtering
                 # (We need the 'dropped' count from inside the function, maybe return it?)
                 # For now, just compare total loaded vs original parquet size
                 if total_agg_interactions != expected_interactions:
                     # This might be okay if some users were genuinely missing, but 0 is suspicious
                     print(f"Warning: Expected {expected_interactions} interactions originally, but loaded {total_agg_interactions}.")
                     if total_agg_interactions == 0 and expected_interactions > 0:
                         # Check if the parquet was filtered to zero inside the function
                         if interactions_df_temp is not None:
                              # Re-query users loaded in TX1 *within this check scope* for accuracy
                              with session_scope() as check_session_users:
                                   users_in_db_now = {u.student_id for u in check_session_users.query(User.student_id).all()}
                              filtered_count_expected = interactions_df_temp[interactions_df_temp['id_student'].isin(users_in_db_now)].shape[0]
                              if filtered_count_expected > 0:
                                   print("CRITICAL WARNING: Expected interactions after filtering > 0, but loaded 0. Check loading logic.")
                              else:
                                   print("Note: All interactions were filtered out based on users loaded in Transaction 1.")


            print("--- Transaction 2 Committed Successfully ---")
        except Exception as e:
            print(f"\n--- ERROR during Transaction 2 ---"); print(e)
    else:
         print("\nSkipping Transaction 2 due to issues in Transaction 1.")

    # Final Summary (unchanged)
    print("\n--- Database Loading Summary ---")
    print(f" Users loaded: {total_users}"); print(f" Courses added: {total_courses}"); print(f" Presentations loaded: {total_presentations}"); print(f" Aggregated Interactions loaded: {total_agg_interactions}")
    end_time = time.time(); print(f"\n--- Database Loading Script Finished ---"); print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()