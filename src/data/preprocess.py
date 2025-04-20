# src/data/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
from src.data import load_raw
from src.data import utils

# --- Cleaning Functions (Keep as they were) ---
def clean_student_info(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the studentInfo dataframe."""
    print("Cleaning studentInfo data...")
    df_clean = df.copy()
    df_clean['imd_band'] = df_clean['imd_band'].fillna('Missing')
    print(f"Filled {df_clean['imd_band'].isnull().sum()} missing imd_band values with 'Missing'.")
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['final_result'], inplace=True)
    if initial_rows > df_clean.shape[0]:
        print(f"Dropped {initial_rows - df_clean.shape[0]} rows with missing final_result.")
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned studentInfo shape: {df_clean.shape}")
    return df_clean

def clean_registrations(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the studentRegistration dataframe."""
    print("Cleaning studentRegistration data...")
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['date_registration'], inplace=True)
    dropped_count = initial_rows - df_clean.shape[0]
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing date_registration.")
    df_clean['date_registration'] = df_clean['date_registration'].astype(int)
    df_clean['is_unregistered'] = df_clean['date_unregistration'].notna()
    df_clean['date_unregistration'] = df_clean['date_unregistration'].fillna(-999).astype(int)
    print("Created 'is_unregistered' flag and filled NaNs in 'date_unregistration'.")
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned studentRegistration shape: {df_clean.shape}")
    return df_clean

def clean_assessments(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the assessments dataframe."""
    print("Cleaning assessments data...")
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['date'], inplace=True)
    dropped_count = initial_rows - df_clean.shape[0]
    if dropped_count > 0:
        print(f"Dropped {dropped_count} assessments with missing date (deadline).")
    df_clean['date'] = df_clean['date'].astype(int)
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned assessments shape: {df_clean.shape}")
    return df_clean

def clean_student_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the studentAssessment dataframe."""
    print("Cleaning studentAssessment data...")
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['score'], inplace=True)
    dropped_count = initial_rows - df_clean.shape[0]
    if dropped_count > 0:
        print(f"Dropped {dropped_count} student assessment records with missing score.")
    df_clean['score'] = df_clean['score'].astype(int)
    df_clean['date_submitted'] = df_clean['date_submitted'].astype(int)
    df_clean['is_banked'] = df_clean['is_banked'].astype(bool)
    print(f"Cleaned studentAssessment shape: {df_clean.shape}")
    return df_clean

def clean_vle(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the vle dataframe."""
    print("Cleaning VLE data...")
    df_clean = df.copy()
    df_clean['week_from'] = df_clean['week_from'].fillna(-1).astype(int)
    df_clean['week_to'] = df_clean['week_to'].fillna(-1).astype(int)
    print("Filled missing 'week_from'/'week_to' with -1.")
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned VLE shape: {df_clean.shape}")
    return df_clean

def clean_student_vle(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the studentVle dataframe."""
    print("Cleaning studentVle data...")
    df_clean = df.copy()
    df_clean['date'] = df_clean['date'].astype(int)
    df_clean['sum_click'] = df_clean['sum_click'].astype(np.int32)
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned studentVle shape: {df_clean.shape}")
    return df_clean

# --- Filtering and Feature Engineering Functions ---

def filter_interactions_by_registration(student_vle_df: pd.DataFrame, registrations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters student VLE interactions to only include those that occurred
    while the student was actively registered for the corresponding presentation.
    """
    print("Filtering VLE interactions based on registration dates...")
    reg_dates = registrations_df[['id_student', 'presentation_id', 'date_registration', 'date_unregistration', 'is_unregistered']].copy()
    interactions_merged = pd.merge(
        student_vle_df,
        reg_dates,
        on=['id_student', 'presentation_id'],
        how='inner'
    )
    print(f"Merged interactions with registration info. Shape: {interactions_merged.shape}")
    cond_after_reg = interactions_merged['date'] >= interactions_merged['date_registration']
    cond_before_unreg = ~interactions_merged['is_unregistered'] | (interactions_merged['date'] < interactions_merged['date_unregistration'])
    interactions_filtered = interactions_merged[cond_after_reg & cond_before_unreg].copy()
    dropped_count = interactions_merged.shape[0] - interactions_filtered.shape[0]
    print(f"Filtered out {dropped_count} interactions falling outside registration periods.")
    print(f"Filtered interactions shape: {interactions_filtered.shape}")
    interactions_filtered.drop(columns=['date_registration', 'date_unregistration', 'is_unregistered'], inplace=True)
    return interactions_filtered


# --- NEW FILTERING FUNCTION (Applied *before* aggregation) ---
def apply_interaction_count_filters(df: pd.DataFrame,
                                    min_interactions_per_user: int = config.MIN_INTERACTIONS_PER_USER,
                                    min_users_per_item: int = config.MIN_USERS_PER_ITEM,
                                    user_col: str = 'id_student',
                                    item_col: str = 'presentation_id') -> pd.DataFrame:
    """
    Applies collaborative filtering style filters based on interaction counts:
    - Removes users with fewer than 'min_interactions_per_user' *interaction records*.
    - Removes items with fewer than 'min_users_per_item' *unique interacting users*.
    Iteratively applies filters until no more users/items are removed.

    Args:
        df (pd.DataFrame): DataFrame with interaction records (e.g., filtered student_vle).
                           Requires user_col, item_col.
        min_interactions_per_user (int): Minimum interaction records a user must have.
        min_users_per_item (int): Minimum unique users an item must have.
        user_col (str): Name of the user ID column.
        item_col (str): Name of the item ID column.

    Returns:
        pd.DataFrame: Filtered interactions DataFrame.
    """
    print(f"Applying interaction count filters (min_records_per_user={min_interactions_per_user}, min_users_per_item={min_users_per_item})...")
    df_filtered = df.copy()
    initial_rows = df_filtered.shape[0]

    while True:
        start_rows = df_filtered.shape[0]

        # Filter by minimum interactions per user
        user_interaction_counts = df_filtered.groupby(user_col).size()
        valid_users = user_interaction_counts[user_interaction_counts >= min_interactions_per_user].index
        df_filtered = df_filtered[df_filtered[user_col].isin(valid_users)]
        rows_after_user_filter = df_filtered.shape[0]
        print(f" Filter by user interaction count: {start_rows} -> {rows_after_user_filter} rows")

        if rows_after_user_filter == 0: break # Avoid errors if everything is filtered out

        # Filter by minimum users per item
        item_user_counts = df_filtered.groupby(item_col)[user_col].nunique()
        valid_items = item_user_counts[item_user_counts >= min_users_per_item].index
        df_filtered = df_filtered[df_filtered[item_col].isin(valid_items)]
        rows_after_item_filter = df_filtered.shape[0]
        print(f" Filter by item user count: {rows_after_user_filter} -> {rows_after_item_filter} rows")


        # Stop if no rows were removed in this iteration
        if df_filtered.shape[0] == start_rows:
            break

    final_rows = df_filtered.shape[0]
    print(f"Finished interaction count filtering. Removed {initial_rows - final_rows} interaction records.")
    print(f"Final filtered interactions shape before aggregation: {df_filtered.shape}")
    return df_filtered


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates interaction data (already filtered) to create implicit feedback features per student/presentation.
    """
    print("Creating aggregated interaction features (implicit feedback)...")
    if df.empty:
        print("Warning: Input DataFrame for aggregation is empty. Returning empty DataFrame.")
        # Define columns to match expected output schema, even if empty
        return pd.DataFrame(columns=[
            'id_student', 'presentation_id', 'total_clicks', 'interaction_days',
            'first_interaction_date', 'last_interaction_date', 'implicit_feedback'
        ])

    # Aggregate clicks per student per item (presentation) per day first
    # This step might be redundant if the input 'df' is already filtered granular interactions
    daily_interactions = df.groupby(['id_student', 'presentation_id', 'date'])['sum_click'].sum().reset_index()

    # Now aggregate per student and presentation
    user_item_interactions = daily_interactions.groupby(['id_student', 'presentation_id']).agg(
        total_clicks=('sum_click', 'sum'),
        interaction_days=('date', 'nunique'), # Count distinct days interacted
        first_interaction_date=('date', 'min'),
        last_interaction_date=('date', 'max')
    ).reset_index()

    # Create implicit feedback score - log(total_clicks + 1)
    user_item_interactions['implicit_feedback'] = np.log1p(user_item_interactions['total_clicks'])

    print(f"Created aggregated interaction features. Shape: {user_item_interactions.shape}")
    print(user_item_interactions.head())
    return user_item_interactions

# --- Feature Generation Functions (Keep as they were, but ensure they use IDs from final aggregated data) ---
def generate_user_features(student_info_clean: pd.DataFrame, valid_user_ids: np.ndarray) -> pd.DataFrame:
    """ Generates final user features table for users present in the final interaction set."""
    print("Generating user features...")
    # Filter student_info for valid users first
    student_info_filtered = student_info_clean[student_info_clean['id_student'].isin(valid_user_ids)].copy()
    if student_info_filtered.empty:
        print("Warning: No valid users found in student_info. Returning empty user features DataFrame.")
        return pd.DataFrame(index=pd.Index([], name='id_student'),
                            columns=['num_of_prev_attempts', 'studied_credits', 'gender_mapped',
                                     'highest_education_mapped', 'imd_band_mapped', 'age_band_mapped',
                                     'disability_mapped', 'region'])


    # Take the record corresponding to their *last* registration for consistency
    student_info_filtered = student_info_filtered.sort_values(by=['id_student', 'code_presentation'], ascending=[True, False])
    users_df = student_info_filtered.drop_duplicates(subset=['id_student'], keep='first')

    users_df = users_df[[
        'id_student', 'gender', 'region', 'highest_education', 'imd_band',
        'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
    ]].copy()

    # Apply mappings
    users_df['gender_mapped'] = users_df['gender'].apply(utils.map_gender)
    users_df['highest_education_mapped'] = users_df['highest_education'].apply(utils.map_highest_education)
    users_df['imd_band_mapped'] = users_df['imd_band'].apply(utils.map_imd_band)
    users_df['age_band_mapped'] = users_df['age_band'].apply(utils.map_age_band)
    users_df['disability_mapped'] = users_df['disability'].apply(utils.map_disability)

    users_final = users_df[[
        'id_student', 'num_of_prev_attempts', 'studied_credits',
        'gender_mapped', 'highest_education_mapped', 'imd_band_mapped',
        'age_band_mapped', 'disability_mapped', 'region'
    ]].set_index('id_student')

    print(f"Generated user features table. Shape: {users_final.shape}")
    print(users_final.head())
    return users_final

def generate_item_features(courses_df: pd.DataFrame, vle_clean: pd.DataFrame, valid_item_ids: np.ndarray) -> pd.DataFrame:
    """ Generates final item (presentation) features table for items present in the final interaction set. """
    print("Generating item (presentation) features...")
    # Filter courses and VLE for valid items first
    items_df = courses_df[courses_df['presentation_id'].isin(valid_item_ids)][['presentation_id', 'module_presentation_length']].copy()
    vle_filtered = vle_clean[vle_clean['presentation_id'].isin(valid_item_ids)].copy()

    if items_df.empty:
         print("Warning: No valid items found in courses_df. Returning empty item features DataFrame.")
         return pd.DataFrame(index=pd.Index([], name='presentation_id'),
                             columns=['module_presentation_length'])


    # Add VLE activity type features
    if not vle_filtered.empty:
        vle_counts = vle_filtered.groupby('presentation_id')['activity_type'].value_counts().unstack(fill_value=0)
        vle_proportions = vle_counts.apply(lambda x: x / x.sum(), axis=1)
        vle_proportions.columns = [f'vle_prop_{col}' for col in vle_proportions.columns]
        # Merge features
        items_final = pd.merge(items_df, vle_proportions, on='presentation_id', how='left')
    else:
        print("No VLE data for valid items. Item features will only contain presentation length.")
        items_final = items_df # Only length is available

    # Add assessment features here if needed, filtering assessments_clean by valid_item_ids

    # Fill NaNs and set index
    items_final.fillna(0, inplace=True)
    items_final = items_final.set_index('presentation_id')

    print(f"Generated item features table. Shape: {items_final.shape}")
    print(items_final.head())
    return items_final

# --- Main Preprocessing Pipeline ---
def preprocess_all_data() -> dict[str, pd.DataFrame]:
    """
    Runs the full preprocessing pipeline: load, clean, filter (interactions), aggregate, generate features.
    Returns a dictionary of final processed DataFrames ready for saving/loading to DB.
    """
    # 1. Load Raw Data
    raw_data = load_raw.load_all_raw_data()

    # 2. Clean individual DataFrames
    student_info_clean = clean_student_info(raw_data['student_info'])
    registrations_clean = clean_registrations(raw_data['student_registration'])
    # assessments_clean = clean_assessments(raw_data['assessments']) # Clean if needed for item features
    # student_assessment_clean = clean_student_assessment(raw_data['student_assessment']) # Clean if needed for interactions
    vle_clean = clean_vle(raw_data['vle'])
    student_vle_clean = clean_student_vle(raw_data['student_vle'])
    courses_df = utils.create_presentation_id(raw_data['courses'])

    # 3. Filter Interactions based on Registration Dates
    interactions_filtered_by_reg = filter_interactions_by_registration(
        student_vle_clean, registrations_clean
    )

    # --- !!! 4. Apply Interaction Count Filters (NEW STEP) !!! ---
    interactions_count_filtered = apply_interaction_count_filters(
        interactions_filtered_by_reg
        # Can adjust thresholds here or keep using config defaults
        # min_interactions_per_user=10,
        # min_users_per_item=10
    )

    # 5. Create Aggregated Interaction Features (Implicit Feedback)
    final_interactions = create_interaction_features(interactions_count_filtered)

    # --- REMOVED OLD FILTERING STEP ---
    # apply_activity_filters was here, now removed

    # Handle case where filtering removed all data
    if final_interactions.empty:
        print("Warning: All interactions were filtered out. Returning empty dataframes.")
        empty_users = pd.DataFrame(index=pd.Index([], name='id_student'))
        empty_items = pd.DataFrame(index=pd.Index([], name='presentation_id'))
        processed_data = {'users': empty_users, 'items': empty_items, 'interactions': final_interactions}
    else:
        # 6. Generate User Features (using only users present in final interactions)
        valid_user_ids = final_interactions['id_student'].unique()
        users_final = generate_user_features(student_info_clean, valid_user_ids)

        # 7. Generate Item Features (using only items present in final interactions)
        valid_item_ids = final_interactions['presentation_id'].unique()
        items_final = generate_item_features(courses_df, vle_clean, valid_item_ids)

        # 8. Prepare final dataframes
        # Ensure final_interactions only contains users/items present in features tables (should be guaranteed by filtering order)
        final_interactions = final_interactions[
            final_interactions['id_student'].isin(users_final.index) &
            final_interactions['presentation_id'].isin(items_final.index)
        ].reset_index(drop=True)

        processed_data = {
            'users': users_final,
            'items': items_final,
            'interactions': final_interactions
        }

    print("\n--- Preprocessing Finished ---")
    print(f"Final Users shape: {processed_data['users'].shape}")
    print(f"Final Items shape: {processed_data['items'].shape}")
    print(f"Final Interactions shape: {processed_data['interactions'].shape}")

    return processed_data


if __name__ == "__main__":
    processed_data = preprocess_all_data()