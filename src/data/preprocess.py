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

# --- Cleaning Functions (No changes needed) ---
def clean_student_info(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning studentInfo data...") # ... (rest of function is unchanged) ...
    df_clean = df.copy()
    df_clean['imd_band'] = df_clean['imd_band'].fillna('Missing')
    # print(f"Filled {df_clean['imd_band'].isnull().sum()} missing imd_band values with 'Missing'.") # Less verbose
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['final_result'], inplace=True)
    if initial_rows > df_clean.shape[0]: print(f"Dropped {initial_rows - df_clean.shape[0]} rows with missing final_result.")
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned studentInfo shape: {df_clean.shape}")
    return df_clean

def clean_registrations(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning studentRegistration data...") # ... (rest of function is unchanged) ...
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['date_registration'], inplace=True)
    dropped_count = initial_rows - df_clean.shape[0]
    if dropped_count > 0: print(f"Dropped {dropped_count} rows with missing date_registration.")
    df_clean['date_registration'] = df_clean['date_registration'].astype(int)
    df_clean['is_unregistered'] = df_clean['date_unregistration'].notna()
    df_clean['date_unregistration'] = df_clean['date_unregistration'].fillna(-999).astype(int)
    # print("Created 'is_unregistered' flag and filled NaNs in 'date_unregistration'.") # Less verbose
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned studentRegistration shape: {df_clean.shape}")
    return df_clean

def clean_assessments(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning assessments data...") # ... (rest of function is unchanged) ...
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['date'], inplace=True)
    dropped_count = initial_rows - df_clean.shape[0]
    if dropped_count > 0: print(f"Dropped {dropped_count} assessments with missing date (deadline).")
    df_clean['date'] = df_clean['date'].astype(int)
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned assessments shape: {df_clean.shape}")
    return df_clean

def clean_student_assessment(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning studentAssessment data...") # ... (rest of function is unchanged) ...
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['score'], inplace=True)
    dropped_count = initial_rows - df_clean.shape[0]
    if dropped_count > 0: print(f"Dropped {dropped_count} student assessment records with missing score.")
    df_clean['score'] = df_clean['score'].astype(int)
    df_clean['date_submitted'] = df_clean['date_submitted'].astype(int)
    df_clean['is_banked'] = df_clean['is_banked'].astype(bool)
    print(f"Cleaned studentAssessment shape: {df_clean.shape}")
    return df_clean

def clean_vle(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning VLE data...") # ... (rest of function is unchanged) ...
    df_clean = df.copy()
    df_clean['week_from'] = df_clean['week_from'].fillna(-1).astype(int)
    df_clean['week_to'] = df_clean['week_to'].fillna(-1).astype(int)
    # print("Filled missing 'week_from'/'week_to' with -1.") # Less verbose
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned VLE shape: {df_clean.shape}")
    return df_clean

def clean_student_vle(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning studentVle data...") # ... (rest of function is unchanged) ...
    df_clean = df.copy()
    df_clean['date'] = df_clean['date'].astype(int)
    df_clean['sum_click'] = df_clean['sum_click'].astype(np.int32)
    df_clean = utils.create_presentation_id(df_clean)
    print(f"Cleaned studentVle shape: {df_clean.shape}")
    return df_clean

# --- Filtering and Feature Engineering Functions ---

def filter_interactions_by_registration(student_vle_df: pd.DataFrame, registrations_df: pd.DataFrame) -> pd.DataFrame:
    print("Filtering VLE interactions based on registration dates...") # ... (rest of function is unchanged) ...
    reg_dates = registrations_df[['id_student', 'presentation_id', 'date_registration', 'date_unregistration', 'is_unregistered']].copy()
    interactions_merged = pd.merge(student_vle_df, reg_dates, on=['id_student', 'presentation_id'], how='inner')
    # print(f"Merged interactions with registration info. Shape: {interactions_merged.shape}") # Less verbose
    cond_after_reg = interactions_merged['date'] >= interactions_merged['date_registration']
    cond_before_unreg = ~interactions_merged['is_unregistered'] | (interactions_merged['date'] < interactions_merged['date_unregistration'])
    interactions_filtered = interactions_merged[cond_after_reg & cond_before_unreg].copy()
    dropped_count = interactions_merged.shape[0] - interactions_filtered.shape[0]
    print(f"Filtered out {dropped_count} interactions falling outside registration periods.")
    interactions_filtered.drop(columns=['date_registration', 'date_unregistration', 'is_unregistered'], inplace=True)
    print(f"Filtered interactions shape: {interactions_filtered.shape}")
    return interactions_filtered

def apply_interaction_count_filters(df: pd.DataFrame, # ... (function definition is unchanged) ...
                                    min_interactions_per_user: int = config.MIN_INTERACTIONS_PER_USER,
                                    min_users_per_item: int = config.MIN_USERS_PER_ITEM,
                                    user_col: str = 'id_student',
                                    item_col: str = 'presentation_id') -> pd.DataFrame:
    print(f"Applying interaction count filters (min_records_per_user={min_interactions_per_user}, min_users_per_item={min_users_per_item})...") # ... (rest of function is unchanged) ...
    df_filtered = df.copy(); initial_rows = df_filtered.shape[0]
    while True:
        start_rows = df_filtered.shape[0]
        user_interaction_counts = df_filtered.groupby(user_col).size()
        valid_users = user_interaction_counts[user_interaction_counts >= min_interactions_per_user].index
        df_filtered = df_filtered[df_filtered[user_col].isin(valid_users)]
        rows_after_user_filter = df_filtered.shape[0]
        # print(f" Filter by user interaction count: {start_rows} -> {rows_after_user_filter} rows") # Less verbose
        if rows_after_user_filter == 0: break
        item_user_counts = df_filtered.groupby(item_col)[user_col].nunique()
        valid_items = item_user_counts[item_user_counts >= min_users_per_item].index
        df_filtered = df_filtered[df_filtered[item_col].isin(valid_items)]
        rows_after_item_filter = df_filtered.shape[0]
        # print(f" Filter by item user count: {rows_after_user_filter} -> {rows_after_item_filter} rows") # Less verbose
        if df_filtered.shape[0] == start_rows: break
    final_rows = df_filtered.shape[0]
    print(f"Finished interaction count filtering. Removed {initial_rows - final_rows} interaction records.")
    print(f"Final filtered interactions shape before aggregation: {df_filtered.shape}")
    return df_filtered

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating aggregated interaction features (implicit feedback)...") # ... (rest of function is unchanged) ...
    if df.empty: # ... (empty check is unchanged) ...
        print("Warning: Input DataFrame for aggregation is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=['id_student', 'presentation_id', 'total_clicks', 'interaction_days','first_interaction_date', 'last_interaction_date', 'implicit_feedback'])
    daily_interactions = df.groupby(['id_student', 'presentation_id', 'date'])['sum_click'].sum().reset_index()
    user_item_interactions = daily_interactions.groupby(['id_student', 'presentation_id']).agg(
        total_clicks=('sum_click', 'sum'), interaction_days=('date', 'nunique'),
        first_interaction_date=('date', 'min'), last_interaction_date=('date', 'max')
    ).reset_index()
    user_item_interactions['implicit_feedback'] = np.log1p(user_item_interactions['total_clicks'])
    print(f"Created aggregated interaction features. Shape: {user_item_interactions.shape}")
    # print(user_item_interactions.head()) # Less verbose
    return user_item_interactions

# --- UPDATED USER FEATURE GENERATION ---
def generate_user_features(student_info_clean: pd.DataFrame, valid_user_ids: np.ndarray) -> pd.DataFrame:
    """
    Generates final user features table for a specific list of valid user IDs.
    Ensures all valid users are present in the output, merging info from student_info_clean.
    """
    print(f"Generating user features for {len(valid_user_ids)} valid users...")
    if len(valid_user_ids) == 0:
        print("Warning: No valid user IDs provided. Returning empty user features DataFrame.")
        return pd.DataFrame(index=pd.Index([], name='id_student'))

    # Create a base DataFrame with just the valid user IDs
    users_final = pd.DataFrame({'id_student': valid_user_ids}).set_index('id_student')

    # --- Merge student info ---
    # Prepare student_info: take latest record per student and select relevant columns
    student_info_agg = student_info_clean.sort_values(by=['id_student', 'code_presentation'], ascending=[True, False])
    student_info_agg = student_info_agg.drop_duplicates(subset=['id_student'], keep='first')
    student_info_agg = student_info_agg[[
        'id_student', 'gender', 'region', 'highest_education', 'imd_band',
        'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
    ]].set_index('id_student')

    # Left merge onto the base DataFrame of valid IDs
    users_final = users_final.merge(student_info_agg, left_index=True, right_index=True, how='left')

    # --- Apply mappings and handle potential NaNs from merge ---
    # (Handles cases where a valid user ID was not found in student_info_clean)
    users_final['gender_mapped'] = users_final['gender'].apply(utils.map_gender).fillna(-1).astype(int)
    users_final['highest_education_mapped'] = users_final['highest_education'].apply(utils.map_highest_education).fillna(-1).astype(int)
    users_final['imd_band_mapped'] = users_final['imd_band'].apply(utils.map_imd_band).fillna(0).astype(int) # map_imd_band handles NaN->0
    users_final['age_band_mapped'] = users_final['age_band'].apply(utils.map_age_band).fillna(-1).astype(int)
    users_final['disability_mapped'] = users_final['disability'].apply(utils.map_disability).fillna(-1).astype(int)

    # Fill NaNs for numerical columns merged (e.g., if user info was missing)
    users_final['num_of_prev_attempts'] = users_final['num_of_prev_attempts'].fillna(0).astype(int)
    users_final['studied_credits'] = users_final['studied_credits'].fillna(0).astype(int)

    # Keep only necessary final columns
    # Keep original categoricals for potential inspection, but use mapped for models
    final_cols = [
        'num_of_prev_attempts', 'studied_credits',
        'gender_mapped', 'highest_education_mapped', 'imd_band_mapped',
        'age_band_mapped', 'disability_mapped', 'region', # Keep original region
        # Add original categoricals if needed: 'gender', 'highest_education', 'imd_band', 'age_band', 'disability'
    ]
    # Ensure region column exists even if all merges failed (unlikely)
    if 'region' not in users_final.columns:
        users_final['region'] = None
    users_final['region'] = users_final['region'].fillna('Unknown') # Fill NaN regions

    # Select final columns, handling potential missing ones if merge failed completely
    cols_to_select = [col for col in final_cols if col in users_final.columns]
    users_final = users_final[cols_to_select]

    print(f"Generated user features table. Shape: {users_final.shape}")
    # print(users_final.head()) # Less verbose
    return users_final

def generate_item_features(courses_df: pd.DataFrame, vle_clean: pd.DataFrame, valid_item_ids: np.ndarray) -> pd.DataFrame:
    print("Generating item (presentation) features...") # ... (rest of function is unchanged) ...
    items_df = courses_df[courses_df['presentation_id'].isin(valid_item_ids)][['presentation_id', 'module_presentation_length']].copy()
    vle_filtered = vle_clean[vle_clean['presentation_id'].isin(valid_item_ids)].copy()
    if items_df.empty: # ... (empty check is unchanged) ...
         print("Warning: No valid items found in courses_df. Returning empty item features DataFrame.")
         return pd.DataFrame(index=pd.Index([], name='presentation_id'), columns=['module_presentation_length'])
    if not vle_filtered.empty: # ... (VLE feature calculation is unchanged) ...
        vle_counts = vle_filtered.groupby('presentation_id')['activity_type'].value_counts().unstack(fill_value=0)
        vle_proportions = vle_counts.apply(lambda x: x / x.sum() if x.sum() > 0 else 0, axis=1) # Avoid divide by zero
        vle_proportions.columns = [f'vle_prop_{col}' for col in vle_proportions.columns]
        items_final = pd.merge(items_df, vle_proportions, on='presentation_id', how='left')
    else: # ... (handling empty VLE is unchanged) ...
        print("No VLE data for valid items. Item features will only contain presentation length.")
        items_final = items_df
    items_final.fillna(0, inplace=True)
    items_final = items_final.set_index('presentation_id')
    print(f"Generated item features table. Shape: {items_final.shape}")
    # print(items_final.head()) # Less verbose
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
    vle_clean = clean_vle(raw_data['vle'])
    student_vle_clean = clean_student_vle(raw_data['student_vle'])
    courses_df = utils.create_presentation_id(raw_data['courses'])
    # assessments_clean = clean_assessments(raw_data['assessments']) # Clean if needed later
    # student_assessment_clean = clean_student_assessment(raw_data['student_assessment']) # Clean if needed later

    # 3. Filter Interactions based on Registration Dates
    interactions_filtered_by_reg = filter_interactions_by_registration(
        student_vle_clean, registrations_clean
    )

    # 4. Apply Interaction Count Filters (on detailed interactions)
    interactions_count_filtered = apply_interaction_count_filters(
        interactions_filtered_by_reg
    )

    # 5. Create Aggregated Interaction Features
    final_interactions = create_interaction_features(interactions_count_filtered)

    # Handle case where filtering removed all data
    if final_interactions.empty:
        print("Warning: All interactions were filtered out. Returning empty dataframes.")
        empty_users = pd.DataFrame(index=pd.Index([], name='id_student'))
        empty_items = pd.DataFrame(index=pd.Index([], name='presentation_id'))
        processed_data = {'users': empty_users, 'items': empty_items, 'interactions': final_interactions}
    else:
        # --- UPDATED: Derive valid IDs *after* aggregation ---
        valid_user_ids = final_interactions['id_student'].unique()
        valid_item_ids = final_interactions['presentation_id'].unique()
        print(f"Final unique users in interactions: {len(valid_user_ids)}")
        print(f"Final unique items in interactions: {len(valid_item_ids)}")

        # 6. Generate User Features (passing the definitive list of users)
        users_final = generate_user_features(student_info_clean, valid_user_ids)

        # 7. Generate Item Features (passing the definitive list of items)
        items_final = generate_item_features(courses_df, vle_clean, valid_item_ids)

        # 8. Final Check (optional but good practice): Filter interactions again by the generated features' indices
        # This ensures consistency if feature generation somehow dropped an ID (e.g., due to merge issues)
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