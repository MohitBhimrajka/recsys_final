# tests/test_preprocessing.py

import pandas as pd
import pytest
import numpy as np
import math

from src.data import preprocess, utils
from src import config # To use config columns

# --- Setup Dummy Data ---
@pytest.fixture
def dummy_reg_data():
    data = {
        config.USER_COL: [1, 1, 2, 3],
        # Use different items to prevent automatic presentation_id collision if not careful
        config.ITEM_COL: ['A', 'B', 'C', 'D'],
        'date_registration': [-10, 0, -5, -20],
        'date_unregistration': [pd.NA, 50, pd.NA, 30], # User 1, 2 stay; User 3 unregs day 30
    }
    df = pd.DataFrame(data)
    # Simulate presentation_id being created *before* clean_registrations
    df['code_module'] = df[config.ITEM_COL] # Simple way for test
    df['code_presentation'] = 'P1'
    df = utils.create_presentation_id(df)
    return df

@pytest.fixture
def dummy_vle_data():
    data = {
        config.USER_COL: [1, 1, 1, 2, 3, 3],
        # Use items consistent with dummy_reg_data if filtering by registration depends on it
        config.ITEM_COL: ['A', 'A', 'B', 'C', 'D', 'D'], # Maps to A_P1, B_P1, C_P1, D_P1
        'id_site': [101, 101, 102, 103, 104, 104],
        'date': [-5, 15, 5, 0, 25, 35], # Note interaction dates relative to reg/unreg
        'sum_click': [2, 3, 1, 5, 1, 1]
    }
    df = pd.DataFrame(data)
    # Simulate presentation_id being created
    df['code_module'] = df[config.ITEM_COL]
    df['code_presentation'] = 'P1'
    df = utils.create_presentation_id(df)
    return df

@pytest.fixture
def dummy_interaction_data():
    """Creates a dummy interactions dataframe for testing splits."""
    # Data for interaction filtering and time split tests
    data = {
        config.USER_COL: [1, 1, 1, 1, 1,  2, 2, 2,  3,  4, 4, 4, 4, 5, 5, 5, 5],
        config.ITEM_COL: ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'D', 'A', 'C', 'D', 'A', 'B', 'E', 'E'],
        config.TIME_COL: [10, 20, 30, 40, 50, 15, 25, 35, 60, 5, 15, 45, 55, 10, 20, 30, 40],
        config.SCORE_COL:[1, 1, 1, 1, 1,  1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1] # Scores don't matter for split
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_student_info():
     data = {
          config.USER_COL: [1, 2, 3, 4],
          'code_module': ['M1', 'M1', 'M2', 'M1'],
          'code_presentation': ['P1', 'P1', 'P2', 'P1'],
          'gender': ['M', 'F', 'M', 'F'],
          'region': ['R1', 'R2', 'R1', 'R3'],
          # Use exact strings from the dataset/utils mapping
          'highest_education': ['HE Qualification', 'A Level or Equivalent', 'Lower Than A Level', 'HE Qualification'],
          'imd_band': ['10-20%', pd.NA, '0-10%', '50-60%'], # Use % sign
          'age_band': ['0-35', '0-35', '35-55', '55<='],
          'num_of_prev_attempts': [0, 1, 0, 2],
          'studied_credits': [60, 120, 60, 30],
          'disability': ['N', 'N', 'Y', 'N'],
          'final_result': ['Pass', 'Fail', 'Withdrawn', 'Pass'] # Needed for clean_student_info
     }
     return pd.DataFrame(data)

@pytest.fixture
def dummy_courses():
     data = {
          'code_module': ['M1', 'M2'],
          'code_presentation': ['P1', 'P2'],
          'module_presentation_length': [260, 240]
     }
     return pd.DataFrame(data)

@pytest.fixture
def dummy_vle_meta(): # Corresponds to vle.csv
     data = {
          'id_site': [101, 102, 201, 202],
          'code_module': ['M1', 'M1', 'M2', 'M2'],
          'code_presentation': ['P1', 'P1', 'P2', 'P2'],
          'activity_type': ['resource', 'forumng', 'oucontent', 'resource'],
          # Add week_from/week_to needed by clean_vle
          'week_from': [1, 1, 2, 3],
          'week_to': [1, 1, 2, 4]
     }
     return pd.DataFrame(data)

# --- Tests for cleaning functions ---
def test_clean_student_info(dummy_student_info):
    cleaned = preprocess.clean_student_info(dummy_student_info.copy())
    assert 'presentation_id' in cleaned.columns
    assert cleaned['imd_band'].isnull().sum() == 0
    assert 'Missing' in cleaned['imd_band'].unique()

def test_clean_registrations(dummy_reg_data):
    cleaned = preprocess.clean_registrations(dummy_reg_data.copy())
    assert 'presentation_id' in cleaned.columns
    assert 'is_unregistered' in cleaned.columns
    assert cleaned['is_unregistered'].dtype == bool
    # Use boolean comparison, not 'is' for safety
    assert cleaned.loc[cleaned[config.USER_COL] == 3, 'is_unregistered'].iloc[0] == True
    assert cleaned.loc[cleaned[config.USER_COL] == 1, 'is_unregistered'].iloc[0] == False
    assert cleaned['date_unregistration'].isnull().sum() == 0 # NAs filled

# --- Tests for filtering ---
def test_filter_interactions_by_registration(dummy_vle_data, dummy_reg_data):
    # Clean reg data first to get the right columns
    reg_clean = preprocess.clean_registrations(dummy_reg_data.copy())
    # VLE data needs presentation_id which fixture already creates
    #vle_clean = utils.create_presentation_id(dummy_vle_data.copy())

    filtered = preprocess.filter_interactions_by_registration(dummy_vle_data, reg_clean) # Pass original vle data

    # Expected:
    # User 1 (Reg -10, Unreg NA, Item A_P1, B_P1): Keeps interactions A@-5, A@15, B@5. (All dates >= -10)
    # User 2 (Reg -5, Unreg NA, Item C_P1): Keeps interaction C@0. (Date >= -5)
    # User 3 (Reg -20, Unreg 30, Item D_P1): Keeps interaction D@25. Drops interaction D@35. (Date >= -20 and < 30)
    assert len(filtered) == 5
    assert 1 in filtered[config.USER_COL].values
    assert 2 in filtered[config.USER_COL].values
    assert 3 in filtered[config.USER_COL].values
    assert filtered[filtered[config.USER_COL] == 3]['date'].iloc[0] == 25 # Check kept interaction for user 3

def test_apply_interaction_count_filters(dummy_interaction_data):
    # Default: min_user=5, min_item=5 -> Removes all data in this small set
    filtered_default = preprocess.apply_interaction_count_filters(dummy_interaction_data.copy())
    assert filtered_default.empty

    # Let's use min=2
    filtered_min2 = preprocess.apply_interaction_count_filters(dummy_interaction_data.copy(), min_interactions_per_user=2, min_users_per_item=2)
    # Trace:
    # Iter 1: User counts {1:5, 2:3, 3:1, 4:4, 5:4}. Remove User 3 (1<2). DF size = 16.
    # Iter 1: Item counts (users) A:{1,2,4,5}=4, B:{1,5}=2, C:{1,2,4}=3, D:{4}=1, E:{5}=1. Remove D, E (<2 users). DF size = 12 (Interactions for A, B, C remain)
    # Iter 2: User counts {1:3(A,B,C), 2:2(A,C), 4:2(A,C), 5:2(A,B)}. All >= 2. No users removed.
    # Iter 2: Item counts A:{1,2,4,5}=4, B:{1,5}=2, C:{1,2,4}=3. All >= 2. No items removed.
    # Loop terminates.
    assert len(filtered_min2) == 12 # Corrected assertion
    assert 3 not in filtered_min2[config.USER_COL].unique()
    assert 'D' not in filtered_min2[config.ITEM_COL].unique()
    assert 'E' not in filtered_min2[config.ITEM_COL].unique()

# --- Tests for feature generation ---
def test_generate_user_features(dummy_student_info):
     valid_ids = np.array([1, 3, 5]) # User 5 is not in dummy_student_info
     cleaned_info = preprocess.clean_student_info(dummy_student_info.copy())
     user_features = preprocess.generate_user_features(cleaned_info, valid_ids)

     assert len(user_features) == 3 # Should have rows for all valid IDs
     assert user_features.index.tolist() == [1, 3, 5]
     # Check User 1 data (exists)
     assert user_features.loc[1, 'gender_mapped'] == 0 # Mapped from M
     assert user_features.loc[1, 'highest_education_mapped'] == 3 # Mapped from 'HE Qualification'
     # Check User 3 data (exists)
     assert user_features.loc[3, 'disability_mapped'] == 1 # Mapped from Y
     assert user_features.loc[3, 'highest_education_mapped'] == 1 # Mapped from 'Lower Than A Level'
     assert user_features.loc[3, 'imd_band_mapped'] == 1 # Mapped from 0-10%
     # Check User 5 data (missing) - should have defaults or NaNs filled appropriately
     assert user_features.loc[5, 'gender_mapped'] == -1 # Default fillna
     assert user_features.loc[5, 'num_of_prev_attempts'] == 0 # Default fillna

def test_generate_item_features(dummy_courses, dummy_vle_meta):
    valid_ids = np.array(['M1_P1', 'M2_P2', 'M3_P3']) # M3_P3 not in courses/vle
    courses_with_id = utils.create_presentation_id(dummy_courses.copy())
    # Clean VLE needs week_from/to, which fixture now has
    vle_clean_with_id = preprocess.clean_vle(utils.create_presentation_id(dummy_vle_meta.copy()))

    item_features = preprocess.generate_item_features(courses_with_id, vle_clean_with_id, valid_ids)

    assert len(item_features) == 2 # Only valid IDs found in courses_df remain
    assert 'M1_P1' in item_features.index
    assert 'M2_P2' in item_features.index
    assert 'M3_P3' not in item_features.index
    assert 'module_presentation_length' in item_features.columns
    assert 'vle_prop_resource' in item_features.columns # Check one VLE prop
    assert 'vle_prop_forumng' in item_features.columns
    # Check calculation for M1_P1: 1 resource, 1 forumng -> 0.5 each
    assert item_features.loc['M1_P1', 'vle_prop_resource'] == pytest.approx(0.5)
    assert item_features.loc['M1_P1', 'vle_prop_forumng'] == pytest.approx(0.5)
    # Check calculation for M2_P2: 1 oucontent, 1 resource -> 0.5 each
    assert item_features.loc['M2_P2', 'vle_prop_oucontent'] == pytest.approx(0.5)
    assert item_features.loc['M2_P2', 'vle_prop_resource'] == pytest.approx(0.5)

# --- Tests for time_based_split (Keep existing tests, update assertions) ---
def test_time_based_split_threshold(dummy_interaction_data):
    """Tests splitting based on a time threshold."""
    threshold = 30
    train_df, test_df = preprocess.time_based_split(
        interactions_df=dummy_interaction_data,
        user_col=config.USER_COL,
        item_col=config.ITEM_COL,
        time_col=config.TIME_COL,
        time_unit_threshold=threshold
    )
    if not train_df.empty: assert train_df[config.TIME_COL].max() <= threshold
    if not test_df.empty: assert test_df[config.TIME_COL].min() > threshold
    # Based on stdout: Train=10 rows, Test=6 rows after filtering
    assert len(train_df) == 10 # Corrected assertion
    assert len(test_df) == 6  # Corrected assertion

def test_time_based_split_ratio(dummy_interaction_data):
    """Tests splitting based on ratio per user."""
    split_ratio = 0.7
    # Filter dummy data first like pipeline would
    filtered_data = preprocess.apply_interaction_count_filters(dummy_interaction_data.copy(), min_interactions_per_user=2, min_users_per_item=2) # Use min=2 example
    train_df, test_df = preprocess.time_based_split(
        interactions_df=filtered_data, # Use filtered data
        user_col=config.USER_COL,
        item_col=config.ITEM_COL,
        time_col=config.TIME_COL,
        split_ratio=split_ratio,
        time_unit_threshold=None # Ensure ratio is used
    )
    # Based on stdout: Train=11, Test=1
    assert len(train_df) == 11 # Corrected assertion
    assert len(test_df) == 1
    assert set(test_df[config.USER_COL].tolist()) == {1}