# tests/test_preprocessing.py

import pandas as pd
import pytest
import numpy as np
import math

from src.data import preprocess, utils
from src import config # To use config columns

# --- Fixture for Dummy Data ---
@pytest.fixture
def dummy_reg_data():
    data = {
        config.USER_COL: [1, 1, 2, 3],
        config.ITEM_COL: ['A', 'B', 'A', 'A'],
        'date_registration': [-10, 0, -5, -20],
        'date_unregistration': [pd.NA, 50, pd.NA, 30], # User 1, 2 stay; User 3 unregs day 30
    }
    df = pd.DataFrame(data)
    # Manually create presentation_id as clean_registrations expects it
    df['presentation_id'] = df[config.ITEM_COL] # Simplified for test
    # Add required columns if clean_registrations uses them directly
    if 'code_module' not in df.columns: df['code_module'] = df[config.ITEM_COL]
    if 'code_presentation' not in df.columns: df['code_presentation'] = 'test'
    return df

@pytest.fixture
def dummy_vle_data():
    data = {
        config.USER_COL: [1, 1, 1, 2, 3, 3],
        config.ITEM_COL: ['A', 'A', 'B', 'A', 'A', 'A'],
        'id_site': [101, 101, 102, 101, 101, 101],
        'date': [-5, 15, 5, 0, 25, 35], # Note interaction dates relative to reg/unreg
        'sum_click': [2, 3, 1, 5, 1, 1]
    }
    df = pd.DataFrame(data)
    df['presentation_id'] = df[config.ITEM_COL] # Simplified
    if 'code_module' not in df.columns: df['code_module'] = df[config.ITEM_COL]
    if 'code_presentation' not in df.columns: df['code_presentation'] = 'test'
    return df

@pytest.fixture
def dummy_interaction_data():
    """Creates a dummy interactions dataframe for testing splits."""
    # User 1: 5 interactions, times 10, 20, 30, 40, 50
    # User 2: 3 interactions, times 15, 25, 35 -> Time 35 > 30
    # User 3: 1 interaction, time 60          -> Time 60 > 30
    # User 4: 4 interactions, times 5, 15, 45, 55 -> Time 45, 55 > 30
    # User 5: 4 interactions, < min_interactions_per_user
    # Item E: < min_users_per_item
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
          'highest_education': ['HE', 'A Level', 'Lower', 'HE'],
          'imd_band': ['10-20', pd.NA, '0-10', '50-60'],
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
          'activity_type': ['resource', 'forumng', 'oucontent', 'resource']
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
    assert cleaned.loc[cleaned[config.USER_COL] == 3, 'is_unregistered'].iloc[0] is True
    assert cleaned.loc[cleaned[config.USER_COL] == 1, 'is_unregistered'].iloc[0] is False
    assert cleaned['date_unregistration'].isnull().sum() == 0 # NAs filled

# --- Tests for filtering ---
def test_filter_interactions_by_registration(dummy_vle_data, dummy_reg_data):
    # Clean reg data first to get the right columns
    reg_clean = preprocess.clean_registrations(dummy_reg_data.copy())
    vle_clean = utils.create_presentation_id(dummy_vle_data.copy()) # Add presentation_id

    filtered = preprocess.filter_interactions_by_registration(vle_clean, reg_clean)

    # Expected:
    # User 1 (Reg -10, Unreg NA): Keeps interactions at -5, 15, 5. (All dates >= -10)
    # User 2 (Reg -5, Unreg NA): Keeps interaction at 0. (Date >= -5)
    # User 3 (Reg -20, Unreg 30): Keeps interaction at 25. Drops interaction at 35. (Date >= -20 and < 30)
    assert len(filtered) == 5
    assert 1 in filtered[config.USER_COL].values
    assert 2 in filtered[config.USER_COL].values
    assert 3 in filtered[config.USER_COL].values
    assert filtered[filtered[config.USER_COL] == 3]['date'].iloc[0] == 25 # Check kept interaction for user 3

def test_apply_interaction_count_filters(dummy_interaction_data):
    # Default: min_user=5, min_item=5
    filtered = preprocess.apply_interaction_count_filters(dummy_interaction_data.copy())
    # User 5 has 4 interactions -> removed
    # Item E has 1 user (User 5, who is removed anyway) -> removed
    # Check User 5 removed
    assert 5 not in filtered[config.USER_COL].unique()
    # Check Item E removed
    assert 'E' not in filtered[config.ITEM_COL].unique()
    # Check others remain (Users 1,2,3,4 have >=5 interactions across items A,B,C,D which have >=5 users)
    # Need to trace carefully. After User 5 removed:
    # Users: {1:5, 2:3, 3:1, 4:4} -> Filter by user (>=5) -> Only User 1 remains.
    # Items for User 1: {A,B,C}. All have >1 user initially.
    # Filter by item (>=5 users): Items A,B,C might still have enough users (1,2,3,4 interact). Need exact counts.
    # Let's re-run logic with min=3 for user and item
    filtered_min3 = preprocess.apply_interaction_count_filters(dummy_interaction_data.copy(), min_interactions_per_user=3, min_users_per_item=3)
    # Iter 1: User counts {1:5, 2:3, 3:1, 4:4, 5:4}. Remove User 3 (1<3). DF size = 16.
    # Iter 1: Item counts (users) A:{1,2,4,5}=4, B:{1,5}=2, C:{1,2,4}=3, D:{4}=1, E:{5}=1. Remove B, D, E (<3 users). DF size = 11 (Interactions for A, C remain)
    # Iter 2: User counts {1:3(A,A,C), 2:2(A,C), 4:2(A,C), 5:1(A)}. Remove User 5 (1<3). DF size = 10.
    # Iter 2: Item counts A:{1,2,4}=3, C:{1,2,4}=3. No items removed.
    # Iter 3: User counts {1:3, 2:2, 4:2}. Remove Users 2, 4 (2<3). DF size = 3 (Only user 1's interactions with A, C remain)
    # Iter 3: Item counts A:{1}=1, C:{1}=1. Remove A, C (<3 users). DF size = 0.
    # --> With min=3, all data is removed.

    # Let's use min=2
    filtered_min2 = preprocess.apply_interaction_count_filters(dummy_interaction_data.copy(), min_interactions_per_user=2, min_users_per_item=2)
    # Iter 1: User counts {1:5, 2:3, 3:1, 4:4, 5:4}. Remove User 3 (1<2). DF size = 16.
    # Iter 1: Item counts A:{1,2,4,5}=4, B:{1,5}=2, C:{1,2,4}=3, D:{4}=1, E:{5}=1. Remove D, E (<2 users). DF size = 13 (Interactions for A, B, C remain)
    # Iter 2: User counts {1:4(A,B,A,C), 2:2(A,C), 4:2(A,C), 5:2(A,B)}. All >= 2. No users removed.
    # Iter 2: Item counts A:{1,2,4,5}=4, B:{1,5}=2, C:{1,2,4}=3. All >= 2. No items removed.
    # Loop terminates.
    assert len(filtered_min2) == 13
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
     assert user_features.loc[1, 'highest_education_mapped'] == 3 # Mapped from HE
     # Check User 3 data (exists)
     assert user_features.loc[3, 'disability_mapped'] == 1 # Mapped from Y
     assert user_features.loc[3, 'imd_band_mapped'] == 1 # Mapped from 0-10
     # Check User 5 data (missing) - should have defaults or NaNs filled appropriately
     assert user_features.loc[5, 'gender_mapped'] == -1 # Default fillna
     assert user_features.loc[5, 'num_of_prev_attempts'] == 0 # Default fillna

def test_generate_item_features(dummy_courses, dummy_vle_meta):
    valid_ids = np.array(['M1_P1', 'M2_P2', 'M3_P3']) # M3_P3 not in courses/vle
    courses_with_id = utils.create_presentation_id(dummy_courses.copy())
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

# --- Tests for time_based_split (Keep existing tests) ---
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
    # Recalculate expected lengths based on the dummy data trace (excluding user 5 / item E due to apply_interaction_filters not being run here):
    # Train (<=30): U1(10,20,30), U2(15,25), U4(5,15). Rows=7.
    # Test (>30): U1(40,50), U2(35), U3(60), U4(45,55). Rows=6.
    # Filter Test: Keep users {1,2,4}. Drop U3(60). -> Test = U1(40,50), U2(35), U4(45,55). Rows=5.
    # Filter Test: Keep items {A,B,C,D}. No change. Rows=5.
    assert len(train_df) == 7 # U1:3, U2:2, U4:2 = 7
    assert len(test_df) == 5 # U1:2, U2:1, U4:2 = 5 (U3 dropped)

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
    # Filtered data (min=2): U1(A10,B20,A30,C40)=4, U2(A15,C25)=2, U4(A15,C45)=2, U5(A10,B20)=2. Total=10 rows.
    # Split logic (ceil(n*0.7)):
    # U1: ceil(4*0.7)=3 train, 1 test
    # U2: ceil(2*0.7)=2 train, 0 test
    # U4: ceil(2*0.7)=2 train, 0 test
    # U5: ceil(2*0.7)=2 train, 0 test
    # Total Train = 3+2+2+2 = 9
    # Total Test = 1+0+0+0 = 1 (U1 only)
    assert len(train_df) == 9
    assert len(test_df) == 1
    assert set(test_df[config.USER_COL].tolist()) == {1}