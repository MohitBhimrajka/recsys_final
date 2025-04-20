# tests/test_database.py

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np

from src.database import load_to_db, schema
from src import config

# --- Fixtures for Dummy Data ---
@pytest.fixture
def dummy_users_df():
    # Corresponds to users_final.parquet structure AFTER preprocessing
    data = {
        'id_student': [1, 2, 3], # Matches model attribute after rename
        'num_of_prev_attempts': [0, 1, 0], # Will be renamed
        'studied_credits': [60, 120, 60],
        'gender_mapped': [0, 1, 0],
        'highest_education_mapped': [2, 3, 1],
        'imd_band_mapped': [2, 0, 1], # 0 for Missing
        'age_band_mapped': [0, 0, 1],
        'disability_mapped': [0, 0, 1],
        'region': ['R1', 'R2', 'R1']
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_items_df():
    # Corresponds to items_final.parquet structure AFTER preprocessing
    data = {
        'presentation_id': ['M1_P1', 'M1_P2', 'M2_P1'], # Used for splitting
        'module_presentation_length': [260, 260, 240],
        'vle_prop_resource': [0.5, 0.6, 0.4],
        'vle_prop_forumng': [0.1, 0.05, 0.2]
        # Add more dummy VLE props if schema expects them
    }
    df = pd.DataFrame(data)
    # Set index as it would be after loading in train.py/evaluate.py
    df = df.set_index('presentation_id')
    df.index.name = config.ITEM_COL # Standardize index name
    return df.reset_index() # Reset for load_presentations which expects col

@pytest.fixture
def dummy_interactions_df():
    # Corresponds to interactions_final.parquet structure
    data = {
        'id_student': [1, 1, 2, 3],
        'presentation_id': ['M1_P1', 'M1_P2', 'M1_P1', 'M2_P1'],
        'total_clicks': [100, 50, 200, 80],
        'interaction_days': [10, 5, 15, 8],
        'first_interaction_date': [-5, 0, -2, 10],
        'last_interaction_date': [200, 40, 180, 150],
        'implicit_feedback': [np.log1p(100), np.log1p(50), np.log1p(200), np.log1p(80)]
    }
    return pd.DataFrame(data)

# --- Mock Session Scope ---
@pytest.fixture
def mock_session():
    with patch('src.database.load_to_db.session_scope') as mock_scope:
        mock_session_obj = MagicMock()
        # Simulate query results if needed, e.g., for existing users/courses
        mock_session_obj.query.return_value.all.return_value = [] # Default: nothing exists
        # Make session_scope yield the mock session object
        mock_scope.return_value.__enter__.return_value = mock_session_obj
        yield mock_session_obj

# --- Tests ---
@patch('src.database.load_to_db.pd.read_parquet')
def test_load_users(mock_read_parquet, mock_session, dummy_users_df):
    mock_read_parquet.return_value = dummy_users_df.copy()
    # Rename columns as done in the function
    df_renamed = dummy_users_df.copy()
    df_renamed.rename(columns={'num_of_prev_attempts': 'num_prev_attempts'}, inplace=True)

    count, ids = load_to_db.load_users(mock_session)

    assert count == 3
    assert ids == {1, 2, 3}
    # Check that bulk_insert_mappings was called with the correct data structure
    mock_session.bulk_insert_mappings.assert_called_once()
    args, kwargs = mock_session.bulk_insert_mappings.call_args
    assert args[0] == schema.User # Check the correct model class was used
    # Check the keys and one record from the passed data list
    inserted_data = args[1]
    assert len(inserted_data) == 3
    expected_keys = {'student_id', 'num_prev_attempts', 'studied_credits', 'gender_mapped',
                     'highest_education_mapped', 'imd_band_mapped', 'age_band_mapped',
                     'disability_mapped', 'region'}
    assert set(inserted_data[0].keys()) == expected_keys
    # Verify renamed column and values for first user
    assert inserted_data[0]['student_id'] == 1
    assert inserted_data[0]['num_prev_attempts'] == 0
    assert inserted_data[0]['region'] == 'R1'

@patch('src.database.load_to_db.pd.read_parquet')
def test_load_presentations(mock_read_parquet, mock_session, dummy_items_df):
    mock_read_parquet.return_value = dummy_items_df.copy()
    # Simulate no existing courses
    mock_session.query.return_value.all.return_value = []

    courses_added, pres_added, ids = load_to_db.load_presentations(mock_session)

    assert courses_added == 2 # M1, M2
    assert pres_added == 3
    assert ids == {'M1_P1', 'M1_P2', 'M2_P1'}

    # Check calls (one for courses, one for presentations)
    assert mock_session.bulk_insert_mappings.call_count == 2
    calls = mock_session.bulk_insert_mappings.call_args_list
    # Call 1: Courses
    assert calls[0][0][0] == schema.Course
    assert len(calls[0][0][1]) == 2 # M1, M2
    assert calls[0][0][1][0]['module_id'] == 'M1'
    # Call 2: Presentations
    assert calls[1][0][0] == schema.Presentation
    assert len(calls[1][0][1]) == 3
    assert calls[1][0][1][0]['module_id'] == 'M1'
    assert calls[1][0][1][0]['presentation_code'] == 'P1'
    assert calls[1][0][1][0]['module_presentation_length'] == 260

@patch('src.database.load_to_db.pd.read_parquet')
def test_load_aggregated_interactions(mock_read_parquet, mock_session, dummy_interactions_df):
    mock_read_parquet.return_value = dummy_interactions_df.copy()
    # Simulate users 1 and 2 exist, but 3 does not
    user1 = MagicMock(); user1.student_id = 1
    user2 = MagicMock(); user2.student_id = 2
    mock_session.query.return_value.all.return_value = [user1, user2]

    count = load_to_db.load_aggregated_interactions(mock_session)

    # User 3's interaction should be filtered out
    assert count == 3
    mock_session.bulk_insert_mappings.assert_called_once()
    args, kwargs = mock_session.bulk_insert_mappings.call_args
    assert args[0] == schema.AggregatedInteraction
    inserted_data = args[1]
    assert len(inserted_data) == 3
    # Check that user 3 is not in the inserted data
    inserted_user_ids = {d['student_id'] for d in inserted_data}
    assert inserted_user_ids == {1, 2}
    # Check structure of one record
    assert inserted_data[0]['student_id'] == 1
    assert inserted_data[0]['module_id'] == 'M1'
    assert inserted_data[0]['presentation_code'] == 'P1'
    assert inserted_data[0]['implicit_feedback'] == pytest.approx(np.log1p(100))