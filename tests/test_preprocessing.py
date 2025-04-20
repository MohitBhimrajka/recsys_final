# tests/test_preprocessing.py

import pandas as pd
import pytest
import math

from src.data import preprocess
from src import config # To use config columns

# --- Setup Dummy Data ---
@pytest.fixture
def dummy_interaction_data():
    """Creates a dummy interactions dataframe for testing splits."""
    # User 1: 5 interactions, times 10, 20, 30, 40, 50
    # User 2: 3 interactions, times 15, 25, 35 -> Time 35 > 30
    # User 3: 1 interaction, time 60          -> Time 60 > 30
    # User 4: 4 interactions, times 5, 15, 45, 55 -> Time 45, 55 > 30
    data = {
        config.USER_COL: [1, 1, 1, 1, 1,  2, 2, 2,  3,  4, 4, 4, 4],
        config.ITEM_COL: ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'D', 'A', 'C', 'D'],
        config.TIME_COL: [10, 20, 30, 40, 50, 15, 25, 35, 60, 5, 15, 45, 55],
        config.SCORE_COL:[1, 1, 1, 1, 1,  1, 1, 1,  1,  1, 1, 1, 1] # Scores don't matter for split
    }
    return pd.DataFrame(data)

# --- Tests for time_based_split ---

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

    # Check time boundaries
    if not train_df.empty:
        assert train_df[config.TIME_COL].max() <= threshold
    if not test_df.empty:
        assert test_df[config.TIME_COL].min() > threshold

    # Check users present in final train/test sets
    train_users = set(train_df[config.USER_COL].unique())
    test_users = set(test_df[config.USER_COL].unique())

    # --- CORRECTED ASSERTIONS ---
    # User 3's interaction is at time 60 (> threshold), so they are NOT in train.
    assert 3 not in train_users
    # Since User 3 is not in train, they are filtered OUT of the final test set.
    assert 3 not in test_users

    # User 2 has interactions at 15, 25 (<= threshold) and 35 (> threshold)
    assert 2 in train_users # Because of interactions <= 30
    assert 2 in test_users  # Because of interaction > 30 AND user 2 is in train

    # User 1 has interactions spanning the threshold
    assert 1 in train_users
    assert 1 in test_users

    # User 4 has interactions spanning the threshold
    assert 4 in train_users
    assert 4 in test_users
    # --- END CORRECTED ASSERTIONS ---

    # Check item filtering: Items only seen in test portion initially should be removed
    train_items = set(train_df[config.ITEM_COL].unique())
    test_items = set(test_df[config.ITEM_COL].unique())
    assert test_items.issubset(train_items) # All items in test must have appeared in train

    # Recalculate expected lengths based on the dummy data trace:
    # Train (<=30): U1(10,20,30), U2(15,25), U4(5,15). Rows=7. Users={1,2,4}. Items={A,B,C,D}
    # Test (>30): U1(40,50), U2(35), U3(60), U4(45,55). Rows=6.
    # Filter Test: Keep users {1,2,4}. Drop U3(60). -> Test = U1(40,50), U2(35), U4(45,55). Rows=5.
    # Filter Test: Keep items {A,B,C,D}. All items are in train. -> No change. Rows=5.
    assert len(train_df) == 7
    assert len(test_df) == 5

def test_time_based_split_ratio(dummy_interaction_data):
    """Tests splitting based on ratio per user."""
    split_ratio = 0.7
    train_df, test_df = preprocess.time_based_split(
        interactions_df=dummy_interaction_data,
        user_col=config.USER_COL,
        item_col=config.ITEM_COL,
        time_col=config.TIME_COL,
        split_ratio=split_ratio,
        time_unit_threshold=None # Ensure ratio is used
    )

    # Check filtering
    train_users = set(train_df[config.USER_COL].unique())
    test_users = set(test_df[config.USER_COL].unique())
    assert test_users.issubset(train_users) # All test users must be in train

    train_items = set(train_df[config.ITEM_COL].unique())
    test_items = set(test_df[config.ITEM_COL].unique())
    assert test_items.issubset(train_items) # All test items must be in train

    # Check counts per user (approximate due to ceiling)
    # User 1: 5 interactions * 0.7 = 3.5 -> ceil(3.5) = 4 train, 1 test
    # User 2: 3 interactions * 0.7 = 2.1 -> ceil(2.1) = 3 train, 0 test
    # User 3: 1 interaction * 0.7 = 0.7 -> ceil(0.7) = 1 train, 0 test
    # User 4: 4 interactions * 0.7 = 2.8 -> ceil(2.8) = 3 train, 1 test
    # Total Train = 4 + 3 + 1 + 3 = 11
    # Total Test Before Filtering = 1 + 0 + 0 + 1 = 2 (U1, U4)
    # Filtering Test: Users 1, 4 are in train. Items for test interactions (U1@50=B, U4@55=D) are in train ({A,B,C,D}). No filtering needed.
    assert len(train_df) == 11
    assert len(test_df) == 2
    assert set(test_df[config.USER_COL].tolist()) == {1, 4} # Users 1 and 4 should have test data