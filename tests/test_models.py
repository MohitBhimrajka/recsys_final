# tests/test_models.py

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from src.models.popularity import PopularityRecommender
from src.models.ncf import NCFRecommender # For save/load testing
from src import config # To use config columns

# --- Fixture for Dummy Data ---
@pytest.fixture
def dummy_train_data():
    """Provides dummy interaction data for model training tests."""
    data = {
        config.USER_COL: [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
        config.ITEM_COL: ['A', 'B', 'C', 'A', 'B', 'C', 'D', 'A', 'C', 'D'],
        config.SCORE_COL:[5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 1.0, 4.0, 3.5, 1.5]
    }
    return pd.DataFrame(data)

# --- Tests for PopularityRecommender ---

def test_popularity_fit(dummy_train_data):
    """Tests if PopularityRecommender fit calculates scores correctly."""
    model = PopularityRecommender(user_col=config.USER_COL,
                                  item_col=config.ITEM_COL,
                                  score_col=config.SCORE_COL)
    model.fit(dummy_train_data)

    # Expected scores: A=5+5+4=14, B=4+2=6, C=3+4.5+3.5=11, D=1+1.5=2.5
    expected_scores = {'A': 14.0, 'B': 6.0, 'C': 11.0, 'D': 2.5}
    assert model.item_popularity_scores == pytest.approx(expected_scores)
    assert model.most_popular_items == ['A', 'C', 'B', 'D'] # Check sorting
    # Check mappings
    assert model.n_users == 4
    assert model.n_items == 4
    assert set(model.get_known_users()) == {1, 2, 3, 4}
    assert set(model.get_known_items()) == {'A', 'B', 'C', 'D'}

def test_popularity_predict(dummy_train_data):
    """Tests PopularityRecommender predict method."""
    model = PopularityRecommender(user_col=config.USER_COL,
                                  item_col=config.ITEM_COL,
                                  score_col=config.SCORE_COL)
    model.fit(dummy_train_data)

    # Predict for known and unknown items, user_id doesn't matter
    items_to_predict = ['A', 'D', 'B', 'X', 'C'] # X is unknown
    predictions = model.predict(user_id=1, item_ids=items_to_predict)

    expected_predictions = [14.0, 2.5, 6.0, 0.0, 11.0] # Scores from fit, 0 for unknown
    assert predictions == pytest.approx(expected_predictions)

def test_popularity_recommend_top_k(dummy_train_data):
    """Tests PopularityRecommender top-k recommendations."""
    model = PopularityRecommender(user_col=config.USER_COL,
                                  item_col=config.ITEM_COL,
                                  score_col=config.SCORE_COL)
    model.fit(dummy_train_data)

    # Test basic top-k
    top_2 = model.recommend_top_k(user_id=1, k=2)
    assert top_2 == ['A', 'C']

    # Test filtering
    liked = {'C', 'B'}
    top_2_filtered = model.recommend_top_k(user_id=1, k=2, filter_already_liked_items=True, liked_items=liked)
    assert top_2_filtered == ['A', 'D'] # A, D are most popular excluding C, B

    # Test filtering without liked items provided (should return top K overall)
    top_1_no_liked = model.recommend_top_k(user_id=1, k=1, filter_already_liked_items=True, liked_items=None)
    assert top_1_no_liked == ['A']

# --- Basic Test for NCF Save/Load (doesn't check accuracy, just mechanics) ---
# This requires torch
@pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not installed")
def test_ncf_save_load(dummy_train_data):
    """Tests saving and loading the NCFRecommender wrapper."""
    model = NCFRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL,
                           score_col=config.SCORE_COL, epochs=1, batch_size=2) # Minimal training
    model.fit(dummy_train_data)

    # Use a temporary directory for saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ncf_test_model.pt"
        model.save_model(str(save_path))
        assert save_path.exists()

        # Load the model
        loaded_model = NCFRecommender.load_model(str(save_path))

        # Basic checks on loaded model
        assert isinstance(loaded_model, NCFRecommender)
        assert loaded_model.n_users == model.n_users
        assert loaded_model.n_items == model.n_items
        assert loaded_model.user_id_to_idx == model.user_id_to_idx
        assert loaded_model.item_id_to_idx == model.item_id_to_idx
        assert loaded_model.model is not None
        # Check if predict runs without error (doesn't check accuracy)
        preds = loaded_model.predict(1, ['A', 'B'])
        assert len(preds) == 2