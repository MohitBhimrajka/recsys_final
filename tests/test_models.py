# tests/test_models.py

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Import models
from src.models.popularity import PopularityRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.matrix_factorization import ImplicitALSWrapper
from src.models.ncf import NCFRecommender # For save/load testing
from src.models.hybrid import HybridNCFRecommender # For save/load testing
from src import config # To use config columns

# --- Fixture for Dummy Data ---
@pytest.fixture
def dummy_train_data():
    """Provides dummy interaction data for model training tests."""
    data = {
        config.USER_COL: [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5], # Added user 5
        config.ITEM_COL: ['A', 'B', 'C', 'A', 'B', 'C', 'D', 'A', 'C', 'D', 'A', 'E'], # Added item E
        config.SCORE_COL:[5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 1.0, 4.0, 3.5, 1.5, 1.0, 5.0] # Added scores
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_item_features():
    """Provides dummy item features, indexed correctly."""
    items = ['A', 'B', 'C', 'D', 'E']
    data = {
        'feature1': np.random.rand(len(items)),
        'feature2': np.random.randint(100, 200, size=len(items))
    }
    df = pd.DataFrame(data, index=pd.Index(items, name=config.ITEM_COL))
    return df

# --- Tests for PopularityRecommender (Keep Existing) ---
def test_popularity_fit(dummy_train_data):
    model = PopularityRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL)
    model.fit(dummy_train_data)
    expected_scores = {'A': 5.0+5.0+4.0+1.0, 'B': 4.0+2.0, 'C': 3.0+4.5+3.5, 'D': 1.0+1.5, 'E': 5.0}
    assert model.item_popularity_scores == pytest.approx(expected_scores)
    assert model.most_popular_items[0] == 'A' # A is most popular
    assert model.n_users == 5
    assert model.n_items == 5

def test_popularity_predict(dummy_train_data):
    model = PopularityRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL)
    model.fit(dummy_train_data)
    items_to_predict = ['A', 'D', 'B', 'X', 'C', 'E']
    predictions = model.predict(user_id=1, item_ids=items_to_predict)
    expected_predictions = [15.0, 2.5, 6.0, 0.0, 11.0, 5.0] # From expected_scores in test_popularity_fit
    assert predictions == pytest.approx(expected_predictions)

def test_popularity_recommend_top_k(dummy_train_data):
    model = PopularityRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL)
    model.fit(dummy_train_data)
    top_2 = model.recommend_top_k(user_id=1, k=2)
    assert top_2 == ['A', 'C'] # A=15, C=11, B=6, E=5, D=2.5
    liked = {'A', 'B'}
    top_2_filtered = model.recommend_top_k(user_id=1, k=2, filter_already_liked_items=True, liked_items=liked)
    assert top_2_filtered == ['C', 'E']

# --- Tests for ItemCFRecommender ---
def test_itemcf_fit(dummy_train_data):
    model = ItemCFRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL)
    model.fit(dummy_train_data)
    assert model.n_users == 5
    assert model.n_items == 5
    assert model.interaction_matrix_sparse is not None
    assert model.interaction_matrix_sparse.shape == (5, 5)
    assert model.item_similarity_matrix is not None
    assert model.item_similarity_matrix.shape == (5, 5)
    # Check diagonal is zero (or close to it due to float precision if not explicitly set)
    # Convert to dense for easier checking in test
    sim_dense = model.item_similarity_matrix.toarray()
    np.fill_diagonal(sim_dense, 0) # Ensure diag is 0 for comparison
    assert np.allclose(np.diag(sim_dense), 0)


def test_itemcf_predict(dummy_train_data):
    model = ItemCFRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL)
    model.fit(dummy_train_data)
    # User 1 interacted with A=5, B=4, C=3
    # Predict for D (similar to C via user 3, A via user 4?), E (only user 5)
    predictions = model.predict(user_id=1, item_ids=['D', 'E', 'A', 'X']) # X unknown
    assert len(predictions) == 4
    assert predictions[3] == 0.0 # Unknown item X
    # Expect non-zero scores for D and E based on learned similarities
    assert predictions[0] > 0 # Score for D should be influenced by C
    assert predictions[1] > 0 # Score for E should be influenced by A (user 5)
    assert predictions[2] == 0 # Score for A (already seen) - *Predict doesn't filter, just calculates score*

# --- Tests for ImplicitALSWrapper ---
@pytest.mark.skipif(not pytest.importorskip("implicit"), reason="implicit library not installed")
def test_als_fit(dummy_train_data):
    model = ImplicitALSWrapper(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL,
                               factors=5, iterations=2) # Minimal params for testing
    model.fit(dummy_train_data)
    assert model.n_users == 5
    assert model.n_items == 5
    assert model.model is not None
    assert model.model.user_factors.shape == (5, 5) # n_users x factors
    assert model.model.item_factors.shape == (5, 5) # n_items x factors
    assert model.user_item_matrix_sparse is not None

@pytest.mark.skipif(not pytest.importorskip("implicit"), reason="implicit library not installed")
def test_als_predict(dummy_train_data):
    model = ImplicitALSWrapper(user_col=config.USER_COL, item_col=config.ITEM_COL, score_col=config.SCORE_COL,
                               factors=5, iterations=2)
    model.fit(dummy_train_data)
    predictions = model.predict(user_id=1, item_ids=['D', 'E', 'A', 'X'])
    assert len(predictions) == 4
    assert predictions[3] == 0.0 # Unknown item X
    # ALS scores can be positive or negative, just check they are floats
    assert isinstance(predictions[0], float)
    assert isinstance(predictions[1], float)
    assert isinstance(predictions[2], float)
    # Predict for unknown user
    predictions_unknown_user = model.predict(user_id=99, item_ids=['A', 'B'])
    assert predictions_unknown_user == [0.0, 0.0]

# --- Tests for NCFRecommender Save/Load (Keep Existing) ---
@pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not installed")
def test_ncf_save_load(dummy_train_data):
    model = NCFRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL,
                           score_col=config.SCORE_COL, epochs=1, batch_size=2) # Minimal training
    model.fit(dummy_train_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ncf_test_model.pt"
        model.save_model(str(save_path))
        assert save_path.exists()
        loaded_model = NCFRecommender.load_model(str(save_path))
        assert isinstance(loaded_model, NCFRecommender)
        assert loaded_model.n_users == model.n_users
        assert loaded_model.n_items == model.n_items
        preds = loaded_model.predict(1, ['A', 'B'])
        assert len(preds) == 2

# --- Tests for HybridNCFRecommender Save/Load ---
@pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not installed")
def test_hybrid_save_load(dummy_train_data, dummy_item_features):
    model = HybridNCFRecommender(user_col=config.USER_COL, item_col=config.ITEM_COL,
                                 score_col=config.SCORE_COL, epochs=1, batch_size=2) # Minimal training
    # Fit requires item features
    model.fit(dummy_train_data, dummy_item_features)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "hybrid_test_model.pt"
        model.save_model(str(save_path))
        assert save_path.exists()

        loaded_model = HybridNCFRecommender.load_model(str(save_path))

        assert isinstance(loaded_model, HybridNCFRecommender)
        assert loaded_model.n_users == model.n_users
        assert loaded_model.n_items == model.n_items
        assert loaded_model.item_feature_dim == model.item_feature_dim
        assert np.array_equal(loaded_model.item_features_array, model.item_features_array) # Check features loaded
        # Check prediction runs
        preds = loaded_model.predict(1, ['A', 'B', 'X']) # X is unknown item
        assert len(preds) == 3
        assert preds[2] == 0.0 # Score for unknown item should be 0