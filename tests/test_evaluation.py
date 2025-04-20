# tests/test_evaluation.py

import pandas as pd
import pytest
import numpy as np

from src.evaluation import metrics
from src.evaluation.evaluator import RecEvaluator
from src.models.base import BaseRecommender # For mock model
from src import config

# --- Tests for Metrics ---

def test_precision_at_k():
    recs = ['A', 'B', 'C', 'D', 'E']
    rels = ['C', 'A', 'F'] # Relevant: A, C
    assert metrics.precision_at_k(recs, rels, k=3) == pytest.approx(2/3)
    assert metrics.precision_at_k(recs, rels, k=5) == pytest.approx(2/5)
    assert metrics.precision_at_k(recs, [], k=5) == 0.0 # No relevant items
    assert metrics.precision_at_k([], rels, k=5) == 0.0 # No recommended items
    assert metrics.precision_at_k(recs, rels, k=0) == 0.0 # K=0

def test_recall_at_k():
    recs = ['A', 'B', 'C', 'D', 'E']
    rels = ['C', 'A', 'F', 'G'] # Relevant: A, C, F, G (Total 4)
    assert metrics.recall_at_k(recs, rels, k=3) == pytest.approx(2/4) # Found A, C
    assert metrics.recall_at_k(recs, rels, k=5) == pytest.approx(2/4) # Found A, C
    assert metrics.recall_at_k(recs, [], k=5) == 0.0 # No relevant items
    assert metrics.recall_at_k([], rels, k=5) == 0.0 # No recommended items
    assert metrics.recall_at_k(recs, rels, k=0) == 0.0 # K=0

def test_ndcg_at_k():
    recs = ['A', 'B', 'C', 'D', 'E'] # Ranks: 1, 2, 3, 4, 5
    rels = ['C', 'A'] # Relevant at rank 1 (A) and 3 (C) -> CORRECTION: Relevant A (rank 1), C (rank 3)
    # DCG = 1/log2(1+1) + 1/log2(3+1) = 1/log2(2) + 1/log2(4) = 1/1 + 1/2 = 1.5
    # Ideal Recs = ['A', 'C', 'B', 'D', 'E']
    # IDCG = 1/log2(1+1) + 1/log2(2+1) = 1/log2(2) + 1/log2(3) = 1 + 1/1.58496 = 1.6309
    # NDCG = 1.5 / 1.6309 = 0.9197
    assert metrics.ndcg_at_k(recs, rels, k=5) == pytest.approx(0.9197, abs=1e-4)
    # Check K=1: A is relevant. DCG = 1/log2(1+1) = 1. Ideal Recs = ['A', 'C', ...]. IDCG = 1/log2(1+1) = 1. NDCG = 1/1 = 1.0
    assert metrics.ndcg_at_k(recs, rels, k=1) == pytest.approx(1.0) # Corrected expected value check logic
    assert metrics.ndcg_at_k(recs, [], k=5) == 0.0
    assert metrics.ndcg_at_k([], rels, k=5) == 0.0
    assert metrics.ndcg_at_k(recs, rels, k=0) == 0.0

# --- Tests for RecEvaluator ---

@pytest.fixture
def dummy_eval_data():
    """Provides dummy data for evaluator tests."""
    train_data = {config.USER_COL: [1, 1, 2, 3, 3, 3],
                  config.ITEM_COL: ['A', 'B', 'A', 'B', 'C', 'D'],
                  config.SCORE_COL: [1, 1, 1, 1, 1, 1]}
    test_data = {config.USER_COL: [1, 2, 3, 4], # User 4 is new in test
                 config.ITEM_COL: ['C', 'B', 'A', 'A'],
                 config.SCORE_COL: [1, 1, 1, 1]}
    item_data = {config.ITEM_COL: ['A', 'B', 'C', 'D', 'E']} # E is unseen in interactions

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    # Ensure items_df has item_col as index for evaluator
    items_df = pd.DataFrame(item_data).set_index(config.ITEM_COL)
    items_df.index.name = config.ITEM_COL # Ensure index name matches

    return train_df, test_df, items_df

def test_rec_evaluator_init(dummy_eval_data):
    """Tests the initialization and data preparation of RecEvaluator."""
    train_df, test_df, items_df = dummy_eval_data
    evaluator = RecEvaluator(train_df, test_df, items_df,
                             user_col=config.USER_COL,
                             item_col=config.ITEM_COL, k=5)

    assert evaluator.k == 5
    assert evaluator.all_items == {'A', 'B', 'C', 'D', 'E'}

    # Check train map - User 4 should be added with an empty set
    expected_train_map = {1: {'A', 'B'}, 2: {'A'}, 3: {'B', 'C', 'D'}, 4: set()} # Added user 4
    assert evaluator.train_user_item_map == expected_train_map

    # Check test map (should include User 4 even if not in train map initially)
    assert evaluator.test_user_item_map == {1: ['C'], 2: ['B'], 3: ['A'], 4: ['A']}
    assert 4 in evaluator.train_user_item_map # Check user 4 was added
    assert evaluator.train_user_item_map[4] == set() # Should be empty set

# --- Mock Model for evaluate_model test ---
class MockRecommender(BaseRecommender):
    def __init__(self, user_col, item_col, score_col, fixed_predictions):
        super().__init__(user_col, item_col, score_col)
        # Known users ONLY defined by the keys in fixed_predictions
        self._known_users = set(fixed_predictions.keys())
        self._known_items = set()
        for items in fixed_predictions.values():
             self._known_items.update(items.keys())
        # Add items that might only exist in dummy_eval_data items_df
        self._known_items.update(['A', 'B', 'C', 'D', 'E']) # Ensure all items are known
        self.fixed_predictions = fixed_predictions # {user: {item: score}}


    def fit(self, interactions_df: pd.DataFrame):
        # Mock fit, mappings are derived from fixed_predictions and dummy items
        self.user_id_to_idx = {u: i for i, u in enumerate(self._known_users)}
        self.item_id_to_idx = {i: idx for idx, i in enumerate(self._known_items)}
        self.n_users = len(self._known_users)
        self.n_items = len(self._known_items)

    def predict(self, user_id, item_ids):
        user_preds = self.fixed_predictions.get(user_id, {})
        return [float(user_preds.get(item_id, 0.0)) for item_id in item_ids] # Return 0 for unknowns

    def get_known_users(self): return self._known_users
    def get_known_items(self): return self._known_items


def test_rec_evaluator_evaluate_model(dummy_eval_data):
    """Tests the evaluate_model method with a mock model."""
    train_df, test_df, items_df = dummy_eval_data
    k=2

    # Mock predictions: User 1 likes C>A>B, User 2 likes A>B, User 3 likes A>D>C>B
    # User 4 IS NOT included here, so model doesn't "know" them.
    mock_preds = {
        1: {'A': 0.8, 'B': 0.6, 'C': 0.9, 'D': 0.1, 'E': 0.0},
        2: {'A': 0.9, 'B': 0.7, 'C': 0.1, 'D': 0.2, 'E': 0.3},
        3: {'A': 0.9, 'B': 0.5, 'C': 0.6, 'D': 0.7, 'E': 0.1},
    }
    mock_model = MockRecommender(config.USER_COL, config.ITEM_COL, config.SCORE_COL, mock_preds)
    mock_model.fit(train_df) # Mock fit

    evaluator = RecEvaluator(train_df, test_df, items_df,
                             user_col=config.USER_COL,
                             item_col=config.ITEM_COL, k=k)

    results = evaluator.evaluate_model(mock_model, n_neg_samples=None) # Full evaluation

    # --- Calculate expected results manually (Updated understanding) ---
    # Model knows users {1, 2, 3}. Test set has users {1, 2, 3, 4}.
    # Evaluator will iterate through test users {1, 2, 3, 4}.
    # It checks if user is known by model:
    # User 1: Known. Test={C}. Train={A,B}. ItemsToPred={C,D,E}. Rank=[C,D,E]. Top2=[C,D]. P=0.5, R=1, NDCG=1.
    # User 2: Known. Test={B}. Train={A}. ItemsToPred={B,C,D,E}. Rank=[B,E,D,C]. Top2=[B,E]. P=0.5, R=1, NDCG=1.
    # User 3: Known. Test={A}. Train={B,C,D}. ItemsToPred={A,E}. Rank=[A,E]. Top2=[A,E]. P=0.5, R=1, NDCG=1.
    # User 4: NOT Known by model. Evaluator skips this user.
    # N Users evaluated = 3

    # Expected Average: P=0.5, R=1.0, NDCG=1.0 (average over users 1, 2, 3)
    assert results[f'Precision@{k}'] == pytest.approx(0.5)
    assert results[f'Recall@{k}'] == pytest.approx(1.0)
    assert results[f'NDCG@{k}'] == pytest.approx(1.0)
    # Corrected assertion for number of users evaluated
    assert results['n_users_evaluated'] == 3