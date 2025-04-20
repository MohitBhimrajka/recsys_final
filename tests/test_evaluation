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
    rels = ['C', 'A'] # Relevant at rank 1 (A) and 3 (C)
    # DCG = 1/log2(1+1) + 1/log2(3+1) = 1/log2(2) + 1/log2(4) = 1/1 + 1/2 = 1.5
    # Ideal Recs = ['A', 'C', 'B', 'D', 'E']
    # IDCG = 1/log2(1+1) + 1/log2(2+1) = 1/log2(2) + 1/log2(3) = 1 + 1/1.58496 = 1.6309
    # NDCG = 1.5 / 1.6309 = 0.9197
    assert metrics.ndcg_at_k(recs, rels, k=5) == pytest.approx(0.9197, abs=1e-4)
    assert metrics.ndcg_at_k(recs, rels, k=1) == pytest.approx(1.0 / (1/np.log2(2))) # DCG = 1/log2(2)=1. IDCG=1/log2(2)=1. NDCG=1.
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
    items_df = pd.DataFrame(item_data).set_index(config.ITEM_COL) # Set index
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

    # Check train map
    assert evaluator.train_user_item_map == {1: {'A', 'B'}, 2: {'A'}, 3: {'B', 'C', 'D'}}

    # Check test map (should include User 4 even if not in train map initially)
    assert evaluator.test_user_item_map == {1: ['C'], 2: ['B'], 3: ['A'], 4: ['A']}
    assert 4 in evaluator.train_user_item_map # Check user 4 was added
    assert evaluator.train_user_item_map[4] == set() # Should be empty set

# --- Mock Model for evaluate_model test ---
class MockRecommender(BaseRecommender):
    def __init__(self, user_col, item_col, score_col, fixed_predictions):
        super().__init__(user_col, item_col, score_col)
        self._known_users = set(fixed_predictions.keys())
        self._known_items = set()
        for items in fixed_predictions.values():
             self._known_items.update(items.keys())
        self.fixed_predictions = fixed_predictions # {user: {item: score}}

    def fit(self, interactions_df: pd.DataFrame):
        # Mock fit, mappings are derived from fixed_predictions
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
    # User 4 (new) gets 0s
    mock_preds = {
        1: {'A': 0.8, 'B': 0.6, 'C': 0.9, 'D': 0.1, 'E': 0.0},
        2: {'A': 0.9, 'B': 0.7, 'C': 0.1, 'D': 0.2, 'E': 0.3},
        3: {'A': 0.9, 'B': 0.5, 'C': 0.6, 'D': 0.7, 'E': 0.1},
        # User 4 not in mock_preds -> predict returns 0s
    }
    mock_model = MockRecommender(config.USER_COL, config.ITEM_COL, config.SCORE_COL, mock_preds)
    mock_model.fit(train_df) # Mock fit

    evaluator = RecEvaluator(train_df, test_df, items_df,
                             user_col=config.USER_COL,
                             item_col=config.ITEM_COL, k=k)

    results = evaluator.evaluate_model(mock_model, n_neg_samples=None) # Full evaluation

    # --- Calculate expected results manually ---
    # User 1: Train={A,B}, Test={C}. Recs=['C','A']. Hits=1. P@2=1/2. R@2=1/1. NDCG@2 = (1/log2(1+1))/ (1/log2(1+1)) = 1.0
    # User 2: Train={A}, Test={B}. Recs=['A','B']. A is filtered. Should predict on B,C,D,E. -> Let's re-run evaluator logic...
    # Evaluator Logic:
    # U1: Test={C}, KnownTrain={A,B}. ItemsToPred={A,B,C,D,E}-{A,B}={C,D,E}. Add test pos C -> {C,D,E}. Scores C=0.9, D=0.1, E=0.0. Rank=[C,D,E]. Top2=[C,D]. Hits=1 (C). P=1/2. R=1/1. NDCG=1.
    # U2: Test={B}, KnownTrain={A}. ItemsToPred={A,B,C,D,E}-{A}={B,C,D,E}. Add test pos B -> {B,C,D,E}. Scores B=0.7,C=0.1,D=0.2,E=0.3. Rank=[B,E,D,C]. Top2=[B,E]. Hits=1 (B). P=1/2. R=1/1. NDCG=1.
    # U3: Test={A}, KnownTrain={B,C,D}. ItemsToPred={A,B,C,D,E}-{B,C,D}={A,E}. Add test pos A -> {A,E}. Scores A=0.9, E=0.1. Rank=[A,E]. Top2=[A,E]. Hits=1 (A). P=1/2. R=1/1. NDCG=1.
    # User 4: Test={A}, KnownTrain={}. ItemsToPred={A,B,C,D,E}. Add test pos A -> {A,B,C,D,E}. Predict returns 0 for all. Rank=[A,B,C,D,E] (order depends on sort stability). Top2=[A,B]. Hits=1 (A). P=1/2. R=1/1. NDCG=1.
    # Note: User 4 is KNOWN by the model because they are in mock_preds keys during fit.
    # If User 4 was not in mock_preds.keys(), they would be skipped by evaluator.

    # Expected: All users have P=0.5, R=1.0, NDCG=1.0
    # Average: P=0.5, R=1.0, NDCG=1.0
    # N Users evaluated = 4 (all test users are known by mock model)

    assert results[f'Precision@{k}'] == pytest.approx(0.5)
    assert results[f'Recall@{k}'] == pytest.approx(1.0)
    assert results[f'NDCG@{k}'] == pytest.approx(1.0)
    assert results['n_users_evaluated'] == 4