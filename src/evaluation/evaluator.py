# src/evaluation/evaluator.py

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm # Use tqdm.notebook for notebook environment if running there
# from tqdm import tqdm # Use standard tqdm for script execution

import sys
from pathlib import Path
import random

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
# Import metrics calculation functions
from src.evaluation import metrics

class RecEvaluator:
    """
    Handles the evaluation protocol for recommendation models.

    Args:
        train_df (pd.DataFrame): Training interaction data.
        test_df (pd.DataFrame): Test interaction data.
        item_features_df (pd.DataFrame): DataFrame with item features (indexed by item_id).
        user_col (str): Name of the user ID column.
        item_col (str): Name of the item ID column.
        k (int): The 'K' value for top-K evaluation metrics (e.g., K=10).
    """
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, item_features_df: pd.DataFrame,
                 user_col: str = 'student_id', item_col: str = 'presentation_id', k: int = config.TOP_K):

        self.user_col = user_col
        self.item_col = item_col
        self.k = k

        # Store all unique items present in the training data (potential candidates)
        self.all_items = set(train_df[self.item_col].unique())
        # Also include items from test set that might have been filtered but are in features
        if item_features_df is not None:
             self.all_items.update(set(item_features_df.index))
        print(f"Evaluator initialized with {len(self.all_items)} unique candidate items.")

        # Store training interactions for filtering candidates
        # Create a set of (user, item) tuples for fast lookup
        self.train_interactions_set = set(zip(train_df[self.user_col], train_df[self.item_col]))
        print(f"Stored {len(self.train_interactions_set)} training interactions for filtering.")

        # Prepare test data: Group relevant items by user
        self.test_user_items = test_df.groupby(self.user_col)[self.item_col].apply(list).to_dict()
        self.test_users = list(self.test_user_items.keys())
        print(f"Prepared test data for {len(self.test_users)} users.")

    def _get_candidate_items(self, user_id: int) -> list:
        """
        Generates candidate items for a given user.
        Includes all items NOT seen by the user in the training set.
        """
        # Items the user interacted with in training
        user_train_items = {item for (user, item) in self.train_interactions_set if user == user_id}

        # Candidate items are all items minus those the user saw in training
        candidates = list(self.all_items - user_train_items)

        return candidates

    def evaluate_model(self, model, n_neg_samples: int = None) -> dict:
        """
        Evaluates a trained recommendation model.

        Args:
            model: A trained model object with a `predict(user_id, item_ids)` method
                   that returns scores for the given items for that user.
            n_neg_samples (int, optional): If set, instead of scoring all candidates,
                                           samples 'n_neg_samples' negative items per user
                                           along with the ground truth positive items for scoring.
                                           Speeds up evaluation but makes it approximate. Defaults to None (score all).

        Returns:
            dict: A dictionary containing average evaluation metrics (Precision@K, Recall@K, NDCG@K).
        """
        print(f"\n--- Evaluating Model: {type(model).__name__} ---")
        all_precisions = []
        all_recalls = []
        all_ndcgs = []

        # Using tqdm for progress bar
        for user_id in tqdm(self.test_users, desc="Evaluating users"):
            # 1. Get Ground Truth relevant items for this user from the test set
            relevant_items = self.test_user_items.get(user_id, [])
            if not relevant_items:
                continue # Skip user if they have no relevant items in the test set

            # 2. Generate Candidate Items for prediction
            if n_neg_samples is not None:
                # Sample negative items (items user DID NOT interact with in train OR test)
                user_train_items = {item for (user, item) in self.train_interactions_set if user == user_id}
                user_test_items = set(relevant_items)
                potential_negatives = list(self.all_items - user_train_items - user_test_items)

                # Ensure we don't sample more negatives than available
                num_to_sample = min(n_neg_samples, len(potential_negatives))
                sampled_negatives = random.sample(potential_negatives, num_to_sample)

                # Candidates = ground truth positives + sampled negatives
                candidate_items = relevant_items + sampled_negatives
                if not candidate_items: continue # Skip if no candidates somehow

                print(f" User {user_id}: Scoring {len(relevant_items)} positives + {len(sampled_negatives)} negatives.")
            else:
                # Score all items not seen in training
                candidate_items = self._get_candidate_items(user_id)
                if not candidate_items: continue # Skip if no candidates

            # 3. Get Model Predictions (Scores) for candidate items
            try:
                # Model needs to predict scores for the user and the list of candidate items
                scores = model.predict(user_id, candidate_items) # This needs implementation in each model class

                if len(scores) != len(candidate_items):
                     print(f"Warning: Mismatch between scores ({len(scores)}) and candidates ({len(candidate_items)}) for user {user_id}. Skipping user.")
                     continue

                # Combine items and scores, then sort to get ranked list
                scored_items = list(zip(candidate_items, scores))
                scored_items.sort(key=lambda x: x[1], reverse=True) # Sort by score descending

                # Get the ranked list of item IDs
                ranked_recommendations = [item_id for item_id, score in scored_items]

            except Exception as e:
                print(f"Error predicting for user {user_id}: {e}. Skipping user.")
                continue

            # 4. Calculate Metrics for this user
            precision = metrics.precision_at_k(ranked_recommendations, relevant_items, self.k)
            recall = metrics.recall_at_k(ranked_recommendations, relevant_items, self.k)
            ndcg = metrics.ndcg_at_k(ranked_recommendations, relevant_items, self.k)

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_ndcgs.append(ndcg)

        # 5. Average Metrics across all evaluated users
        avg_precision = np.mean(all_precisions) if all_precisions else 0.0
        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        avg_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0

        results = {
            f'Precision@{self.k}': avg_precision,
            f'Recall@{self.k}': avg_recall,
            f'NDCG@{self.k}': avg_ndcg,
            'n_users_evaluated': len(all_precisions)
        }

        print(f"\n--- Evaluation Results (K={self.k}) ---")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 30)

        return results