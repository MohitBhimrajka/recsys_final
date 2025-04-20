# src/evaluation/evaluator.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
from typing import List, Dict, Set, Optional, Any # Added Optional, Any

from .metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.models.base import BaseRecommender # Import base class

class RecEvaluator:
    """
    Handles the evaluation of recommendation models using standard ranking metrics.
    Uses a time-based split methodology (train/test) and considers filtering
    already seen items.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 item_features_df: pd.DataFrame, # Should have item_col as index or column
                 user_col: str = 'id_student',
                 item_col: str = 'presentation_id',
                 k: int = 10):
        """
        Initializes the evaluator.

        Args:
            train_df (pd.DataFrame): Training interactions dataframe.
            test_df (pd.DataFrame): Test interactions dataframe (ground truth).
            item_features_df (pd.DataFrame): DataFrame containing all possible item IDs
                                             and potentially their features. Must contain
                                             item_col either as index or a column.
            user_col (str): Name of the user ID column.
            item_col (str): Name of the item ID column.
            k (int): The number of recommendations to consider for metrics@k.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.user_col = user_col
        self.item_col = item_col
        self.k = k

        # Extract all unique item IDs from the item features dataframe
        if item_col in item_features_df.columns:
             self.all_items = set(item_features_df[item_col].unique())
        elif item_features_df.index.name == item_col:
             self.all_items = set(item_features_df.index.unique())
        else:
             raise ValueError(f"Item column '{item_col}' not found in item_features_df columns or index.")

        print(f"Evaluator initialized with {len(self.all_items)} unique candidate items.")

        # Pre-process data for faster lookup during evaluation
        self._prepare_data()

    def _prepare_data(self):
        """Pre-processes training and test data for efficient evaluation."""
        # Group training data by user for quick filtering of seen items
        self.train_user_item_map = self.train_df.groupby(self.user_col)[self.item_col].agg(set).to_dict()
        print(f"Stored {len(self.train_user_item_map)} training interactions for filtering.")

        # Group test data by user (ground truth for evaluation)
        self.test_user_item_map = self.test_df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        # Ensure users in test exist in train map (add empty set if not)
        for user in self.test_user_item_map:
             if user not in self.train_user_item_map:
                  self.train_user_item_map[user] = set()
        print(f"Prepared test data for {len(self.test_user_item_map)} users.")


    def evaluate_model(self, model: BaseRecommender, n_neg_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluates a given recommendation model.

        Args:
            model (BaseRecommender): The recommendation model instance to evaluate (must implement fit and predict).
            n_neg_samples (Optional[int]): If set, evaluates using negative sampling. For each user,
                                           predicts scores for their positive test items plus
                                           `n_neg_samples` randomly sampled negative items
                                           (items not interacted with by the user in train or test,
                                           and known to the model). If None, predicts for all items
                                           known to the model (excluding training positives).
                                           Default is None (full evaluation).

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics (Precision@k, Recall@k, NDCG@k).
        """
        print(f"\n--- Evaluating Model: {model.__class__.__name__} ---")
        precisions, recalls, ndcgs = [], [], []
        skipped_users = 0
        evaluated_users = 0

        test_users = list(self.test_user_item_map.keys())
        if not test_users:
            print("Warning: No users found in the test set.")
            return {'Precision@k': 0, 'Recall@k': 0, 'NDCG@k': 0, 'n_users_evaluated': 0}

        # Get items known by this specific model instance ONCE before the loop
        model_known_items = model.get_known_items()
        if not model_known_items:
             print(f"Error: Model {model.__class__.__name__} reported no known items. Cannot evaluate.")
             return {'Precision@k': 0, 'Recall@k': 0, 'NDCG@k': 0, 'n_users_evaluated': 0}

        # Filter test users: only evaluate users known by the model
        model_known_users = model.get_known_users()
        test_users_known_by_model = [u for u in test_users if u in model_known_users]
        print(f"Total test users: {len(test_users)}. Evaluating {len(test_users_known_by_model)} users known by the model.")

        if not test_users_known_by_model:
            print("Warning: No test users are known by the model. Cannot evaluate.")
            return {'Precision@k': 0, 'Recall@k': 0, 'NDCG@k': 0, 'n_users_evaluated': 0}


        for user_id in tqdm(test_users_known_by_model, desc="Evaluating users"):
            test_positives = self.test_user_item_map.get(user_id, [])
            known_positives = self.train_user_item_map.get(user_id, set())

            if not test_positives:
                # print(f"User {user_id} has no positive items in test set. Skipping.")
                skipped_users += 1
                continue

             # Ensure test positives are actually known by the model before proceeding
            test_positives_known = [item for item in test_positives if item in model_known_items]
            if not test_positives_known:
                 # print(f"User {user_id}: None of their test positive items are known to the model. Skipping.")
                 skipped_users += 1
                 continue


            # --- Determine items to predict ---
            items_to_predict = []
            if n_neg_samples is not None:
                # Negative Sampling: Use only items known to *this* model
                # Exclude items the user interacted with in train *and* test
                possible_negatives = list(model_known_items - known_positives - set(test_positives))

                if len(possible_negatives) < n_neg_samples:
                    # print(f"Warning: User {user_id}: Not enough negatives ({len(possible_negatives)}) known to model to sample {n_neg_samples}. Using all available.")
                    sampled_negatives = possible_negatives
                else:
                    # Ensure sampling only from possible negatives
                    sampled_negatives = random.sample(possible_negatives, n_neg_samples)

                # Combine known test positives with sampled negatives
                items_to_predict = test_positives_known + sampled_negatives
                # print(f" User {user_id}: Scoring {len(test_positives_known)} positives + {len(sampled_negatives)} negatives.")

            else:
                # Full Evaluation: Predict for all items known to the model, excluding training items
                items_to_predict = list(model_known_items - known_positives)
                # Ensure test positives are included (they might have been filtered by the set difference)
                items_to_predict = list(set(items_to_predict + test_positives_known))
                # print(f" User {user_id}: Scoring {len(items_to_predict)} items (all known items minus train positives).")


            if not items_to_predict:
                # print(f"Warning: No items to predict for user {user_id} after filtering. Skipping.")
                skipped_users += 1
                continue

            # --- Predict scores ---
            try:
                # Model's predict method should handle unknown items gracefully now (return 0s)
                # We pass the list of items we determined above
                scores = model.predict(user_id, items_to_predict)

                if len(scores) != len(items_to_predict):
                     print(f"Error: Score list length ({len(scores)}) doesn't match item list length ({len(items_to_predict)}) for user {user_id}. Skipping user.")
                     skipped_users +=1
                     continue

                item_score_map = dict(zip(items_to_predict, scores))

                # Rank items based on score (descending)
                ranked_items = sorted(item_score_map.keys(), key=lambda item: item_score_map[item], reverse=True)

                # Get top K recommendations
                top_k_recs = ranked_items[:self.k]

                # Calculate metrics
                # Note: We use test_positives_known here as the ground truth relevant items *for this model*
                prec = precision_at_k(top_k_recs, test_positives_known, self.k)
                rec = recall_at_k(top_k_recs, test_positives_known, self.k) # Recall denominator uses test_positives_known length
                ndcg = ndcg_at_k(top_k_recs, test_positives_known, self.k)

                precisions.append(prec)
                recalls.append(rec)
                ndcgs.append(ndcg)
                evaluated_users += 1

            except Exception as e:
                # Catch errors during prediction or metric calculation for a specific user
                print(f"Error evaluating user {user_id}: {e}. Skipping user.")
                skipped_users += 1
                continue # Skip to the next user

        # --- Aggregate Results ---
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0

        print(f"\n--- Evaluation Results (K={self.k}) ---")
        print(f"Precision@{self.k}: {avg_precision:.4f}")
        print(f"Recall@{self.k}: {avg_recall:.4f}")
        print(f"NDCG@{self.k}: {avg_ndcg:.4f}")
        print(f"n_users_evaluated: {evaluated_users:.4f}")
        print(f"n_users_skipped: {skipped_users:.4f}")
        print("------------------------------")


        return {
            f'Precision@{self.k}': avg_precision,
            f'Recall@{self.k}': avg_recall,
            f'NDCG@{self.k}': avg_ndcg,
            'n_users_evaluated': evaluated_users
        }