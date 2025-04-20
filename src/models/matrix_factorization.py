# src/models/matrix_factorization.py

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import implicit # The library for ALS/BPR

from .base import BaseRecommender
from typing import List, Any, Set

class ImplicitALSWrapper(BaseRecommender):
    """
    Wrapper for the Alternating Least Squares (ALS) algorithm from the 'implicit' library.
    Handles user-item interaction data and provides predictions.
    Uses model.rank_items for prediction.
    """

    def __init__(self, user_col: str, item_col: str, score_col: str,
                 factors: int = 100, regularization: float = 0.01,
                 iterations: int = 15, random_state: int = None,
                 use_gpu: bool = False, calculate_training_loss: bool = False):
        super().__init__(user_col, item_col, score_col)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.calculate_training_loss = calculate_training_loss
        self.model = None
        # --- Store the user-item matrix needed for rank_items ---
        self.user_item_matrix_sparse = None

    def fit(self, interactions_df: pd.DataFrame):
        """ Fits the ALS model and stores the interaction matrix. """
        print(f"Fitting {self.__class__.__name__}...")
        if self.score_col not in interactions_df.columns:
             raise ValueError(f"Score column '{self.score_col}' not found in interactions_df.")

        self._create_mappings(interactions_df)

        print("Creating user-item interaction matrix for Implicit ALS...")
        user_indices = interactions_df[self.user_col].map(self.user_id_to_idx).values
        item_indices = interactions_df[self.item_col].map(self.item_id_to_idx).values
        scores = interactions_df[self.score_col].values

        user_item_matrix_coo = coo_matrix(
            (scores, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items)
        )
        # --- Store the CSR matrix ---
        self.user_item_matrix_sparse = user_item_matrix_coo.tocsr()
        print(f" Created sparse matrix (Users x Items) shape: {self.user_item_matrix_sparse.shape} density: {self.user_item_matrix_sparse.nnz / (self.n_users * self.n_items):.4f}")

        print(f"Initializing implicit.als.AlternatingLeastSquares...")
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            calculate_training_loss=self.calculate_training_loss
        )

        # Pass the CSR matrix directly
        print(f"Fitting model on User x Item matrix shape: {self.user_item_matrix_sparse.shape}...")
        self.model.fit(self.user_item_matrix_sparse)

        print("Model fitting complete.")


    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
        """
        Predicts scores for a given user and a list of item IDs using model.rank_items.
        """
        if self.model is None or self.user_item_matrix_sparse is None:
            raise RuntimeError("Model has not been fitted yet or interaction matrix not stored. Call fit() first.")

        user_idx = self.user_id_to_idx.get(user_id)

        if user_idx is None:
            # print(f"User {user_id} not in training data. Returning zeros.")
            return [0.0] * len(item_ids)

        # Map input item_ids to internal indices, keeping only known items
        item_idxs_to_rank = []
        original_pos_map = {} # Map internal index back to original position in item_ids list
        for i, item_id in enumerate(item_ids):
            item_idx = self.item_id_to_idx.get(item_id)
            if item_idx is not None and 0 <= item_idx < self.n_items:
                 item_idxs_to_rank.append(item_idx)
                 original_pos_map[item_idx] = i

        if not item_idxs_to_rank:
            # print(f"No known items to rank for user {user_id}. Returning zeros.")
            return [0.0] * len(item_ids)

        # Get the user's row from the sparse interaction matrix
        user_items_row = self.user_item_matrix_sparse[user_idx]

        try:
            # Use rank_items to get scores for the specific items
            ranked_indices, ranked_scores = self.model.rank_items(
                user_idx,
                user_items_row,
                selected_items=item_idxs_to_rank
            )

            # Create a dictionary mapping the ranked internal index to its score
            score_map = dict(zip(ranked_indices, ranked_scores))

            # Reconstruct the score list in the original order of item_ids
            final_scores = [0.0] * len(item_ids)
            for internal_idx, original_idx in original_pos_map.items():
                # Get the score for this internal index from the ranked results
                # If an item wasn't ranked (e.g., score was effectively zero or filtered?), default to 0.0
                final_scores[original_idx] = float(score_map.get(internal_idx, 0.0))

            return final_scores

        except IndexError as e:
            print(f"!!! IndexError during rank_items for user {user_id} (idx {user_idx}), items_to_rank {item_idxs_to_rank}. Error: {e}")
            # It's still possible rank_items has index issues, return zeros as fallback
            return [0.0] * len(item_ids)
        except Exception as e:
             print(f"!!! Unexpected error during rank_items for user {user_id}: {e}")
             # Re-raise unexpected errors
             raise e


    def get_known_items(self) -> Set[Any]:
        """Returns a set of item IDs known to the model (seen during fit)."""
        if hasattr(self, 'item_id_to_idx') and self.item_id_to_idx:
            return set(self.item_id_to_idx.keys())
        return set()

    def get_known_users(self) -> Set[Any]:
        """Returns a set of user IDs known to the model (seen during fit)."""
        if hasattr(self, 'user_id_to_idx') and self.user_id_to_idx:
            return set(self.user_id_to_idx.keys())
        return set()