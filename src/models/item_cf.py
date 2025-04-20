# src/models/item_cf.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
from pathlib import Path
from typing import List, Any, Set # Added types

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.base import BaseRecommender # Import the base class

class ItemCFRecommender(BaseRecommender): # Inherit from BaseRecommender
    """
    Recommends items based on item-item similarity (cosine similarity)
    calculated from user-item interactions. Inherits from BaseRecommender.
    """
    def __init__(self, user_col='id_student', item_col='presentation_id', score_col='implicit_feedback'):
        # Call the parent class's __init__
        super().__init__(user_col=user_col, item_col=item_col, score_col=score_col)

        # These are now primarily handled by the BaseRecommender's mappings
        # self.user_id_to_idx = {} (inherited)
        # self.item_id_to_idx = {} (inherited)
        # self.idx_to_item_id = {} (inherited)

        # User-item interaction matrix (users x items)
        self.interaction_matrix_sparse = None

        # Item-item similarity matrix
        self.item_similarity_matrix = None

        # Store training interactions per user for prediction filtering
        # (Can still be useful, but ensure keys match base mappings if used)
        self.user_train_items = defaultdict(set)

    def _create_sparse_matrix(self, df: pd.DataFrame):
        """
        Creates the sparse user-item interaction matrix using BaseRecommender mappings.
        (This replaces the standalone mapping creation in the previous version)
        """
        print("Creating user-item interaction matrix...")

        # Ensure mappings exist (should have been created by fit calling _create_mappings)
        if not self.user_id_to_idx or not self.item_id_to_idx:
             raise RuntimeError("User/Item mappings not created. Ensure _create_mappings() was called in fit().")

        # Get user and item indices using the mappings from BaseRecommender
        user_indices = df[self.user_col].map(self.user_id_to_idx).values
        item_indices = df[self.item_col].map(self.item_id_to_idx).values

        # Handle potential NaN values if mapping failed for some reason (defensive check)
        if np.isnan(user_indices).any() or np.isnan(item_indices).any():
            print("Warning: Found NaN indices after mapping. Filtering out problematic rows.")
            valid_mask = ~np.isnan(user_indices) & ~np.isnan(item_indices)
            user_indices = user_indices[valid_mask].astype(int)
            item_indices = item_indices[valid_mask].astype(int)
            scores = df.loc[valid_mask, self.score_col].values
        else:
            user_indices = user_indices.astype(int)
            item_indices = item_indices.astype(int)
            scores = df[self.score_col].values


        # Create CSR sparse matrix (Compressed Sparse Row) using n_users and n_items from Base
        self.interaction_matrix_sparse = csr_matrix(
            (scores, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items) # Use dimensions from BaseRecommender
        )
        print(f" Created sparse matrix with shape: {self.interaction_matrix_sparse.shape} and density: {self.interaction_matrix_sparse.nnz / np.prod(self.interaction_matrix_sparse.shape):.4f}")

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the ItemCF model:
        1. Creates user/item mappings using BaseRecommender method.
        2. Creates the user-item interaction matrix.
        3. Calculates the item-item cosine similarity matrix.
        4. Stores training interactions for filtering.

        Args:
            train_df (pd.DataFrame): Training interactions dataframe.
        """
        print(f"Fitting {self.__class__.__name__}...")
        if not all(col in train_df.columns for col in [self.user_col, self.item_col, self.score_col]):
            raise ValueError(f"Training DataFrame must contain columns: '{self.user_col}', '{self.item_col}', '{self.score_col}'")

        # 1. Create Mappings using BaseRecommender's helper
        # This populates self.user_id_to_idx, self.item_id_to_idx, self.n_users, self.n_items
        self._create_mappings(train_df)

        # 2. Create sparse matrix using the created mappings
        self._create_sparse_matrix(train_df)

        # 3. Calculate Item-Item Similarity
        print("Calculating item-item cosine similarity...")
        # Transpose the user-item matrix to get item-user
        item_user_matrix = self.interaction_matrix_sparse.T.tocsr()
        self.item_similarity_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        self.item_similarity_matrix.setdiag(0) # Item not similar to itself
        self.item_similarity_matrix.eliminate_zeros() # Keep it sparse
        print(f" Calculated item similarity matrix shape: {self.item_similarity_matrix.shape}")

        # 4. Store training interactions for filtering during prediction
        self.user_train_items = train_df.groupby(self.user_col)[self.item_col].agg(set)
        print("Stored training interactions for prediction filtering.")
        print("Fit complete.")

    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]: # Match types
        """
        Predicts scores for a list of items for a given user based on item similarity.
        Score = Sum(similarity(target_item, user_interacted_item) * interaction_score(user_interacted_item))

        Args:
            user_id (Any): The user ID.
            item_ids (List[Any]): A list of item IDs for which to predict scores.

        Returns:
            List[float]: A list of predicted scores corresponding to the input item_ids.
        """
        user_idx = self.user_id_to_idx.get(user_id)
        if user_idx is None:
            # print(f"Warning: User {user_id} not found in training data. Returning zeros.")
            return [0.0] * len(item_ids)

        if self.interaction_matrix_sparse is None or self.item_similarity_matrix is None:
             raise RuntimeError("Model not fitted properly. Interaction or similarity matrix missing.")

        # Get the items the user interacted with in training (sparse row vector)
        user_interactions_vector = self.interaction_matrix_sparse[user_idx]

        # Convert target item_ids to internal indices
        target_item_indices = [self.item_id_to_idx.get(iid) for iid in item_ids]

        scores = [0.0] * len(item_ids) # Initialize with floats
        for i, target_idx in enumerate(target_item_indices):
            if target_idx is None or not (0 <= target_idx < self.n_items):
                # Item was not seen during training or index is invalid
                continue

            # Get similarities between the target item and all other items
            target_item_similarities = self.item_similarity_matrix[target_idx] # Sparse row

            # Calculate the weighted sum using dot product
            try:
                score = user_interactions_vector.dot(target_item_similarities.T).toarray()[0, 0]
                scores[i] = float(score) # Ensure float
            except IndexError as e:
                 print(f"Warning: IndexError during dot product for user {user_id}, item index {target_idx}. Error: {e}. Score remains 0.")
                 # Score remains 0.0
            except Exception as e:
                 print(f"Warning: Unexpected error during dot product for user {user_id}, item index {target_idx}. Error: {e}. Score remains 0.")
                 # Score remains 0.0

        return scores

    # get_known_items and get_known_users are inherited from BaseRecommender
    # and will work correctly because self.item_id_to_idx and self.user_id_to_idx
    # are populated by self._create_mappings in fit().