# src/models/item_cf.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
from pathlib import Path

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# from src import config # Not needed directly

class ItemCFRecommender:
    """
    Recommends items based on item-item similarity (cosine similarity)
    calculated from user-item interactions.
    """
    def __init__(self, user_col='student_id', item_col='presentation_id', score_col='implicit_feedback'):
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col

        # Mappings for sparse matrix creation
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}

        # User-item interaction matrix (users x items)
        self.interaction_matrix_sparse = None

        # Item-item similarity matrix
        self.item_similarity_matrix = None

        # Store training interactions per user for prediction filtering
        self.user_train_items = defaultdict(set)

    def _create_sparse_matrix(self, df: pd.DataFrame):
        """ Creates the sparse user-item interaction matrix and mappings. """
        print("Creating user-item interaction matrix...")

        # Create contiguous user and item indices
        self.user_id_to_idx = {uid: i for i, uid in enumerate(df[self.user_col].unique())}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(df[self.item_col].unique())}
        self.idx_to_item_id = {i: iid for iid, i in self.item_id_to_idx.items()}
        print(f" Mapped {len(self.user_id_to_idx)} users and {len(self.item_id_to_idx)} items.")

        # Get user and item indices for each interaction
        user_indices = df[self.user_col].map(self.user_id_to_idx).values
        item_indices = df[self.item_col].map(self.item_id_to_idx).values
        scores = df[self.score_col].values

        # Create CSR sparse matrix (Compressed Sparse Row)
        self.interaction_matrix_sparse = csr_matrix(
            (scores, (user_indices, item_indices)),
            shape=(len(self.user_id_to_idx), len(self.item_id_to_idx))
        )
        print(f" Created sparse matrix with shape: {self.interaction_matrix_sparse.shape} and density: {self.interaction_matrix_sparse.nnz / np.prod(self.interaction_matrix_sparse.shape):.4f}")

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the ItemCF model:
        1. Creates the user-item interaction matrix.
        2. Calculates the item-item cosine similarity matrix.
        3. Stores training interactions for filtering.

        Args:
            train_df (pd.DataFrame): Training interactions dataframe.
        """
        print("Fitting ItemCFRecommender...")
        if not all(col in train_df.columns for col in [self.user_col, self.item_col, self.score_col]):
            raise ValueError(f"Training DataFrame must contain columns: '{self.user_col}', '{self.item_col}', '{self.score_col}'")

        # 1. Create sparse matrix and mappings
        self._create_sparse_matrix(train_df)

        # 2. Calculate Item-Item Similarity
        print("Calculating item-item cosine similarity...")
        # We need ITEMS x USERS matrix for item similarity based on user interactions
        # So, transpose the user-item matrix
        item_user_matrix = self.interaction_matrix_sparse.T.tocsr()
        self.item_similarity_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        # Ensure diagonal is 0 (item shouldn't be similar to itself for recs)
        self.item_similarity_matrix.setdiag(0)
        self.item_similarity_matrix.eliminate_zeros() # Keep it sparse
        print(f" Calculated item similarity matrix shape: {self.item_similarity_matrix.shape}")

        # 3. Store training interactions for filtering during prediction
        self.user_train_items = train_df.groupby(self.user_col)[self.item_col].agg(set)
        print("Stored training interactions for prediction filtering.")
        print("Fit complete.")

    def predict(self, user_id: int, item_ids: list) -> np.ndarray:
        """
        Predicts scores for a list of items for a given user based on item similarity.
        Score = Sum(similarity(target_item, user_interacted_item) * interaction_score(user_interacted_item))

        Args:
            user_id (int): The user ID.
            item_ids (list): A list of item IDs for which to predict scores.

        Returns:
            np.ndarray: An array of predicted scores corresponding to the input item_ids.
        """
        # Check if user is known (was in training data)
        if user_id not in self.user_id_to_idx:
            # print(f"Warning: User {user_id} not found in training data. Returning 0 scores.")
            return np.zeros(len(item_ids))

        user_idx = self.user_id_to_idx[user_id]

        # Get the items the user interacted with in training (sparse row vector)
        user_interactions_vector = self.interaction_matrix_sparse[user_idx]

        # Convert target item_ids to internal indices
        target_item_indices = [self.item_id_to_idx.get(iid) for iid in item_ids]

        # Calculate scores only for valid target items seen during training
        scores = np.zeros(len(item_ids))
        for i, target_idx in enumerate(target_item_indices):
            if target_idx is None:
                # Item was not seen during training, score is 0
                continue

            # Get similarities between the target item and all other items
            # item_similarity_matrix is (items x items)
            target_item_similarities = self.item_similarity_matrix[target_idx] # This is a sparse row

            # Calculate the weighted sum of similarities
            # Dot product between user's interaction vector and target item's similarity vector
            # Ensure dimensions match: user_interactions_vector (1 x n_items), target_item_similarities (1 x n_items) -> transpose one
            score = user_interactions_vector.dot(target_item_similarities.T).toarray()[0, 0]
            scores[i] = score

        return scores


# --- Example Usage ---
if __name__ == "__main__":
    data = {
        'student_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5],
        'presentation_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'A', 'B', 'D', 'A', 'C', 'D'],
        'implicit_feedback': [5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 3.5, 1.0, 4.0, 3.5, 1.5, 4.5, 1.0, 2.0]
    }
    dummy_train_df = pd.DataFrame(data)

    # Initialize and fit
    itemcf_model = ItemCFRecommender()
    itemcf_model.fit(dummy_train_df)

    # Test prediction
    test_user = 1 # Known user
    test_items = ['D', 'A', 'X'] # User 1 liked A, B, C. D is similar to C & A. X is unknown.
    predicted_scores = itemcf_model.predict(test_user, test_items)
    print(f"\nPredicting for user {test_user}, items {test_items}")
    print(f"Predicted scores: {predicted_scores}")

    test_user_unknown = 999 # Unknown user
    predicted_scores_unknown = itemcf_model.predict(test_user_unknown, test_items)
    print(f"\nPredicting for user {test_user_unknown}, items {test_items}")
    print(f"Predicted scores: {predicted_scores_unknown}") # Should be all zeros