# src/models/matrix_factorization.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import implicit # The library for ALS
import sys
from pathlib import Path
from collections import defaultdict

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from src import config # Import config for RANDOM_SEED

class ImplicitALSWrapper:
    """
    A wrapper class for the implicit library's ALS model.
    Handles data conversion, training, and prediction/recommendation.
    Fits on Item x User matrix. Accesses factors directly from model,
    acknowledging the library's dimension-based factor storage.
    """
    def __init__(self,
                 user_col='student_id',
                 item_col='presentation_id',
                 score_col='implicit_feedback',
                 factors=64,
                 regularization=0.01,
                 iterations=20,
                 random_state=config.RANDOM_SEED):
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
        self.n_users = 0
        self.n_items = 0
        self.user_item_matrix = None
        self.model = None
        self.user_train_items = defaultdict(set)

    def _create_sparse_matrix(self, df: pd.DataFrame):
        """ Creates the sparse user-item interaction matrix and mappings. """
        print("Creating user-item interaction matrix for Implicit ALS...")
        unique_users = df[self.user_col].unique()
        unique_items = df[self.item_col].unique()
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}
        self.idx_to_item_id = {i: iid for iid, i in self.item_id_to_idx.items()}
        print(f" Mapped {self.n_users} users and {self.n_items} items.")
        user_indices = df[self.user_col].map(self.user_id_to_idx).values
        item_indices = df[self.item_col].map(self.item_id_to_idx).values
        scores = df[self.score_col].values
        self.user_item_matrix = csr_matrix(
            (scores, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items)
        )
        print(f" Created sparse matrix (Users x Items) shape: {self.user_item_matrix.shape} density: {self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape):.4f}")

    def fit(self, train_df: pd.DataFrame):
        """ Fits the Implicit ALS model on Item x User matrix. """
        print("Fitting ImplicitALSWrapper...")
        required_cols = [self.user_col, self.item_col, self.score_col]
        if not all(col in train_df.columns for col in required_cols):
            raise ValueError(f"Training DataFrame must contain columns: {required_cols}")

        # 1. Create sparse matrix and mappings
        self._create_sparse_matrix(train_df)

        # 2. Initialize the ALS model
        print(f"Initializing implicit.als.AlternatingLeastSquares with factors={self.factors}, regularization={self.regularization}, iterations={self.iterations}...")
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=implicit.gpu.HAS_CUDA
        )

        # Fit on Item x User in CSR format
        item_user_matrix_csr = self.user_item_matrix.T.tocsr()
        print(f"Fitting model on Item x User matrix shape: {item_user_matrix_csr.shape}...")
        self.model.fit(item_user_matrix_csr, show_progress=True)
        print("Model fitting complete.")

        # Sanity check the factor shapes *as returned by the library*
        # Expect user_factors to match items dim, item_factors to match users dim
        if self.model.user_factors is not None:
             print(f" Sanity Check: model.user_factors shape: {self.model.user_factors.shape} (Dimension 0 should match Items: {self.n_items})")
        if self.model.item_factors is not None:
             print(f" Sanity Check: model.item_factors shape: {self.model.item_factors.shape} (Dimension 0 should match Users: {self.n_users})")

        # 3. Store training interactions
        self.user_train_items = train_df.groupby(self.user_col)[self.item_col].agg(set)
        print("Stored training interactions for prediction filtering.")


    # --- CORRECTED PREDICT METHOD ---
    def predict(self, user_id: int, item_ids: list) -> np.ndarray:
        """
        Predicts scores using dot product of user and item factors.
        Accesses factors based on library's observed storage behavior
        (user factors are in model.item_factors, item factors in model.user_factors).
        """
        if self.model is None:
            raise RuntimeError("Model not available. Call fit() first.")
        # Check if factors exist (might be None if fit failed unexpectedly)
        if self.model.user_factors is None or self.model.item_factors is None:
             print("Warning: Model factors are not available. Returning zero scores.")
             return np.zeros(len(item_ids))


        user_idx = self.user_id_to_idx.get(user_id)
        # Check if user_idx is valid for the array holding user factors (model.item_factors)
        if user_idx is None or user_idx >= self.model.item_factors.shape[0]:
            # print(f"Debug: User {user_id} (idx {user_idx}) not found or out of bounds for model.item_factors shape {self.model.item_factors.shape}.")
            return np.zeros(len(item_ids))

        # Get user vector from the correct array
        user_vec = self.model.item_factors[user_idx]

        scores = np.zeros(len(item_ids))
        for i, item_id in enumerate(item_ids):
            item_idx = self.item_id_to_idx.get(item_id)
            # Check if item_idx is valid for the array holding item factors (model.user_factors)
            if item_idx is not None and item_idx < self.model.user_factors.shape[0]:
                # Get item vector from the correct array
                item_vec = self.model.user_factors[item_idx]
                scores[i] = user_vec.dot(item_vec)
            # else: item not seen or index mismatch, score remains 0
        return scores

    # --- CORRECTED RECOMMEND_TOP_K METHOD ---
    def recommend_top_k(self, user_id: int, k: int, filter_already_liked_items: bool = True) -> list:
        """ Recommends top K items using the implicit model's method. """
        if self.model is None or self.user_item_matrix is None:
            raise RuntimeError("Model or user_item_matrix not available. Call fit() first.")
        if self.model.item_factors is None: # Need user factors for recommend call
             print("Warning: Model factors not available. Cannot recommend.")
             return []


        user_idx = self.user_id_to_idx.get(user_id)
        # Check index against the dimension representing users in the factors (model.item_factors)
        if user_idx is None or user_idx >= self.model.item_factors.shape[0]:
             print(f"Warning: User {user_id} (idx {user_idx}) not found or out of bounds for model.item_factors shape {self.model.item_factors.shape}. Cannot recommend.")
             return []

        try:
            # Pass the USER index and the original user-item matrix (users x items)
            recommended_indices, scores = self.model.recommend(
                userid=user_idx,
                user_items=self.user_item_matrix[user_idx],
                N=k,
                filter_already_liked_items=filter_already_liked_items,
                # Ensure the library uses the correct factors internally based on userid
            )
            # Convert indices back to original item IDs
            recommended_item_ids = [self.idx_to_item_id.get(idx, None) for idx in recommended_indices]
            recommended_item_ids = [rid for rid in recommended_item_ids if rid is not None]
            return recommended_item_ids
        except IndexError as e:
             print(f"IndexError during implicit recommend for user {user_id} (idx {user_idx}): {e}")
             print(f" Check library factor shapes: UserFactors={self.model.user_factors.shape}, ItemFactors={self.model.item_factors.shape}")
             return []
        except Exception as e:
            print(f"Unexpected error during implicit recommend for user {user_id} (idx {user_idx}): {e}")
            return []

# --- Example Usage (unchanged) ---
if __name__ == "__main__":
    data = {
        'student_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5],
        'presentation_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'A', 'B', 'D', 'A', 'C', 'D'],
        'implicit_feedback': [5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 3.5, 1.0, 4.0, 3.5, 1.5, 4.5, 1.0, 2.0]
    }
    dummy_train_df = pd.DataFrame(data)
    als_model = ImplicitALSWrapper(factors=10, regularization=0.1, iterations=10, random_state=42)
    als_model.fit(dummy_train_df)
    test_user = 1
    test_items = ['D', 'A', 'X']
    pred_scores = als_model.predict(test_user, test_items)
    print(f"\nPredicting for user {test_user}, items {test_items}"); print(f"Predicted scores: {pred_scores}")
    top_recs = als_model.recommend_top_k(test_user, k=2)
    print(f"\nTop 2 recommendations for user {test_user}: {top_recs}")