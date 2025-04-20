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
    """

    def __init__(self, user_col: str, item_col: str, score_col: str,
                 factors: int = 100, regularization: float = 0.01,
                 iterations: int = 15, random_state: int = None,
                 use_gpu: bool = False, calculate_training_loss: bool = False):
        """
        Initialize the Implicit ALS Wrapper.

        Args:
            user_col (str): Name of the user ID column.
            item_col (str): Name of the item ID column.
            score_col (str): Name of the column representing interaction strength/confidence.
            factors (int): Number of latent factors to compute.
            regularization (float): Regularization parameter.
            iterations (int): Number of ALS iterations to run.
            random_state (int, optional): Seed for the random number generator. Defaults to None.
            use_gpu (bool): Whether to use GPU for calculations (if available and implicit is built with GPU support).
            calculate_training_loss (bool): Whether to calculate the training loss at each iteration.
        """
        super().__init__(user_col, item_col, score_col)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.calculate_training_loss = calculate_training_loss
        self.model = None # This will hold the trained implicit ALS model

    def fit(self, interactions_df: pd.DataFrame):
        """
        Fit the ALS model to the interaction data.

        Args:
            interactions_df (pd.DataFrame): DataFrame with user-item interactions.
                                            Must contain user_col, item_col, and score_col.
        """
        print(f"Fitting {self.__class__.__name__}...")
        if self.score_col not in interactions_df.columns:
             raise ValueError(f"Score column '{self.score_col}' not found in interactions_df.")

        # 1. Create Mappings (using BaseRecommender's helper)
        self._create_mappings(interactions_df)

        # 2. Create sparse user-item interaction matrix (COO format)
        print("Creating user-item interaction matrix for Implicit ALS...")
        user_indices = interactions_df[self.user_col].map(self.user_id_to_idx).values
        item_indices = interactions_df[self.item_col].map(self.item_id_to_idx).values
        scores = interactions_df[self.score_col].values

        # Create User x Item matrix (implicit expects CSR)
        user_item_matrix_coo = coo_matrix(
            (scores, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items)
        )
        user_item_matrix_csr = user_item_matrix_coo.tocsr()
        print(f" Created sparse matrix (Users x Items) shape: {user_item_matrix_csr.shape} density: {user_item_matrix_csr.nnz / (self.n_users * self.n_items):.4f}")

        # 3. Initialize and Train the implicit ALS model
        print(f"Initializing implicit.als.AlternatingLeastSquares with factors={self.factors}, regularization={self.regularization}, iterations={self.iterations}...")
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            calculate_training_loss=self.calculate_training_loss
        )

        # Fit the model (implicit expects Item x User if using default, or User x Item if passing csr_matrix directly)
        # We pass User x Item CSR matrix directly
        print(f"Fitting model on User x Item matrix shape: {user_item_matrix_csr.shape}...")
        self.model.fit(user_item_matrix_csr)

        # --- DEBUG PRINT FACTOR SHAPES POST-FIT ---
        if hasattr(self.model, 'user_factors') and hasattr(self.model, 'item_factors'):
            print(f"DEBUG FIT: self.model.user_factors shape: {self.model.user_factors.shape}") # Should be (n_users, factors)
            print(f"DEBUG FIT: self.model.item_factors shape: {self.model.item_factors.shape}") # Should be (n_items, factors)
        else:
            print("DEBUG FIT: Warning - Model factors not found immediately after fit.")
        # --- END DEBUG PRINT ---

        print("Model fitting complete.")


    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
        """
        Predicts scores for a given user and a list of item IDs using the trained ALS model.
        Handles users or items not seen during training by returning zero scores.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if not hasattr(self.model, 'user_factors') or not hasattr(self.model, 'item_factors'):
             raise RuntimeError("Model factors not found. Fit may have failed.")

        # --- DEBUG PRINTS ---
        print(f"\n--- Predicting for User: {user_id} ---")
        print(f"DEBUG PREDICT: Expected n_users: {self.n_users}, Expected n_items: {self.n_items}")
        print(f"DEBUG PREDICT: Actual model.user_factors shape: {self.model.user_factors.shape}")
        print(f"DEBUG PREDICT: Actual model.item_factors shape: {self.model.item_factors.shape}")
        # --- END DEBUG PRINTS ---

        # Get the internal integer index for the user
        user_idx = self.user_id_to_idx.get(user_id)
        print(f"DEBUG PREDICT: Mapped User ID {user_id} to index: {user_idx}") # DEBUG

        # Handle case where user was not in training data
        if user_idx is None:
            print(f"-> User {user_id} not found in training data. Returning zeros.")
            return np.zeros(len(item_ids)).tolist() # Return list of floats

        # --- DEBUG CHECK USER INDEX ---
        if not (0 <= user_idx < self.n_users):
             print(f"!!! ERROR: User index {user_idx} is out of bounds for n_users={self.n_users} !!!")
             # Optionally raise an error here or return zeros
             # raise ValueError(f"Invalid user index {user_idx} derived for user {user_id}")
             return np.zeros(len(item_ids)).tolist()
        # --- END DEBUG CHECK USER INDEX ---


        # Map requested item IDs to internal indices, keeping track of valid ones
        item_idxs_mapped = []
        valid_input_indices = [] # Store indices *from the input list* that were valid
        print(f"DEBUG PREDICT: Mapping requested item IDs: {item_ids}") # DEBUG
        for i, item_id in enumerate(item_ids):
            item_idx = self.item_id_to_idx.get(item_id)
            # print(f"  Item ID: {item_id} -> Index: {item_idx}") # DEBUG (can be verbose)
            if item_idx is not None:
                # Ensure the index is within the bounds of the learned factors
                if 0 <= item_idx < self.n_items:
                    item_idxs_mapped.append(item_idx)
                    valid_input_indices.append(i)
                    # print(f"    -> Added valid index {item_idx} (original input index {i})") # DEBUG (can be verbose)
                else:
                     print(f"    -> WARNING: Mapped index {item_idx} for item {item_id} out of bounds (0-{self.n_items-1}). Skipping.") # DEBUG

        print(f"DEBUG PREDICT: Final mapped item indices: {item_idxs_mapped}") # DEBUG
        print(f"DEBUG PREDICT: Corresponding original input indices: {valid_input_indices}") # DEBUG

        # Handle case where none of the requested items were known or valid
        if not item_idxs_mapped:
            print(f"-> No valid/known items to score for user {user_id}. Returning zeros.")
            return np.zeros(len(item_ids)).tolist() # Return list of floats

        # Calculate scores ONLY for the known, mapped items
        try:
            # --- DEBUG PRINT BEFORE ACCESSING FACTORS ---
            print(f"DEBUG PREDICT: Accessing user_factors at index: {user_idx}")
            # --- END DEBUG PRINT ---

            # Get user factor and item factors for the valid items
            # **** THIS IS THE LINE FROM THE TRACEBACK ****
            user_vector = self.model.user_factors[user_idx]
            # **** END OF TRACEBACK LINE ****

            print(f"DEBUG PREDICT: Successfully accessed user_vector shape: {user_vector.shape}") # DEBUG

            print(f"DEBUG PREDICT: Accessing item_factors at indices: {item_idxs_mapped}") # DEBUG
            item_vectors = self.model.item_factors[item_idxs_mapped]
            print(f"DEBUG PREDICT: Successfully accessed item_vectors shape: {item_vectors.shape}") # DEBUG


            # Calculate scores (dot product)
            scores_valid = user_vector @ item_vectors.T # Should be shape (1, num_valid_items) -> squeeze or flatten
            print(f"DEBUG PREDICT: Calculated scores_valid: {scores_valid}") # DEBUG


        except IndexError as e:
            # This catch is an extra safety net; the checks above should prevent most cases
            print(f"!!! FATAL IndexError during score calculation for user {user_id} (idx {user_idx}), items {item_idxs_mapped}. Error: {e}")
            # Re-raise the error after printing debug info to halt execution and examine
            raise e
            # return np.zeros(len(item_ids)).tolist() # Return list of floats
        except Exception as e:
             print(f"!!! FATAL Unexpected error during score calculation for user {user_id}: {e}")
             # Re-raise the error
             raise e
             # return np.zeros(len(item_ids)).tolist()

        # Create the final scores list, placing calculated scores in the correct positions
        # corresponding to the original input item_ids list. Fill others with 0.
        final_scores = [0.0] * len(item_ids) # Initialize with floats
        for i, score in zip(valid_input_indices, scores_valid):
            final_scores[i] = float(score) # Ensure it's a float

        # print(f"-> Final scores returned: {final_scores}") # DEBUG (can be verbose)
        return final_scores

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