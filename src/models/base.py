# src/models/base.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Set, Any # Added Set, Any, List

class BaseRecommender(ABC):
    """
    Abstract base class for all recommender models.
    Requires fit and predict methods to be implemented.
    """

    def __init__(self, user_col: str, item_col: str, score_col: str = None):
        """
        Initialize base recommender.

        Args:
            user_col (str): Name of the user ID column in the interaction data.
            item_col (str): Name of the item ID column in the interaction data.
            score_col (str, optional): Name of the interaction score/value column.
                                       Defaults to None (for implicit feedback models).
        """
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
        self.n_users = 0
        self.n_items = 0
        print(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame):
        """
        Trains the recommender model on the provided interaction data.
        This method must be implemented by subclasses. It should typically
        populate the user/item mappings and learn model parameters.

        Args:
            interactions_df (pd.DataFrame): DataFrame containing user-item interactions.
                                            Must include user_col and item_col.
                                            May include score_col if relevant.
        """
        pass

    @abstractmethod
    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
        """
        Predicts recommendation scores for a given user and a list of item IDs.
        This method must be implemented by subclasses.

        Args:
            user_id (Any): The ID of the user for whom to predict scores.
            item_ids (List[Any]): A list of item IDs for which to predict scores.

        Returns:
            List[float]: A list of predicted scores corresponding to the input item_ids.
                         The order of scores must match the order of item_ids.
                         Scores should represent the likelihood or rating.
        """
        pass

    def _create_mappings(self, interactions_df: pd.DataFrame):
        """Helper function to create user and item ID to index mappings."""
        unique_users = interactions_df[self.user_col].unique()
        unique_items = interactions_df[self.item_col].unique()

        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}

        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        print(f" Mapped {self.n_users} users and {self.n_items} items.")


    def get_known_items(self) -> Set[Any]:
        """
        Returns a set of item IDs known to the model (seen during fit).
        Relies on self.item_id_to_idx being populated by fit().
        """
        if hasattr(self, 'item_id_to_idx') and self.item_id_to_idx:
            return set(self.item_id_to_idx.keys())
        else:
            print(f"Warning: {self.__class__.__name__} has not been fitted or has no item mapping. Returning empty set for known items.")
            return set()

    def get_known_users(self) -> Set[Any]:
        """
        Returns a set of user IDs known to the model (seen during fit).
        Relies on self.user_id_to_idx being populated by fit().
        """
        if hasattr(self, 'user_id_to_idx') and self.user_id_to_idx:
            return set(self.user_id_to_idx.keys())
        else:
            print(f"Warning: {self.__class__.__name__} has not been fitted or has no user mapping. Returning empty set for known users.")
            return set()

    def save_model(self, path: str):
        """Saves the model state (e.g., using pickle or specific library methods)."""
        # Implementation depends on the model type
        raise NotImplementedError(f"Save method not implemented for {self.__class__.__name__}")

    def load_model(self, path: str):
        """Loads the model state."""
        # Implementation depends on the model type
        raise NotImplementedError(f"Load method not implemented for {self.__class__.__name__}")