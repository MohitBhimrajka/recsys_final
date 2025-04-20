# src/models/popularity.py

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Any, Set # Import necessary types

import sys
from pathlib import Path

# Add project root to sys.path if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.base import BaseRecommender # Import the base class

class PopularityRecommender(BaseRecommender): # Inherit from BaseRecommender
    """
    Recommends items based on their overall popularity in the training data.
    Popularity is measured by the sum of implicit feedback scores across all users.
    Inherits from BaseRecommender.
    """
    def __init__(self, user_col='id_student', item_col='presentation_id', score_col='implicit_feedback'):
        # Call the parent class's __init__
        super().__init__(user_col=user_col, item_col=item_col, score_col=score_col)
        self.item_popularity_scores = defaultdict(float) # Stores popularity score for each item
        self.most_popular_items = [] # Sorted list of items by popularity

    def fit(self, train_df: pd.DataFrame):
        """
        Calculates item popularity based on the training data and creates mappings.

        Args:
            train_df (pd.DataFrame): Training interactions dataframe,
                                     expected to have item_col and score_col.
        """
        print(f"Fitting {self.__class__.__name__}...")
        if not all(col in train_df.columns for col in [self.user_col, self.item_col, self.score_col]):
            raise ValueError(f"Training DataFrame must contain columns: '{self.user_col}', '{self.item_col}' and '{self.score_col}'")

        # --- Create Mappings (Required by Base Class and Evaluator) ---
        # Even though popularity doesn't use users for prediction, the evaluator
        # might use get_known_users(), so we create the mappings.
        self._create_mappings(train_df) # This populates self.user_id_to_idx, self.item_id_to_idx etc.

        # --- Calculate Popularity ---
        popularity = train_df.groupby(self.item_col)[self.score_col].sum()

        # Store scores in the defaultdict
        self.item_popularity_scores.update(popularity.to_dict())

        # Create a sorted list of items (most popular first)
        self.most_popular_items = popularity.sort_values(ascending=False).index.tolist()

        # --- Consistency Check: Ensure all mapped items have a score ---
        # Add items seen in training but with 0 score sum if any exist
        for item_id in self.item_id_to_idx.keys():
            if item_id not in self.item_popularity_scores:
                self.item_popularity_scores[item_id] = 0.0
                print(f"Warning: Item {item_id} found in training but had 0 total score.")

        print(f"Fit complete. Calculated popularity for {len(self.item_popularity_scores)} items.")
        if self.most_popular_items:
            print(f"Top 5 most popular items: {self.most_popular_items[:5]}")
        else:
             print("No popular items found (possibly empty training data).")


    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]: # Match type hints from Base
        """
        Predicts the "relevance" of items based on their popularity.
        The score is simply the pre-calculated popularity score.
        User ID is ignored as this model is not personalized.

        Args:
            user_id (Any): The user ID (ignored).
            item_ids (List[Any]): A list of item IDs for which to predict scores.

        Returns:
            List[float]: A list of popularity scores corresponding to the input item_ids.
                         Items not seen during training will have a score of 0.
        """
        # Note: user_id is ignored in this non-personalized model
        # Base class provides self.item_popularity_scores which includes all items from fit
        scores = [float(self.item_popularity_scores.get(item_id, 0.0)) for item_id in item_ids]
        return scores

    # get_known_items and get_known_users are inherited from BaseRecommender
    # and will work correctly because self.item_id_to_idx and self.user_id_to_idx
    # are populated by self._create_mappings in fit().

    def recommend_top_k(self, user_id: Any, k: int, filter_already_liked_items: bool = False, liked_items: Set[Any] = None) -> List[Any]:
        """
        Recommends the top K most popular items overall.

        Args:
            user_id (Any): The user ID (ignored, but kept for consistent interface).
            k (int): The number of items to recommend.
            filter_already_liked_items (bool): If True, filter out items from the 'liked_items' set.
            liked_items (Set[Any]): A set of item IDs the user has already interacted with (from training data). Required if filter_already_liked_items is True.

        Returns:
            List[Any]: A list of the top K recommended item IDs.
        """
        if filter_already_liked_items:
            if liked_items is None:
                # In a real scenario, you might fetch this or require it
                print("Warning: filter_already_liked_items is True, but liked_items set was not provided. Returning top K overall.")
                return self.most_popular_items[:k]
            else:
                recommendations = [item for item in self.most_popular_items if item not in liked_items]
                return recommendations[:k]
        else:
            return self.most_popular_items[:k]