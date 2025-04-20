# src/models/popularity.py

import pandas as pd
import numpy as np
from collections import defaultdict

import sys
from pathlib import Path

# Add project root to sys.path if needed, although not strictly necessary for this simple model
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# from src import config # Not needed directly here, but good practice potentially

class PopularityRecommender:
    """
    Recommends items based on their overall popularity in the training data.
    Popularity is measured by the sum of implicit feedback scores across all users.
    """
    def __init__(self, user_col='student_id', item_col='presentation_id', score_col='implicit_feedback'):
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.item_popularity_scores = defaultdict(float) # Stores popularity score for each item
        self.most_popular_items = [] # Sorted list of items by popularity

    def fit(self, train_df: pd.DataFrame):
        """
        Calculates item popularity based on the training data.

        Args:
            train_df (pd.DataFrame): Training interactions dataframe,
                                     expected to have item_col and score_col.
        """
        print(f"Fitting PopularityRecommender...")
        if not all(col in train_df.columns for col in [self.item_col, self.score_col]):
            raise ValueError(f"Training DataFrame must contain columns: '{self.item_col}' and '{self.score_col}'")

        # Calculate popularity: Sum of scores per item
        popularity = train_df.groupby(self.item_col)[self.score_col].sum()

        # Store scores in the defaultdict
        self.item_popularity_scores.update(popularity.to_dict())

        # Create a sorted list of items (most popular first)
        self.most_popular_items = popularity.sort_values(ascending=False).index.tolist()

        print(f"Fit complete. Calculated popularity for {len(self.item_popularity_scores)} items.")
        print(f"Top 5 most popular items: {self.most_popular_items[:5]}")


    def predict(self, user_id: int, item_ids: list) -> np.ndarray:
        """
        Predicts the "relevance" of items based on their popularity.
        The score is simply the pre-calculated popularity score.
        User ID is ignored as this model is not personalized.

        Args:
            user_id (int): The user ID (ignored).
            item_ids (list): A list of item IDs for which to predict scores.

        Returns:
            np.ndarray: An array of popularity scores corresponding to the input item_ids.
                        Items not seen during training will have a score of 0.
        """
        # Note: user_id is ignored in this non-personalized model

        scores = np.array([self.item_popularity_scores.get(item_id, 0.0) for item_id in item_ids])
        return scores

    def recommend_top_k(self, user_id: int, k: int, filter_already_liked_items: bool = False, liked_items: set = None) -> list:
        """
        Recommends the top K most popular items overall.

        Args:
            user_id (int): The user ID (ignored, but kept for consistent interface).
            k (int): The number of items to recommend.
            filter_already_liked_items (bool): If True, filter out items from the 'liked_items' set.
            liked_items (set): A set of item IDs the user has already interacted with (e.g., from training data).

        Returns:
            list: A list of the top K recommended item IDs.
        """
        if filter_already_liked_items and liked_items is not None:
            recommendations = [item for item in self.most_popular_items if item not in liked_items]
        else:
            recommendations = self.most_popular_items

        return recommendations[:k]


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data
    data = {
        'student_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
        'presentation_id': ['A_2013', 'B_2014', 'C_2013', 'A_2013', 'C_2013', 'B_2014', 'C_2013', 'D_2014', 'A_2013', 'B_2014'],
        'implicit_feedback': [5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 3.5, 1.0, 4.0, 3.5] # Example scores
    }
    dummy_train_df = pd.DataFrame(data)

    # Initialize and fit
    pop_model = PopularityRecommender()
    pop_model.fit(dummy_train_df)

    # Test predict method
    test_user = 999 # User ID doesn't matter
    test_items = ['C_2013', 'A_2013', 'D_2014', 'X_2020'] # Include an unknown item
    predicted_scores = pop_model.predict(test_user, test_items)
    print(f"\nPredicting for user {test_user}, items {test_items}")
    print(f"Predicted scores (popularity): {predicted_scores}")
    # Expected Popularity: A=5+5+4=14, B=4+4.5+3.5=12, C=3+2+3.5=8.5, D=1, X=0
    # Expected Output: [ 8.5 14.   1.   0. ]

    # Test recommend_top_k method
    top_3_recs = pop_model.recommend_top_k(test_user, k=3)
    print(f"\nTop 3 recommendations for user {test_user}: {top_3_recs}")
    # Expected Output: ['A_2013', 'B_2014', 'C_2013']

    # Test with filtering
    user_1_likes = set(dummy_train_df[dummy_train_df['student_id'] == 1]['presentation_id'])
    print(f"User 1 liked: {user_1_likes}")
    top_3_recs_filtered = pop_model.recommend_top_k(1, k=3, filter_already_liked_items=True, liked_items=user_1_likes)
    print(f"Top 3 recommendations for user 1 (filtered): {top_3_recs_filtered}")
     # Expected Output: ['D_2014'] (only 1 left after filtering A, B, C)