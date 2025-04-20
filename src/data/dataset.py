# src/data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from pathlib import Path
import sys

# Add project root to sys.path if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config

class CFDataset(Dataset):
    """
    Custom PyTorch Dataset for Collaborative Filtering models.
    Handles mapping IDs to indices and negative sampling.

    Args:
        interactions_df (pd.DataFrame): DataFrame containing user-item interactions.
        user_col (str): Name of the user ID column.
        item_col (str): Name of the item ID column.
        all_item_ids (list or set): Collection of all unique item IDs in the dataset,
                                    used for negative sampling.
        user_id_map (dict): Pre-computed mapping from original user ID to contiguous index.
        item_id_map (dict): Pre-computed mapping from original item ID to contiguous index.
        num_negatives (int): Number of negative samples to generate for each positive instance during training.
                             Set to 0 for evaluation/prediction mode.
    """
    def __init__(self,
                 interactions_df: pd.DataFrame,
                 all_item_ids: list,
                 user_id_map: dict,
                 item_id_map: dict,
                 user_col: str = 'id_student',
                 item_col: str = 'presentation_id',
                 num_negatives: int = 4): # Common value, can be tuned

        self.interactions_df = interactions_df
        self.user_col = user_col
        self.item_col = item_col
        self.num_negatives = num_negatives

        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        # Store all item *indices* for faster sampling
        self.all_item_indices = set(item_id_map.values())

        # --- Prepare data for efficient access ---
        print("Preparing CFDataset...")
        # 1. Map original IDs in the interaction dataframe to indices
        self.user_indices = self.interactions_df[self.user_col].map(self.user_id_map).values
        self.item_indices = self.interactions_df[self.item_col].map(self.item_id_map).values

        # 2. Store positive interactions efficiently
        # List of tuples: (user_idx, positive_item_idx)
        self.positive_interactions = list(zip(self.user_indices, self.item_indices))

        # 3. Precompute user's positive items for fast negative sampling checks
        # Maps user_idx -> set of positive item_idx
        self.user_to_positive_items = defaultdict(set)
        for u_idx, i_idx in self.positive_interactions:
            self.user_to_positive_items[u_idx].add(i_idx)

        print(f"Dataset contains {len(self.positive_interactions)} positive interactions.")
        if self.num_negatives > 0:
            print(f"Generating {self.num_negatives} negative samples per positive interaction.")
        print("CFDataset preparation complete.")

    def __len__(self):
        """ Returns the total number of samples (positives + negatives). """
        if self.num_negatives > 0:
            # Each positive interaction yields 1 positive + num_negatives negative samples
            return len(self.positive_interactions) * (1 + self.num_negatives)
        else:
            # Only positive interactions (e.g., for evaluation)
            return len(self.positive_interactions)

    def __getitem__(self, index):
        """
        Generates one sample (user_idx, item_idx, label).
        Handles negative sampling if num_negatives > 0.
        """
        if self.num_negatives > 0:
            # Determine which positive interaction this index corresponds to
            positive_idx = index // (1 + self.num_negatives)
            is_positive_sample = (index % (1 + self.num_negatives) == 0)

            user_idx, positive_item_idx = self.positive_interactions[positive_idx]

            if is_positive_sample:
                item_idx = positive_item_idx
                label = 1.0
            else:
                # Sample a negative item
                negative_item_idx = random.choice(list(self.all_item_indices))
                # Ensure the sampled item is truly negative (user hasn't interacted with it)
                while negative_item_idx in self.user_to_positive_items[user_idx]:
                    negative_item_idx = random.choice(list(self.all_item_indices))
                item_idx = negative_item_idx
                label = 0.0

            return torch.tensor(user_idx, dtype=torch.long), \
                   torch.tensor(item_idx, dtype=torch.long), \
                   torch.tensor(label, dtype=torch.float32) # Use float for BCELoss

        else:
            # Only return positive interactions (user_idx, item_idx) - label is implicitly 1
            user_idx, item_idx = self.positive_interactions[index]
            return torch.tensor(user_idx, dtype=torch.long), \
                   torch.tensor(item_idx, dtype=torch.long)


# --- Helper function to create mappings and get unique IDs ---
def create_mappings_and_unique_ids(interactions_df, user_col, item_col):
    """Creates user/item ID to index mappings and returns unique IDs."""
    unique_users = interactions_df[user_col].unique()
    unique_items = interactions_df[item_col].unique()

    user_id_map = {uid: i for i, uid in enumerate(unique_users)}
    item_id_map = {iid: i for i, iid in enumerate(unique_items)}

    return user_id_map, item_id_map, unique_users, unique_items

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing CFDataset ---")
    # Create dummy interaction data (similar to aggregated interactions)
    data = {
        'id_student': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 1], # User 1 interacts with A again
        'presentation_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'A', 'B', 'D', 'A', 'C', 'D', 'A'],
        'implicit_feedback': [5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 3.5, 1.0, 4.0, 3.5, 1.5, 4.5, 1.0, 2.0, 6.0]
    }
    dummy_interactions_df = pd.DataFrame(data)
    # Remove duplicates - keep latest interaction implicitly by score in real data
    dummy_interactions_df = dummy_interactions_df.drop_duplicates(subset=['id_student', 'presentation_id'], keep='last')
    print("Dummy Interactions:\n", dummy_interactions_df)

    # Get all unique items
    all_items = dummy_interactions_df['presentation_id'].unique().tolist()
    print("\nAll Unique Items:", all_items)

    # Create mappings
    user_map, item_map, _, _ = create_mappings_and_unique_ids(dummy_interactions_df, 'id_student', 'presentation_id')
    print("\nUser Map:", user_map)
    print("Item Map:", item_map)

    # --- Test Training Mode (with negative sampling) ---
    print("\n--- Testing Training Mode (num_negatives=2) ---")
    train_dataset = CFDataset(
        interactions_df=dummy_interactions_df,
        all_item_ids=all_items,
        user_id_map=user_map,
        item_id_map=item_map,
        num_negatives=2
    )
    print(f"Dataset length: {len(train_dataset)}") # Should be 14 * (1 + 2) = 42

    # Get a few samples
    print("Sample data points (user_idx, item_idx, label):")
    for i in range(6):
        print(f" Sample {i}: {train_dataset[i]}")

    # Test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print("\nSample batch from DataLoader:")
    for batch in train_loader:
        users, items, labels = batch
        print(" Users:", users)
        print(" Items:", items)
        print(" Labels:", labels)
        break # Show only one batch

    # --- Test Evaluation Mode (no negative sampling) ---
    print("\n--- Testing Evaluation Mode (num_negatives=0) ---")
    eval_dataset = CFDataset(
        interactions_df=dummy_interactions_df,
        all_item_ids=all_items,
        user_id_map=user_map,
        item_id_map=item_map,
        num_negatives=0
    )
    print(f"Dataset length: {len(eval_dataset)}") # Should be 14

    # Get a few samples
    print("Sample data points (user_idx, item_idx):")
    for i in range(5):
        print(f" Sample {i}: {eval_dataset[i]}")