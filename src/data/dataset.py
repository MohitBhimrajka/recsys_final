# src/data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from pathlib import Path
import sys
import math # Import math for CFDataset if not already present

# Add project root to sys.path if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config

# --- Helper function to create mappings and get unique IDs ---
def create_mappings_and_unique_ids(interactions_df, user_col, item_col):
    """Creates user/item ID to index mappings and returns unique IDs."""
    unique_users = interactions_df[user_col].unique()
    unique_items = interactions_df[item_col].unique()

    user_id_map = {uid: i for i, uid in enumerate(unique_users)}
    item_id_map = {iid: i for i, iid in enumerate(unique_items)}

    return user_id_map, item_id_map, unique_users, unique_items


class CFDataset(Dataset):
    """
    Custom PyTorch Dataset for Collaborative Filtering models.
    Handles mapping IDs to indices and negative sampling.

    Args:
        interactions_df (pd.DataFrame): DataFrame containing user-item interactions.
        all_item_ids (list or set): Collection of all unique item IDs in the dataset,
                                    used for negative sampling.
        user_id_map (dict): Pre-computed mapping from original user ID to contiguous index.
        item_id_map (dict): Pre-computed mapping from original item ID to contiguous index.
        user_col (str): Name of the user ID column.
        item_col (str): Name of the item ID column.
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
        # Check if maps cover all IDs in the provided dataframe slice
        users_in_df = set(self.interactions_df[self.user_col].unique())
        items_in_df = set(self.interactions_df[self.item_col].unique())
        users_in_map = set(self.user_id_map.keys())
        items_in_map = set(self.item_id_map.keys())

        if not users_in_df.issubset(users_in_map):
            print(f"Warning: {len(users_in_df - users_in_map)} users in interactions_df not found in user_id_map!")
        if not items_in_df.issubset(items_in_map):
             print(f"Warning: {len(items_in_df - items_in_map)} items in interactions_df not found in item_id_map!")

        # Map original IDs in the interaction dataframe to indices, handle potential misses
        self.user_indices = self.interactions_df[self.user_col].map(self.user_id_map).values
        self.item_indices = self.interactions_df[self.item_col].map(self.item_id_map).values

        # Filter out interactions where mapping failed (returned NaN)
        valid_mask = ~np.isnan(self.user_indices) & ~np.isnan(self.item_indices)
        if (~valid_mask).sum() > 0:
            print(f"Filtering out {(~valid_mask).sum()} interactions due to missing ID mappings.")
            self.user_indices = self.user_indices[valid_mask].astype(int)
            self.item_indices = self.item_indices[valid_mask].astype(int)

        # Store positive interactions efficiently: List of tuples (user_idx, positive_item_idx)
        self.positive_interactions = list(zip(self.user_indices, self.item_indices))

        # Precompute user's positive items for fast negative sampling checks
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
            return len(self.positive_interactions) * (1 + self.num_negatives)
        else:
            return len(self.positive_interactions)

    def __getitem__(self, index):
        """ Generates one sample (user_idx, item_idx, label). """
        if not self.positive_interactions: # Handle empty dataset case
            raise IndexError("Dataset is empty")

        if self.num_negatives > 0:
            positive_idx = index // (1 + self.num_negatives)
            is_positive_sample = (index % (1 + self.num_negatives) == 0)

            # Ensure positive_idx is within bounds
            if positive_idx >= len(self.positive_interactions):
                 raise IndexError(f"Index {index} out of bounds for dataset with {len(self.positive_interactions)} positive interactions and {self.num_negatives} negatives.")

            user_idx, positive_item_idx = self.positive_interactions[positive_idx]

            if is_positive_sample:
                item_idx = positive_item_idx
                label = 1.0
            else:
                negative_item_idx = random.choice(list(self.all_item_indices))
                while negative_item_idx in self.user_to_positive_items.get(user_idx, set()):
                    negative_item_idx = random.choice(list(self.all_item_indices))
                item_idx = negative_item_idx
                label = 0.0

            return torch.tensor(user_idx, dtype=torch.long), \
                   torch.tensor(item_idx, dtype=torch.long), \
                   torch.tensor(label, dtype=torch.float32)
        else:
            # Ensure index is within bounds
            if index >= len(self.positive_interactions):
                raise IndexError(f"Index {index} out of bounds for dataset with {len(self.positive_interactions)} positive interactions.")

            user_idx, item_idx = self.positive_interactions[index]
            return torch.tensor(user_idx, dtype=torch.long), \
                   torch.tensor(item_idx, dtype=torch.long)


class HybridDataset(Dataset):
    """
    Custom PyTorch Dataset for Hybrid models (CF + Content).
    Handles mapping IDs to indices, negative sampling, and item feature lookup.

    Args:
        interactions_df (pd.DataFrame): DataFrame containing user-item interactions.
        item_features_df (pd.DataFrame): DataFrame containing item features,
                                         indexed by the original item ID (e.g., presentation_id).
        all_item_ids (list): Collection of all unique *original* item IDs in the dataset.
        user_id_map (dict): Pre-computed mapping from original user ID to contiguous index.
        item_id_map (dict): Pre-computed mapping from original item ID to contiguous index.
        user_col (str): Name of the user ID column in interactions_df.
        item_col (str): Name of the item ID column in interactions_df.
        num_negatives (int): Number of negative samples for training. Set to 0 for evaluation.
    """
    def __init__(self,
                 interactions_df: pd.DataFrame,
                 item_features_df: pd.DataFrame, # Added item features
                 all_item_ids: list, # Needs all original item IDs
                 user_id_map: dict,
                 item_id_map: dict,
                 user_col: str = 'id_student',
                 item_col: str = 'presentation_id',
                 num_negatives: int = 4):

        self.interactions_df = interactions_df
        self.item_features_df = item_features_df.astype(np.float32) # Ensure float32
        self.user_col = user_col
        self.item_col = item_col
        self.num_negatives = num_negatives

        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.all_item_indices = set(self.item_id_map.values()) # Store all possible item *indices*
        self.idx_to_item_id = {i: iid for iid, i in self.item_id_map.items()}

        # --- Prepare interaction data (similar to CFDataset) ---
        print("Preparing HybridDataset...")
        users_in_df = set(self.interactions_df[self.user_col].unique())
        items_in_df = set(self.interactions_df[self.item_col].unique())
        users_in_map = set(self.user_id_map.keys())
        items_in_map = set(self.item_id_map.keys())
        if not users_in_df.issubset(users_in_map): print(f"Warning: {len(users_in_df - users_in_map)} users in interactions_df not in user_id_map!")
        if not items_in_df.issubset(items_in_map): print(f"Warning: {len(items_in_df - items_in_map)} items in interactions_df not in item_id_map!")

        self.user_indices = self.interactions_df[self.user_col].map(self.user_id_map).values
        self.item_indices = self.interactions_df[self.item_col].map(self.item_id_map).values
        valid_mask = ~np.isnan(self.user_indices) & ~np.isnan(self.item_indices)
        if (~valid_mask).sum() > 0:
            print(f"Filtering out {(~valid_mask).sum()} interactions due to missing ID mappings.")
            self.user_indices = self.user_indices[valid_mask].astype(int)
            self.item_indices = self.item_indices[valid_mask].astype(int)
        self.positive_interactions = list(zip(self.user_indices, self.item_indices))
        self.user_to_positive_items = defaultdict(set)
        for u_idx, i_idx in self.positive_interactions: self.user_to_positive_items[u_idx].add(i_idx)

        # --- Prepare item features ---
        # Ensure item_features_df index matches item_col name
        if self.item_features_df.index.name != self.item_col:
            print(f"Warning: item_features_df index name ('{self.item_features_df.index.name}') should match item_col ('{self.item_col}').")

        # Check coverage and create feature array aligned with item indices
        items_with_features = set(self.item_features_df.index)
        missing_features = set(self.item_id_map.keys()) - items_with_features
        if missing_features:
             print(f"Warning: {len(missing_features)} items in item_id_map lack features in item_features_df: {list(missing_features)[:5]}...") # Show first few

        # Create the feature array ordered by item index (0 to n_items-1)
        self.item_features_array = np.zeros((len(self.item_id_map), self.item_features_df.shape[1]), dtype=np.float32)
        feature_miss_count = 0
        for item_id, item_idx in self.item_id_map.items():
            if item_id in items_with_features:
                try:
                    self.item_features_array[item_idx] = self.item_features_df.loc[item_id].values
                except KeyError: # Should not happen if item_id is in items_with_features
                     print(f"Error looking up features for item_id {item_id} (index {item_idx})")
                     feature_miss_count += 1
                except Exception as e:
                     print(f"Unexpected error getting features for item_id {item_id} (index {item_idx}): {e}")
                     feature_miss_count += 1
            else:
                feature_miss_count += 1 # Item doesn't have features, will be zeros

        print(f"Item features array created shape: {self.item_features_array.shape}")
        if feature_miss_count > 0:
             print(f" Note: {feature_miss_count} items had missing features (filled with zeros).")

        print(f"Dataset contains {len(self.positive_interactions)} positive interactions.")
        if self.num_negatives > 0: print(f"Generating {self.num_negatives} negative samples per positive.")
        print("HybridDataset preparation complete.")

    def __len__(self):
        """ Returns total number of samples including negatives. """
        if self.num_negatives > 0:
            return len(self.positive_interactions) * (1 + self.num_negatives)
        else:
            return len(self.positive_interactions)

    def __getitem__(self, index):
        """ Generates one sample: (user_idx, item_idx, item_features, label). """
        if not self.positive_interactions: raise IndexError("Dataset is empty")

        if self.num_negatives > 0:
            positive_idx = index // (1 + self.num_negatives)
            is_positive_sample = (index % (1 + self.num_negatives) == 0)
            if positive_idx >= len(self.positive_interactions): raise IndexError(f"Index {index} out of bounds.")

            user_idx, positive_item_idx = self.positive_interactions[positive_idx]

            if is_positive_sample:
                item_idx = positive_item_idx
                label = 1.0
            else:
                # Sample negative item index
                negative_item_idx = random.choice(list(self.all_item_indices))
                while negative_item_idx in self.user_to_positive_items.get(user_idx, set()):
                    negative_item_idx = random.choice(list(self.all_item_indices))
                item_idx = negative_item_idx
                label = 0.0

            # Fetch item features using the item index
            if 0 <= item_idx < self.item_features_array.shape[0]:
                 item_feats = self.item_features_array[item_idx]
            else:
                 print(f"Warning: Invalid item index {item_idx} encountered in getitem. Using zero features.")
                 item_feats = np.zeros(self.item_features_array.shape[1], dtype=np.float32)

            return torch.tensor(user_idx, dtype=torch.long), \
                   torch.tensor(item_idx, dtype=torch.long), \
                   torch.tensor(item_feats, dtype=torch.float32), \
                   torch.tensor(label, dtype=torch.float32)

        else: # Evaluation mode - only positives
            if index >= len(self.positive_interactions): raise IndexError(f"Index {index} out of bounds.")
            user_idx, item_idx = self.positive_interactions[index]

            if 0 <= item_idx < self.item_features_array.shape[0]:
                 item_feats = self.item_features_array[item_idx]
            else:
                 print(f"Warning: Invalid item index {item_idx} encountered in getitem (eval mode). Using zero features.")
                 item_feats = np.zeros(self.item_features_array.shape[1], dtype=np.float32)

            # Return user, item, features (no label needed for this type of eval)
            return torch.tensor(user_idx, dtype=torch.long), \
                   torch.tensor(item_idx, dtype=torch.long), \
                   torch.tensor(item_feats, dtype=torch.float32)


# --- Example Usage (Updated for HybridDataset) ---
if __name__ == '__main__':
    print("\n--- Testing HybridDataset ---")
    # Reuse dummy data from CFDataset example
    data = {
        'id_student': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 1],
        'presentation_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'A', 'B', 'D', 'A', 'C', 'D', 'A'],
        'implicit_feedback': [5.0, 4.0, 3.0, 5.0, 2.0, 4.5, 3.5, 1.0, 4.0, 3.5, 1.5, 4.5, 1.0, 2.0, 6.0]
    }
    dummy_interactions_df = pd.DataFrame(data).drop_duplicates(subset=['id_student', 'presentation_id'], keep='last')

    # Create dummy item features DataFrame (INDEX must be presentation_id)
    dummy_item_ids = dummy_interactions_df['presentation_id'].unique().tolist()
    dummy_feature_dim = 5
    dummy_features_data = np.random.rand(len(dummy_item_ids), dummy_feature_dim)
    dummy_items_df = pd.DataFrame(dummy_features_data, index=dummy_item_ids, columns=[f'feat_{i}' for i in range(dummy_feature_dim)])
    dummy_items_df.index.name = 'presentation_id' # Set index name correctly
    print("\nDummy Item Features:\n", dummy_items_df)

    # Get mappings and unique IDs
    user_map, item_map, unique_users, unique_items = create_mappings_and_unique_ids(dummy_interactions_df, 'id_student', 'presentation_id')

    # --- Test Training Mode ---
    print("\n--- Testing Hybrid Training Mode (num_negatives=1) ---")
    hybrid_train_dataset = HybridDataset(
        interactions_df=dummy_interactions_df,
        item_features_df=dummy_items_df, # Pass features df
        all_item_ids=unique_items.tolist(), # Pass original unique item IDs
        user_id_map=user_map,
        item_id_map=item_map,
        item_col='presentation_id', # Specify item col name
        num_negatives=1
    )
    print(f"Dataset length: {len(hybrid_train_dataset)}") # Should be 14 * (1 + 1) = 28

    print("Sample data points (user_idx, item_idx, features, label):")
    for i in range(4): # Show a few samples
        user, item, feats, label = hybrid_train_dataset[i]
        print(f" Sample {i}: User={user.item()}, Item={item.item()}, Feats Shape={feats.shape}, Label={label.item()}")

    # Test DataLoader
    hybrid_loader = DataLoader(hybrid_train_dataset, batch_size=4, shuffle=True)
    print("\nSample batch from Hybrid DataLoader:")
    for batch in hybrid_loader:
        users, items, feats, labels = batch
        print(" Users:", users)
        print(" Items:", items)
        print(" Feats Shape:", feats.shape) # Should be (batch_size, num_features)
        print(" Labels:", labels)
        break