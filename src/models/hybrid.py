# src/models/hybrid.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Use tqdm for progress bars
import sys
from pathlib import Path
from typing import List, Any, Set # Import necessary types

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the base class, datasets, ContentEncoder
from src.models.base import BaseRecommender
from src.data.dataset import HybridDataset, create_mappings_and_unique_ids
from src.models.content_encoder import ContentEncoder # Assumes content_encoder.py is correct

# ------------------------------------------
# --- Original HybridNCF PyTorch Module (Neural Network Definition) ---
# ------------------------------------------
class HybridNCF(nn.Module):
    """
    A Hybrid Recommendation Model combining Collaborative Filtering (CF) embeddings
    and Content-Based (CB) item embeddings.
    (Code identical to your provided version, but corrected print statements slightly)
    """
    def __init__(self, n_users: int, n_items: int, item_feature_dim: int,
                 cf_embedding_dim: int = 16,
                 content_embedding_dim: int = 16,
                 content_encoder_hidden_dims: list = [32, 16],
                 final_mlp_layers: list = [32, 16, 8],
                 dropout: float = 0.1):
        super(HybridNCF, self).__init__()

        print("Initializing HybridNCF Network...") # Changed print slightly
        self.n_users = n_users
        self.n_items = n_items
        self.item_feature_dim = item_feature_dim
        self.cf_embedding_dim = cf_embedding_dim
        self.content_embedding_dim = content_embedding_dim

        self.cf_user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=cf_embedding_dim)
        self.cf_item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=cf_embedding_dim)

        self.content_encoder = ContentEncoder(
            input_dim=item_feature_dim, embedding_dim=content_embedding_dim,
            hidden_dims=content_encoder_hidden_dims, dropout=dropout
        )

        final_mlp_input_dim = cf_embedding_dim + cf_embedding_dim + content_embedding_dim
        self.final_mlp = nn.Sequential()
        input_size = final_mlp_input_dim
        for i, layer_size in enumerate(final_mlp_layers):
            self.final_mlp.add_module(f"final_mlp_linear_{i}", nn.Linear(input_size, layer_size))
            self.final_mlp.add_module(f"final_mlp_relu_{i}", nn.ReLU())
            if dropout > 0: self.final_mlp.add_module(f"final_mlp_dropout_{i}", nn.Dropout(p=dropout))
            input_size = layer_size
        self.final_layer = nn.Linear(input_size, 1)

        self._init_weights()
        print("HybridNCF Network Initialized.")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.cf_user_embedding.weight)
        nn.init.xavier_uniform_(self.cf_item_embedding.weight)
        for layer in self.final_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.final_layer.weight)
        if self.final_layer.bias is not None: nn.init.zeros_(self.final_layer.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_cf_emb = self.cf_user_embedding(user_indices)
        item_cf_emb = self.cf_item_embedding(item_indices)
        item_content_emb = self.content_encoder(item_features)
        combined_vector = torch.cat([user_cf_emb, item_cf_emb, item_content_emb], dim=1)
        mlp_output = self.final_mlp(combined_vector)
        logits = self.final_layer(mlp_output)
        return logits.squeeze(-1)

# ------------------------------------------------
# --- Hybrid Wrapper (Implements BaseRecommender Interface) ---
# ------------------------------------------------
class HybridNCFRecommender(BaseRecommender):
    """
    Wrapper for the HybridNCF model to align with the BaseRecommender interface
    and handle the PyTorch training loop with item features.
    """
    def __init__(self,
                 user_col='id_student',
                 item_col='presentation_id',
                 score_col='implicit_feedback', # Not directly used but identifies positive interactions
                 # Model Hyperparameters
                 cf_embedding_dim: int = 16,
                 content_embedding_dim: int = 16,
                 content_encoder_hidden_dims: list = [32, 16],
                 final_mlp_layers: list = [32, 16, 8],
                 dropout: float = 0.1,
                 # Training Hyperparameters
                 learning_rate: float = 0.001,
                 epochs: int = 10,
                 batch_size: int = 512,
                 num_negatives: int = 4,
                 weight_decay: float = 1e-5,
                 device: str = 'auto'):

        super().__init__(user_col=user_col, item_col=item_col, score_col=score_col)

        # Store hyperparameters
        self.cf_embedding_dim = cf_embedding_dim
        self.content_embedding_dim = content_embedding_dim
        self.content_encoder_hidden_dims = content_encoder_hidden_dims
        self.final_mlp_layers = final_mlp_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.weight_decay = weight_decay

        # Determine device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Placeholders
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.item_feature_dim = None # Determined during fit
        self.item_features_array = None # Store features numpy array aligned with item_idx


    def fit(self, interactions_df: pd.DataFrame, item_features_df: pd.DataFrame):
        """
        Trains the HybridNCF model.
        1. Creates mappings.
        2. Processes and stores item features.
        3. Initializes the HybridNCF network.
        4. Sets up Dataset, DataLoader, Optimizer.
        5. Runs the training loop.

        Args:
            interactions_df (pd.DataFrame): User-item interaction data.
            item_features_df (pd.DataFrame): Item content features, indexed by item_col.
        """
        print(f"Fitting {self.__class__.__name__}...")
        if self.user_col not in interactions_df.columns or self.item_col not in interactions_df.columns:
             raise ValueError(f"interactions_df must contain '{self.user_col}' and '{self.item_col}'")
        if item_features_df.index.name != self.item_col:
            raise ValueError(f"item_features_df must be indexed by '{self.item_col}'")

        # 1. Create Mappings (populates self.n_users, self.n_items, mappings)
        self._create_mappings(interactions_df)
        unique_items_list = list(self.item_id_to_idx.keys())

        # 2. Process and Store Item Features aligned with item indices
        self.item_feature_dim = item_features_df.shape[1]
        print(f" Determined item feature dimension: {self.item_feature_dim}")
        self.item_features_array = np.zeros((self.n_items, self.item_feature_dim), dtype=np.float32)
        feature_miss_count = 0
        items_with_features = set(item_features_df.index)
        for item_id, item_idx in self.item_id_to_idx.items():
            if item_id in items_with_features:
                try: self.item_features_array[item_idx] = item_features_df.loc[item_id].values
                except Exception as e: print(f"Error getting features for {item_id}: {e}"); feature_miss_count+=1
            else: feature_miss_count += 1
        if feature_miss_count > 0: print(f" Warning: {feature_miss_count} items had missing features (using zeros).")


        # 3. Initialize HybridNCF Network
        self.model = HybridNCF(
            n_users=self.n_users,
            n_items=self.n_items,
            item_feature_dim=self.item_feature_dim,
            cf_embedding_dim=self.cf_embedding_dim,
            content_embedding_dim=self.content_embedding_dim,
            content_encoder_hidden_dims=self.content_encoder_hidden_dims,
            final_mlp_layers=self.final_mlp_layers,
            dropout=self.dropout
        ).to(self.device)

        # 4. Setup Dataset, DataLoader, Optimizer
        # Pass the original item_features_df to the Dataset constructor
        train_dataset = HybridDataset(
            interactions_df=interactions_df,
            item_features_df=item_features_df, # Pass the original DF here
            all_item_ids=unique_items_list,
            user_id_map=self.user_id_to_idx,
            item_id_map=self.item_id_to_idx,
            user_col=self.user_col,
            item_col=self.item_col,
            num_negatives=self.num_negatives
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 5. Training Loop
        print(f"\n--- Starting HybridNCF Training ({self.epochs} Epochs) ---")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            # Correctly unpack the batch from HybridDataset
            for users, items, features, labels in progress_bar:
                users = users.to(self.device)
                items = items.to(self.device)
                features = features.to(self.device) # Send features to device
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                # Pass features to the model's forward method
                logits = self.model(users, items, features)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {avg_epoch_loss:.4f}")

        print("--- HybridNCF Training Finished ---")
        self.model.eval()

    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
        """ Predicts scores using the trained HybridNCF model. """
        if self.model is None or self.item_features_array is None:
            raise RuntimeError("Model not fitted or item features not processed. Call fit() first.")
        if user_id not in self.user_id_to_idx:
            return [0.0] * len(item_ids)

        user_idx = self.user_id_to_idx[user_id]

        pred_item_indices = []
        pred_item_features = []
        original_pos_map = {}

        for i, iid in enumerate(item_ids):
            item_idx = self.item_id_to_idx.get(iid)
            if item_idx is not None and 0 <= item_idx < self.n_items: # Check if item is known and valid index
                 pred_item_indices.append(item_idx)
                 # Fetch features from the stored numpy array using the item index
                 pred_item_features.append(self.item_features_array[item_idx])
                 original_pos_map[item_idx] = i

        if not pred_item_indices:
            return [0.0] * len(item_ids)

        # Convert lists to tensors
        user_tensor = torch.tensor([user_idx] * len(pred_item_indices), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(pred_item_indices, dtype=torch.long).to(self.device)
        feat_tensor = torch.tensor(np.vstack(pred_item_features), dtype=torch.float32).to(self.device) # Stack features into batch

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(user_tensor, item_tensor, feat_tensor)
        scores = logits.cpu().numpy()

        # Reconstruct output list
        final_scores = [0.0] * len(item_ids)
        for idx, score in zip(pred_item_indices, scores):
            final_scores[original_pos_map[idx]] = float(score)

        return final_scores


    # --- Required Methods from BaseRecommender ---
    def get_known_items(self) -> Set[Any]:
        return set(self.item_id_to_idx.keys())

    def get_known_users(self) -> Set[Any]:
        return set(self.user_id_to_idx.keys())

    # Optional: Implement save/load similar to NCFRecommender, making sure to save/load
    # item_feature_dim and potentially the item_features_array (or recalculate on load).