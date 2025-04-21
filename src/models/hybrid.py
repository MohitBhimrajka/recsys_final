# src/models/hybrid.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
from pathlib import Path
from typing import List, Any, Set, Dict # Added Dict
import os # For cpu_count

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.base import BaseRecommender
from src.data.dataset import HybridDataset, create_mappings_and_unique_ids
from src.models.content_encoder import ContentEncoder

# --- HybridNCF nn.Module Definition (Keep As Is) ---
class HybridNCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, item_feature_dim: int,
                 cf_embedding_dim: int = 16,
                 content_embedding_dim: int = 16,
                 content_encoder_hidden_dims: list = [32, 16],
                 final_mlp_layers: list = [32, 16, 8],
                 dropout: float = 0.1):
        super(HybridNCF, self).__init__()

        print("Initializing HybridNCF Network...")
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
        # Initialize content encoder weights (handled within ContentEncoder)
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

# --- Hybrid Wrapper (Implements BaseRecommender) ---
class HybridNCFRecommender(BaseRecommender):
    # ... (Keep __init__ method as before) ...
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

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.item_feature_dim = None
        self.item_features_array = None

    # ... (Keep fit method as before) ...
    def fit(self, interactions_df: pd.DataFrame, item_features_df: pd.DataFrame):
        print(f"Fitting {self.__class__.__name__}...")
        if self.user_col not in interactions_df.columns or self.item_col not in interactions_df.columns:
             raise ValueError(f"interactions_df must contain '{self.user_col}' and '{self.item_col}'")
        if item_features_df.index.name != self.item_col:
             if self.item_col in item_features_df.columns:
                 print(f"Warning: Setting index of item_features_df to '{self.item_col}'")
                 item_features_df = item_features_df.set_index(self.item_col)
             else:
                 raise ValueError(f"item_features_df must be indexed by or contain column '{self.item_col}'")

        self._create_mappings(interactions_df)
        unique_items_list = list(self.item_id_to_idx.keys())

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

        train_dataset = HybridDataset(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            all_item_ids=unique_items_list, # Use original IDs here
            user_id_map=self.user_id_to_idx,
            item_id_map=self.item_id_to_idx,
            user_col=self.user_col,
            item_col=self.item_col,
            num_negatives=self.num_negatives
        )
        num_workers = min(4, getattr(os, 'cpu_count', lambda: 1)())
        print(f"Using {num_workers} workers for DataLoader.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print(f"\n--- Starting HybridNCF Training ({self.epochs} Epochs) ---")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for users, items, features, labels in progress_bar:
                users = users.to(self.device)
                items = items.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
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

    # ... (Keep predict method as before) ...
    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
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
            if item_idx is not None and 0 <= item_idx < self.n_items:
                 pred_item_indices.append(item_idx)
                 # Retrieve pre-processed features using the internal index
                 pred_item_features.append(self.item_features_array[item_idx])
                 original_pos_map[item_idx] = i

        if not pred_item_indices:
            return [0.0] * len(item_ids)

        user_tensor = torch.tensor([user_idx] * len(pred_item_indices), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(pred_item_indices, dtype=torch.long).to(self.device)
        # Ensure features are stacked correctly even if only one item
        if len(pred_item_features) == 1:
            feat_tensor = torch.tensor(pred_item_features[0], dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            feat_tensor = torch.tensor(np.vstack(pred_item_features), dtype=torch.float32).to(self.device)


        self.model.eval()
        with torch.no_grad():
            logits = self.model(user_tensor, item_tensor, feat_tensor)
        scores = logits.cpu().numpy()

        final_scores = [0.0] * len(item_ids)
        for idx, score in zip(pred_item_indices, scores):
             final_scores[original_pos_map[idx]] = float(score)

        return final_scores

    # --- Required Methods from BaseRecommender (Keep As Is) ---
    def get_known_items(self) -> Set[Any]:
        return set(self.item_id_to_idx.keys()) if hasattr(self, 'item_id_to_idx') else set()

    def get_known_users(self) -> Set[Any]:
        return set(self.user_id_to_idx.keys()) if hasattr(self, 'user_id_to_idx') else set()

    # --- Save/Load Methods (Keep Existing save_model) ---
    def save_model(self, file_path: str):
        """Saves the HybridNCFRecommender state."""
        if self.model is None or self.item_features_array is None:
            raise RuntimeError("Model not fitted or item features not processed. Cannot save.")

        torch.save({
            'hyperparameters': {
                'cf_embedding_dim': self.cf_embedding_dim,
                'content_embedding_dim': self.content_embedding_dim,
                'content_encoder_hidden_dims': self.content_encoder_hidden_dims,
                'final_mlp_layers': self.final_mlp_layers,
                'dropout': self.dropout,
                'lr': self.lr,
                'epochs': self.epochs, # Save actual trained epochs if needed
                'batch_size': self.batch_size,
                'num_negatives': self.num_negatives,
                'weight_decay': self.weight_decay,
                # Dimensions needed to reconstruct the model
                'n_users': self.n_users,
                'n_items': self.n_items,
                'item_feature_dim': self.item_feature_dim,
            },
            'user_id_to_idx': self.user_id_to_idx,
            'item_id_to_idx': self.item_id_to_idx,
            'item_features_array': self.item_features_array, # Save the processed features
            'model_state_dict': self.model.state_dict(),
        }, file_path)
        print(f"HybridNCFRecommender saved to {file_path}")

    # --- UPDATED load_model ---
    @classmethod
    def load_model(cls, file_path: str, device: str = 'auto'):
        """Loads the HybridNCFRecommender state."""
        if device == 'auto':
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)

        # Load checkpoint to CPU first
        # ** Explicitly set weights_only=False **
        # WARNING: Setting weights_only=False can be insecure if the checkpoint is from an untrusted source.
        try:
            print(f"Loading Hybrid checkpoint from {file_path} with weights_only=False...")
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        except AttributeError as e:
            print(f"Error loading checkpoint (possible file corruption or wrong format): {e}")
            raise RuntimeError(f"Could not load checkpoint file: {file_path}") from e
        except RuntimeError as e:
            print(f"RuntimeError loading checkpoint: {e}")
            raise RuntimeError(f"Could not load checkpoint file: {file_path}") from e


        params = checkpoint['hyperparameters']

        # Instantiate wrapper with saved hyperparams
        recommender = cls(
            cf_embedding_dim=params['cf_embedding_dim'],
            content_embedding_dim=params['content_embedding_dim'],
            content_encoder_hidden_dims=params['content_encoder_hidden_dims'],
            final_mlp_layers=params['final_mlp_layers'],
            dropout=params['dropout'],
            learning_rate=params['lr'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            num_negatives=params['num_negatives'],
            weight_decay=params['weight_decay'],
            device=str(resolved_device) # Pass device string
        )

        # Load mappings and dimensions
        recommender.user_id_to_idx = checkpoint['user_id_to_idx']
        recommender.item_id_to_idx = checkpoint['item_id_to_idx']
        recommender.n_users = params['n_users']
        recommender.n_items = params['n_items']
        recommender.item_feature_dim = params['item_feature_dim']
        recommender.idx_to_user_id = {v: k for k, v in recommender.user_id_to_idx.items()}
        recommender.idx_to_item_id = {v: k for k, v in recommender.item_id_to_idx.items()}

        # Load the pre-processed item features array
        recommender.item_features_array = checkpoint['item_features_array']

        # Instantiate the underlying HybridNCF model
        recommender.model = HybridNCF(
            n_users=recommender.n_users,
            n_items=recommender.n_items,
            item_feature_dim=recommender.item_feature_dim,
            cf_embedding_dim=params['cf_embedding_dim'],
            content_embedding_dim=params['content_embedding_dim'],
            content_encoder_hidden_dims=params['content_encoder_hidden_dims'],
            final_mlp_layers=params['final_mlp_layers'],
            dropout=params['dropout']
        )
        # Load the state dict and move to target device
        recommender.model.load_state_dict(checkpoint['model_state_dict'])
        recommender.model = recommender.model.to(resolved_device)
        recommender.model.eval() # Ensure model is in eval mode

        print(f"HybridNCFRecommender loaded from {file_path} to device {resolved_device}")
        return recommender