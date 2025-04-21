# src/models/ncf.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Use tqdm for progress bars
import sys
from pathlib import Path
from typing import Union, List, Any, Set # Import necessary types
import os # For cpu_count

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.base import BaseRecommender # Import base class
from src.data.dataset import CFDataset, create_mappings_and_unique_ids # Import dataset stuff

# --- NCF nn.Module Definition (No Changes) ---
class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) Model.
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    to learn user-item interactions.
    """
    def __init__(self, n_users: int, n_items: int, mf_dim: int = 16, mlp_layers: list = [32, 16, 8], mlp_embedding_dim: int = 16, dropout: float = 0.1):
        super(NCF, self).__init__()

        print("Initializing NCF Network...") # Changed print slightly
        self.n_users = n_users
        self.n_items = n_items
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers
        self.mlp_embedding_dim = mlp_embedding_dim
        self.dropout = dropout

        # --- GMF Components ---
        self.mf_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.mf_dim)
        self.mf_embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.mf_dim)

        # --- MLP Components ---
        self.mlp_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.mlp_embedding_dim)
        self.mlp_embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.mlp_embedding_dim)

        self.mlp = nn.Sequential()
        input_size = 2 * self.mlp_embedding_dim
        for i, layer_size in enumerate(self.mlp_layers):
            self.mlp.add_module(f"mlp_linear_{i}", nn.Linear(input_size, layer_size))
            self.mlp.add_module(f"mlp_relu_{i}", nn.ReLU())
            if self.dropout > 0:
                 self.mlp.add_module(f"mlp_dropout_{i}", nn.Dropout(p=self.dropout))
            input_size = layer_size

        # --- NeuMF Fusion Layer ---
        neumf_input_dim = self.mf_dim + self.mlp_layers[-1]
        self.neumf_layer = nn.Linear(neumf_input_dim, 1)

        self._init_weights()
        print("NCF Network Initialized.")


    def _init_weights(self):
        nn.init.xavier_uniform_(self.mf_embedding_user.weight)
        nn.init.xavier_uniform_(self.mf_embedding_item.weight)
        nn.init.xavier_uniform_(self.mlp_embedding_user.weight)
        nn.init.xavier_uniform_(self.mlp_embedding_item.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.neumf_layer.weight)
        if self.neumf_layer.bias is not None: nn.init.zeros_(self.neumf_layer.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        mf_user_embedding = self.mf_embedding_user(user_indices)
        mf_item_embedding = self.mf_embedding_item(item_indices)
        gmf_output = mf_user_embedding * mf_item_embedding
        mlp_user_embedding = self.mlp_embedding_user(user_indices)
        mlp_item_embedding = self.mlp_embedding_item(item_indices)
        mlp_input = torch.cat((mlp_user_embedding, mlp_item_embedding), dim=-1)
        mlp_output = self.mlp(mlp_input)
        neumf_input = torch.cat((gmf_output, mlp_output), dim=-1)
        logits = self.neumf_layer(neumf_input)
        return logits.squeeze(-1)

    def predict(self, user_indices: Union[torch.Tensor, np.ndarray, int],
                      item_indices: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            current_device = next(self.parameters()).device
            if isinstance(user_indices, int):
                user_tensor = torch.tensor([user_indices] * len(item_indices), dtype=torch.long).to(current_device)
            elif isinstance(user_indices, np.ndarray):
                 user_tensor = torch.from_numpy(user_indices).long().to(current_device)
            else: user_tensor = user_indices.long().to(current_device)
            if isinstance(item_indices, list):
                 item_tensor = torch.tensor(item_indices, dtype=torch.long).to(current_device)
            elif isinstance(item_indices, np.ndarray):
                 item_tensor = torch.from_numpy(item_indices).long().to(current_device)
            else: item_tensor = item_indices.long().to(current_device)
            if user_tensor.shape != item_tensor.shape and user_tensor.ndim == 1 and item_tensor.ndim == 1:
                 if user_tensor.shape[0] == 1 and item_tensor.shape[0] > 1: user_tensor = user_tensor.repeat(item_tensor.shape[0])
                 elif item_tensor.shape[0] == 1 and user_tensor.shape[0] > 1: item_tensor = item_tensor.repeat(user_tensor.shape[0])
            logits = self.forward(user_tensor, item_tensor)
            return logits.cpu().numpy()

# --- NCF Wrapper (Implements BaseRecommender Interface) ---
class NCFRecommender(BaseRecommender):
    # --- __init__ (No Changes) ---
    def __init__(self,
                 user_col='id_student',
                 item_col='presentation_id',
                 score_col='implicit_feedback', # Used to identify positive interactions
                 mf_dim: int = 16,
                 mlp_layers: list = [32, 16, 8],
                 mlp_embedding_dim: int = 16,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 epochs: int = 10,
                 batch_size: int = 1024,
                 num_negatives: int = 4,
                 weight_decay: float = 1e-5,
                 device: str = 'auto'):

        super().__init__(user_col=user_col, item_col=item_col, score_col=score_col)

        # Store hyperparameters
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers
        self.mlp_embedding_dim = mlp_embedding_dim
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

        # Initialize model and optimizer placeholders (created during fit)
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()

    # --- fit (No Changes) ---
    def fit(self, interactions_df: pd.DataFrame):
        """
        Trains the NCF model.
        1. Creates mappings.
        2. Initializes the NCF network.
        3. Sets up Dataset, DataLoader, Optimizer.
        4. Runs the training loop.
        """
        print(f"Fitting {self.__class__.__name__}...")
        if self.user_col not in interactions_df.columns or self.item_col not in interactions_df.columns:
             raise ValueError(f"interactions_df must contain '{self.user_col}' and '{self.item_col}'")

        # 1. Create Mappings (populates self.n_users, self.n_items, mappings)
        self._create_mappings(interactions_df)
        unique_items_list = list(self.item_id_to_idx.keys())

        # 2. Initialize NCF Network
        self.model = NCF(
            n_users=self.n_users,
            n_items=self.n_items,
            mf_dim=self.mf_dim,
            mlp_layers=self.mlp_layers,
            mlp_embedding_dim=self.mlp_embedding_dim,
            dropout=self.dropout
        ).to(self.device)

        # 3. Setup Dataset, DataLoader, Optimizer
        train_dataset = CFDataset(
            interactions_df=interactions_df,
            all_item_ids=unique_items_list, # Pass original item IDs
            user_id_map=self.user_id_to_idx,
            item_id_map=self.item_id_to_idx,
            user_col=self.user_col,
            item_col=self.item_col,
            num_negatives=self.num_negatives
        )
        # Determine num_workers based on CPU cores available, max 4
        num_workers = min(4, getattr(os, 'cpu_count', lambda: 1)()) # Use os.cpu_count if available
        print(f"Using {num_workers} workers for DataLoader.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False # Check device type
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 4. Training Loop
        print(f"\n--- Starting NCF Training ({self.epochs} Epochs) ---")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for users, items, labels in progress_bar:
                users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(users, items)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {avg_epoch_loss:.4f}")

        print("--- NCF Training Finished ---")
        self.model.eval() # Set model to eval mode after training

    # --- predict (FIXED user_id type conversion) ---
    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
        """
        Predicts scores using the trained NCF model.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # --- FIX: Ensure user_id is the correct type for dictionary lookup ---
        try:
            lookup_user_id = int(user_id) # Convert to standard int
        except (ValueError, TypeError):
            print(f"Warning: Could not convert user_id '{user_id}' to int. Returning 0 scores.")
            return [0.0] * len(item_ids)
        # ---------------------------------------------------------------------

        # Use the converted ID for lookup
        user_idx = self.user_id_to_idx.get(lookup_user_id)
        if user_idx is None:
            # print(f"Warning: User {lookup_user_id} not seen during training. Returning 0 scores.") # Keep commented
            return [0.0] * len(item_ids)

        # Map item IDs to indices, keeping track of original positions and unknowns
        pred_item_indices = []
        original_pos_map = {} # Maps internal index back to original position
        known_item_indices = [] # List to store indices of known items

        for i, iid in enumerate(item_ids):
            # --- FIX: Ensure item_id is the correct type (string) ---
            item_idx = self.item_id_to_idx.get(str(iid)) # Convert to string just in case
            # ------------------------------------------------------
            if item_idx is not None: # Check if item is known
                pred_item_indices.append(item_idx)
                original_pos_map[item_idx] = i
                known_item_indices.append(item_idx)

        if not pred_item_indices:
            # print(f"Warning: None of the provided item_ids for user {lookup_user_id} were known. Returning 0 scores.") # Use correct ID in log
            return [0.0] * len(item_ids)

        # Use the model's predict method
        scores_known_items = self.model.predict(user_idx, known_item_indices) # Pass known indices

        # Reconstruct the scores list in the original order
        final_scores = [0.0] * len(item_ids) # Initialize with zeros
        for valid_idx, score in zip(known_item_indices, scores_known_items):
            original_pos = original_pos_map.get(valid_idx)
            if original_pos is not None:
                final_scores[original_pos] = float(score) # Ensure float

        return final_scores

    # --- Required Methods from BaseRecommender (No Changes) ---
    def get_known_items(self) -> Set[Any]:
        return set(self.item_id_to_idx.keys()) if hasattr(self, 'item_id_to_idx') else set()

    def get_known_users(self) -> Set[Any]:
        return set(self.user_id_to_idx.keys()) if hasattr(self, 'user_id_to_idx') else set()

    # --- Save Method (No Changes) ---
    def save_model(self, file_path: str):
         if self.model is None: raise RuntimeError("Model not fitted. Cannot save.")
         # Save hyperparameters and model state_dict
         torch.save({
             'hyperparameters': {
                 'mf_dim': self.mf_dim, 'mlp_layers': self.mlp_layers,
                 'mlp_embedding_dim': self.mlp_embedding_dim, 'dropout': self.dropout,
                 'lr': self.lr, 'epochs': self.epochs, 'batch_size': self.batch_size,
                 'num_negatives': self.num_negatives, 'weight_decay': self.weight_decay,
                 'n_users': self.n_users, 'n_items': self.n_items # Save n_users/n_items
             },
             'user_id_to_idx': self.user_id_to_idx,
             'item_id_to_idx': self.item_id_to_idx,
             'model_state_dict': self.model.state_dict(),
         }, file_path)
         print(f"NCFRecommender saved to {file_path}")

    # --- load_model (FIXED map key type conversion) ---
    @classmethod
    def load_model(cls, file_path: str, device: str = 'auto'):
         """Loads the NCFRecommender state."""
         if device == 'auto':
             resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         else:
             resolved_device = torch.device(device)

         try:
             print(f"Loading NCF checkpoint from {file_path} with weights_only=False...")
             checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
         except AttributeError as e:
             print(f"Error loading checkpoint (possible file corruption or wrong format): {e}")
             raise RuntimeError(f"Could not load checkpoint file: {file_path}") from e
         except RuntimeError as e:
             print(f"RuntimeError loading checkpoint: {e}")
             raise RuntimeError(f"Could not load checkpoint file: {file_path}") from e

         params = checkpoint['hyperparameters']

         recommender = cls(
             mf_dim=params['mf_dim'], mlp_layers=params['mlp_layers'],
             mlp_embedding_dim=params['mlp_embedding_dim'], dropout=params['dropout'],
             learning_rate=params['lr'], epochs=params['epochs'], batch_size=params['batch_size'],
             num_negatives=params['num_negatives'], weight_decay=params['weight_decay'],
             device=str(resolved_device) # Pass device string
         )

         # --- FIX: Load mappings and ensure correct key types ---
         loaded_user_map = checkpoint.get('user_id_to_idx', {})
         loaded_item_map = checkpoint.get('item_id_to_idx', {})
         recommender.user_id_to_idx = {int(k): v for k, v in loaded_user_map.items()} # Ensure int keys
         recommender.item_id_to_idx = {str(k): v for k, v in loaded_item_map.items()} # Ensure str keys
         # -------------------------------------------------------
         recommender.n_users = params['n_users']
         recommender.n_items = params['n_items']
         recommender.idx_to_user_id = {v: k for k, v in recommender.user_id_to_idx.items()}
         recommender.idx_to_item_id = {v: k for k, v in recommender.item_id_to_idx.items()}


         recommender.model = NCF(
             n_users=recommender.n_users, n_items=recommender.n_items,
             mf_dim=params['mf_dim'], mlp_layers=params['mlp_layers'],
             mlp_embedding_dim=params['mlp_embedding_dim'], dropout=params['dropout']
         )
         recommender.model.load_state_dict(checkpoint['model_state_dict'])
         recommender.model = recommender.model.to(resolved_device)
         recommender.model.eval()

         print(f"NCFRecommender loaded from {file_path} to device {resolved_device}")
         return recommender