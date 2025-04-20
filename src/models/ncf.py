# src/models/ncf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Added for predict method return type
import sys
from pathlib import Path
from typing import Union

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# from src import config # Could import for default hyperparameters

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) Model.
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    to learn user-item interactions.

    Args:
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        mf_dim (int): Embedding dimension for the GMF part.
        mlp_layers (list[int]): List defining the size of hidden layers for the MLP part.
                                The first element is the input dimension (2 * mlp_embedding_dim),
                                the last element is the output dimension before the final NeuMF layer.
        mlp_embedding_dim (int): Embedding dimension for the MLP part's user/item embeddings.
                                 Often same as mf_dim, but can be different.
        dropout (float): Dropout rate for MLP layers.
    """
    def __init__(self, n_users: int, n_items: int, mf_dim: int = 16, mlp_layers: list = [32, 16, 8], mlp_embedding_dim: int = 16, dropout: float = 0.1):
        super(NCF, self).__init__()

        print("Initializing NCF Model...")
        self.n_users = n_users
        self.n_items = n_items
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers
        self.mlp_embedding_dim = mlp_embedding_dim
        self.dropout = dropout

        # --- GMF Components ---
        print(f" GMF Embedding Dim: {self.mf_dim}")
        self.mf_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.mf_dim)
        self.mf_embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.mf_dim)

        # --- MLP Components ---
        print(f" MLP Embedding Dim: {self.mlp_embedding_dim}")
        print(f" MLP Layers: {self.mlp_layers}")
        self.mlp_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.mlp_embedding_dim)
        self.mlp_embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.mlp_embedding_dim)

        # Dynamically create MLP layers
        self.mlp = nn.Sequential()
        # Input layer size = 2 * mlp_embedding_dim (concatenated user & item MLP embeddings)
        input_size = 2 * self.mlp_embedding_dim
        for i, layer_size in enumerate(self.mlp_layers):
            self.mlp.add_module(f"mlp_linear_{i}", nn.Linear(input_size, layer_size))
            self.mlp.add_module(f"mlp_relu_{i}", nn.ReLU())
            if self.dropout > 0:
                 self.mlp.add_module(f"mlp_dropout_{i}", nn.Dropout(p=self.dropout))
            input_size = layer_size # Input for next layer is output of current

        # --- NeuMF Fusion Layer ---
        # Input size is the output dim of GMF (mf_dim) + output dim of MLP (last layer size in mlp_layers)
        neumf_input_dim = self.mf_dim + self.mlp_layers[-1]
        print(f" NeuMF Input Dim: {neumf_input_dim} (MF={mf_dim} + MLP={mlp_layers[-1]})")
        self.neumf_layer = nn.Linear(neumf_input_dim, 1)

        # Initialize weights (optional but often recommended)
        self._init_weights()
        print("NCF Model Initialized.")


    def _init_weights(self):
        """ Initializes weights for embeddings and linear layers. """
        nn.init.xavier_uniform_(self.mf_embedding_user.weight)
        nn.init.xavier_uniform_(self.mf_embedding_item.weight)
        nn.init.xavier_uniform_(self.mlp_embedding_user.weight)
        nn.init.xavier_uniform_(self.mlp_embedding_item.weight)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.neumf_layer.weight)
        if self.neumf_layer.bias is not None:
             nn.init.zeros_(self.neumf_layer.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the NCF model. """
        # --- GMF Path ---
        mf_user_embedding = self.mf_embedding_user(user_indices)
        mf_item_embedding = self.mf_embedding_item(item_indices)
        gmf_output = mf_user_embedding * mf_item_embedding

        # --- MLP Path ---
        mlp_user_embedding = self.mlp_embedding_user(user_indices)
        mlp_item_embedding = self.mlp_embedding_item(item_indices)
        mlp_input = torch.cat((mlp_user_embedding, mlp_item_embedding), dim=-1)
        mlp_output = self.mlp(mlp_input)

        # --- NeuMF Fusion ---
        neumf_input = torch.cat((gmf_output, mlp_output), dim=-1)
        logits = self.neumf_layer(neumf_input)
        return logits.squeeze(-1) # Return logits


    # --- ADDED PREDICT METHOD ---
    def predict(self, user_indices: Union[torch.Tensor, np.ndarray, int],
                      item_indices: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
        """
        Predicts scores for given user(s) and item(s).
        Handles single user or batch prediction for evaluation.

        Args:
            user_indices: A single user index (int) or a tensor/array of user indices.
            item_indices: A list/array/tensor of item indices.

        Returns:
            np.ndarray: Array of predicted scores (logits).
        """
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            # Determine device from a model parameter
            current_device = next(self.parameters()).device

            # Convert inputs to tensors and move to device
            if isinstance(user_indices, int):
                # If single user, repeat it for each item
                user_tensor = torch.tensor([user_indices] * len(item_indices), dtype=torch.long).to(current_device)
            elif isinstance(user_indices, np.ndarray):
                 user_tensor = torch.from_numpy(user_indices).long().to(current_device)
            else: # Assume it's already a tensor
                 user_tensor = user_indices.long().to(current_device)

            if isinstance(item_indices, list):
                 item_tensor = torch.tensor(item_indices, dtype=torch.long).to(current_device)
            elif isinstance(item_indices, np.ndarray):
                 item_tensor = torch.from_numpy(item_indices).long().to(current_device)
            else: # Assume it's already a tensor
                 item_tensor = item_indices.long().to(current_device)

            # Ensure tensors have compatible shapes for batch processing if needed
            if user_tensor.shape != item_tensor.shape and user_tensor.ndim == 1 and item_tensor.ndim == 1:
                 # If shapes mismatch and both are 1D vectors
                 if user_tensor.shape[0] == 1 and item_tensor.shape[0] > 1: # Single user, multiple items
                     user_tensor = user_tensor.repeat(item_tensor.shape[0])
                 elif item_tensor.shape[0] == 1 and user_tensor.shape[0] > 1: # Multiple users, single item
                      item_tensor = item_tensor.repeat(user_tensor.shape[0])
                 # Add more shape adjustments if necessary, otherwise assume they match for batch mode

            logits = self.forward(user_tensor, item_tensor)
            return logits.cpu().numpy() # Return scores as numpy array


# --- Example Usage (unchanged) ---
if __name__ == '__main__':
    num_users = 100; num_items = 50; batch_size = 4
    ncf_model = NCF(n_users=num_users, n_items=num_items)
    print("\nModel Architecture:\n", ncf_model)
    dummy_user_idx = torch.randint(0, num_users, (batch_size,))
    dummy_item_idx = torch.randint(0, num_items, (batch_size,))
    print("\nDummy Input User Indices:", dummy_user_idx); print("Dummy Input Item Indices:", dummy_item_idx)
    output_logits = ncf_model(dummy_user_idx, dummy_item_idx)
    print("\nOutput Logits Shape:", output_logits.shape); print("Output Logits:", output_logits)
    output_probs = torch.sigmoid(output_logits)
    print("\nOutput Probabilities:", output_probs)