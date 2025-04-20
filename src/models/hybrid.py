# src/models/hybrid.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the ContentEncoder we just created
from src.models.content_encoder import ContentEncoder

class HybridNCF(nn.Module):
    """
    A Hybrid Recommendation Model combining Collaborative Filtering (CF) embeddings
    and Content-Based (CB) item embeddings.

    It uses:
    - User and Item embeddings for the CF path (similar to MF/GMF).
    - A ContentEncoder (MLP) to generate embeddings from item features.
    - Concatenation of User CF, Item CF, and Item CB embeddings.
    - A final MLP prediction head.

    Args:
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        item_feature_dim (int): Dimensionality of the input item content features.
        cf_embedding_dim (int): Embedding dimension for user and item CF embeddings.
        content_embedding_dim (int): Output dimension of the ContentEncoder.
        content_encoder_hidden_dims (list[int]): Hidden layer sizes for the ContentEncoder MLP.
        final_mlp_layers (list[int]): Hidden layer sizes for the final prediction MLP.
                                      The input size will be cf_embedding_dim * 2 + content_embedding_dim.
        dropout (float): Dropout rate for MLPs.
    """
    def __init__(self, n_users: int, n_items: int, item_feature_dim: int,
                 cf_embedding_dim: int = 16,
                 content_embedding_dim: int = 16,
                 content_encoder_hidden_dims: list = [32, 16], # Example: Input -> 32 -> 16 -> content_embedding_dim
                 final_mlp_layers: list = [32, 16, 8], # Example: Input -> 32 -> 16 -> 8 -> 1
                 dropout: float = 0.1):
        super(HybridNCF, self).__init__()

        print("Initializing HybridNCF Model...")
        self.n_users = n_users
        self.n_items = n_items
        self.item_feature_dim = item_feature_dim
        self.cf_embedding_dim = cf_embedding_dim
        self.content_embedding_dim = content_embedding_dim

        # --- CF Embeddings ---
        print(f" CF Embedding Dim: {cf_embedding_dim}")
        self.cf_user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=cf_embedding_dim)
        self.cf_item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=cf_embedding_dim)

        # --- Content Encoder ---
        print(f" Initializing Content Encoder (Input: {item_feature_dim}, Output: {content_embedding_dim})")
        self.content_encoder = ContentEncoder(
            input_dim=item_feature_dim,
            embedding_dim=content_embedding_dim,
            hidden_dims=content_encoder_hidden_dims, # Pass hidden dims
            dropout=dropout
        )

        # --- Final Prediction MLP ---
        # Input size = User CF Emb + Item CF Emb + Item Content Emb
        final_mlp_input_dim = cf_embedding_dim + cf_embedding_dim + content_embedding_dim
        print(f" Final MLP Input Dim: {final_mlp_input_dim}")
        print(f" Final MLP Layers: {final_mlp_layers}")

        self.final_mlp = nn.Sequential()
        input_size = final_mlp_input_dim
        for i, layer_size in enumerate(final_mlp_layers):
            self.final_mlp.add_module(f"final_mlp_linear_{i}", nn.Linear(input_size, layer_size))
            self.final_mlp.add_module(f"final_mlp_relu_{i}", nn.ReLU())
            if dropout > 0:
                 self.final_mlp.add_module(f"final_mlp_dropout_{i}", nn.Dropout(p=dropout))
            input_size = layer_size

        # Final output layer (1 logit)
        self.final_layer = nn.Linear(input_size, 1)

        # Initialize weights
        self._init_weights()
        print("HybridNCF Model Initialized.")

    def _init_weights(self):
        """ Initializes weights. """
        nn.init.xavier_uniform_(self.cf_user_embedding.weight)
        nn.init.xavier_uniform_(self.cf_item_embedding.weight)
        # ContentEncoder initializes its own weights

        for layer in self.final_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.final_layer.weight)
        if self.final_layer.bias is not None: nn.init.zeros_(self.final_layer.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HybridNCF model.

        Args:
            user_indices (torch.Tensor): Tensor of user indices (batch_size,).
            item_indices (torch.Tensor): Tensor of item indices (batch_size,).
            item_features (torch.Tensor): Tensor of item content features (batch_size, item_feature_dim).

        Returns:
            torch.Tensor: Output tensor (logits before sigmoid) (batch_size,).
        """
        # CF Embeddings
        user_cf_emb = self.cf_user_embedding(user_indices)
        item_cf_emb = self.cf_item_embedding(item_indices)

        # Content Embeddings
        item_content_emb = self.content_encoder(item_features)

        # Concatenate all embeddings
        combined_vector = torch.cat([user_cf_emb, item_cf_emb, item_content_emb], dim=1)

        # Pass through final MLP
        mlp_output = self.final_mlp(combined_vector)
        logits = self.final_layer(mlp_output)

        return logits.squeeze(-1)


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing HybridNCF ---")
    # Dummy parameters
    N_USERS = 100
    N_ITEMS = 50
    ITEM_FEATURE_DIM = 21 # Should match ContentEncoder input
    BATCH_SIZE = 4

    # Initialize model
    hybrid_model = HybridNCF(
        n_users=N_USERS,
        n_items=N_ITEMS,
        item_feature_dim=ITEM_FEATURE_DIM,
        cf_embedding_dim=16,
        content_embedding_dim=8, # Can be different from CF dim
        content_encoder_hidden_dims=[32, 16], # Input(21)->32->16->Output(8)
        final_mlp_layers=[20, 10], # Input(16+16+8=40)->20->10->Output(1)
        dropout=0.1
    )
    print("\nModel Architecture:\n", hybrid_model)

    # Create dummy input batch
    dummy_user_idx = torch.randint(0, N_USERS, (BATCH_SIZE,))
    dummy_item_idx = torch.randint(0, N_ITEMS, (BATCH_SIZE,))
    dummy_item_features = torch.randn(BATCH_SIZE, ITEM_FEATURE_DIM) # Random features

    print("\nDummy Input User Indices:", dummy_user_idx)
    print("Dummy Input Item Indices:", dummy_item_idx)
    print("Dummy Input Item Features Shape:", dummy_item_features.shape)

    # Perform forward pass
    output_logits = hybrid_model(dummy_user_idx, dummy_item_idx, dummy_item_features)
    print("\nOutput Logits Shape:", output_logits.shape) # Should be (batch_size,)
    print("Output Logits:", output_logits)