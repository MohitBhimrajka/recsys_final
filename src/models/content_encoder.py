# src/models/content_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root for imports if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# from src import config # Could import for default hyperparameters

class ContentEncoder(nn.Module):
    """
    Encodes item features (e.g., VLE proportions, length) into a dense embedding vector.
    Uses a simple Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): The number of input features for each item.
        embedding_dim (int): The desired output embedding dimension.
        hidden_dims (list[int], optional): List defining the size of hidden layers.
                                           If None, a single linear layer is used.
                                           Defaults to [64, 32].
        dropout (float): Dropout rate for hidden layers. Defaults to 0.1.
    """
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dims: list = [64, 32], dropout: float = 0.1):
        super(ContentEncoder, self).__init__()

        print("Initializing ContentEncoder Model...")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims if hidden_dims else [] # Ensure it's a list
        self.dropout = dropout

        layer_dims = [self.input_dim] + self.hidden_dims + [self.embedding_dim]
        print(f" Input Dim: {self.input_dim}")
        print(f" Hidden Dims: {self.hidden_dims}")
        print(f" Output Embedding Dim: {self.embedding_dim}")
        print(f" Layer Dimensions: {layer_dims}")

        mlp_layers = []
        for i in range(len(layer_dims) - 1):
            mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            # Apply activation and dropout to all layers except the last one
            if i < len(layer_dims) - 2:
                mlp_layers.append(nn.ReLU())
                if self.dropout > 0:
                    mlp_layers.append(nn.Dropout(p=self.dropout))

        self.mlp = nn.Sequential(*mlp_layers)

        # Initialize weights
        self._init_weights()
        print("ContentEncoder Model Initialized.")

    def _init_weights(self):
        """ Initializes weights for linear layers. """
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Content Encoder.

        Args:
            item_features (torch.Tensor): Tensor of item features (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of item embeddings (batch_size, embedding_dim).
        """
        return self.mlp(item_features)


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing ContentEncoder ---")
    # Assume item features have dimension 21 (length + 20 VLE props)
    # Load dummy item features (replace with actual loading if needed)
    dummy_item_features = pd.DataFrame({
        'module_presentation_length': np.random.randint(200, 300, size=10),
        'vle_prop_resource': np.random.rand(10),
        'vle_prop_oucontent': np.random.rand(10),
        # Add dummy columns for other VLE types to match expected input_dim
        **{f'vle_prop_type_{i}': np.random.rand(10) for i in range(18)}
    })
    print(f"Dummy Item Features shape: {dummy_item_features.shape}")

    INPUT_DIM = dummy_item_features.shape[1]
    EMBEDDING_DIM = 16 # Example output dimension

    # Initialize model
    content_model = ContentEncoder(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)
    print("\nModel Architecture:\n", content_model)

    # Create dummy input tensor
    # Convert pandas df to numpy then to tensor
    features_tensor = torch.tensor(dummy_item_features.values, dtype=torch.float32)
    print("\nInput Tensor Shape:", features_tensor.shape)

    # Perform forward pass
    output_embeddings = content_model(features_tensor)
    print("\nOutput Embeddings Shape:", output_embeddings.shape) # Should be (batch_size, embedding_dim)
    print("Output Embeddings (first 2):", output_embeddings[:2])