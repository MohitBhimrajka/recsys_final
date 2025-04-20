# src/pipelines/train.py

import argparse
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import sys
import time
import torch # Needed for NCF/Hybrid load/save

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
from src.models.popularity import PopularityRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.matrix_factorization import ImplicitALSWrapper
from src.models.ncf import NCFRecommender
from src.models.hybrid import HybridNCFRecommender

def parse_args():
    parser = argparse.ArgumentParser(description="Train a recommendation model.")
    parser.add_argument('--model-name', required=True,
                        choices=['Popularity', 'ItemCF', 'ALS', 'NCF', 'Hybrid'],
                        help='Name of the model to train.')
    parser.add_argument('--interactions-path', type=str,
                        default=str(config.PROCESSED_INTERACTIONS),
                        help='Path to the processed interactions Parquet file.')
    parser.add_argument('--item-features-path', type=str,
                        default=str(config.PROCESSED_ITEMS),
                        help='Path to the processed item features Parquet file (required for Hybrid).')
    # parser.add_argument('--user-features-path', type=str, # Add if needed by future models
    #                     default=str(config.PROCESSED_USERS),
    #                     help='Path to the processed user features Parquet file.')
    parser.add_argument('--output-dir', type=str, default=str(config.SAVED_MODELS_DIR),
                        help='Directory to save the trained model artifact.')

    # --- Model Hyperparameters (add more as needed) ---
    # ALS
    parser.add_argument('--factors', type=int, default=50, help='Number of latent factors (ALS, NCF MF).')
    parser.add_argument('--regularization', type=float, default=0.05, help='Regularization factor (ALS).')
    parser.add_argument('--iterations', type=int, default=25, help='Number of iterations (ALS).')
    # NCF / Hybrid
    parser.add_argument('--mf-dim', type=int, default=16, help='Embedding dimension for NCF MF part.')
    parser.add_argument('--mlp-embed-dim', type=int, default=16, help='Embedding dimension for NCF/Hybrid MLP part.')
    parser.add_argument('--mlp-layers', nargs='+', type=int, default=[32, 16, 8], help='MLP layer sizes (NCF/Hybrid final MLP).')
    parser.add_argument('--content-embed-dim', type=int, default=16, help='Output dimension of content encoder (Hybrid).')
    parser.add_argument('--content-encoder-layers', nargs='+', type=int, default=[32, 16], help='Content encoder hidden layers (Hybrid).')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (NCF/Hybrid).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (NCF/Hybrid).')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (NCF/Hybrid).')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (NCF/Hybrid).')
    parser.add_argument('--num-negatives', type=int, default=4, help='Number of negative samples per positive (NCF/Hybrid).')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (Adam optimizer, NCF/Hybrid).')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device for PyTorch models.')

    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED, help='Random seed.')

    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    print(f"--- Starting Training Pipeline for Model: {args.model_name} ---")
    print(f"Args: {vars(args)}")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    if args.device != 'cpu': # PyTorch specific seeding
         torch.manual_seed(args.seed)
         if torch.cuda.is_available():
             torch.cuda.manual_seed_all(args.seed)

    # --- Load Data ---
    print(f"Loading interactions data from: {args.interactions_path}")
    interactions_path = Path(args.interactions_path)
    if not interactions_path.exists():
        print(f"Error: Interactions file not found at {args.interactions_path}")
        sys.exit(1)
    interactions_df = pd.read_parquet(interactions_path)

    item_features_df = None
    if args.model_name == 'Hybrid':
        print(f"Loading item features from: {args.item_features_path}")
        item_features_path = Path(args.item_features_path)
        if not item_features_path.exists():
            print(f"Error: Item features file not found at {args.item_features_path} (required for Hybrid model)")
            sys.exit(1)
        # Load and set index correctly for Hybrid model wrapper
        item_features_df = pd.read_parquet(item_features_path)
        if config.PROCESSED_ITEMS.stem.split('_')[0] in item_features_df.columns: # e.g. 'items_final.parquet' -> 'items' base name
            item_id_col_in_parquet = config.PROCESSED_ITEMS.stem.split('_')[0] # Get base name
        else:
             # Fallback or raise error if naming convention fails
             potential_id_cols = [col for col in item_features_df.columns if 'id' in col.lower()]
             if len(potential_id_cols) == 1:
                 item_id_col_in_parquet = potential_id_cols[0]
                 print(f"Warning: Assuming item ID column in items parquet is '{item_id_col_in_parquet}'. Set index explicitly if needed.")
             else:
                 raise ValueError(f"Could not reliably determine the item ID column in {args.item_features_path}. Found: {item_features_df.columns}")

        item_features_df = item_features_df.set_index(item_id_col_in_parquet) # Set index based on discovered/assumed name
        item_features_df.index.name = config.ITEM_COL # Standardize index name to match expected col name

    # --- Instantiate Model ---
    print("Instantiating model...")
    model = None
    user_col = config.USER_COL
    item_col = config.ITEM_COL
    score_col = config.SCORE_COL # implicit_feedback

    if args.model_name == 'Popularity':
        model = PopularityRecommender(user_col=user_col, item_col=item_col, score_col=score_col)
    elif args.model_name == 'ItemCF':
        model = ItemCFRecommender(user_col=user_col, item_col=item_col, score_col=score_col)
    elif args.model_name == 'ALS':
        model = ImplicitALSWrapper(user_col=user_col, item_col=item_col, score_col=score_col,
                                   factors=args.factors, regularization=args.regularization,
                                   iterations=args.iterations, random_state=args.seed)
    elif args.model_name == 'NCF':
        model = NCFRecommender(user_col=user_col, item_col=item_col, score_col=score_col,
                               mf_dim=args.mf_dim, mlp_layers=args.mlp_layers,
                               mlp_embedding_dim=args.mlp_embed_dim, dropout=args.dropout,
                               learning_rate=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                               num_negatives=args.num_negatives, weight_decay=args.weight_decay,
                               device=args.device)
    elif args.model_name == 'Hybrid':
        if item_features_df is None:
             print("Error: Item features are required for Hybrid model but were not loaded.")
             sys.exit(1)
        model = HybridNCFRecommender(user_col=user_col, item_col=item_col, score_col=score_col,
                                     cf_embedding_dim=args.mlp_embed_dim, # Using same embed dim for cf/mlp part
                                     content_embedding_dim=args.content_embed_dim,
                                     content_encoder_hidden_dims=args.content_encoder_layers,
                                     final_mlp_layers=args.mlp_layers, dropout=args.dropout,
                                     learning_rate=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                                     num_negatives=args.num_negatives, weight_decay=args.weight_decay,
                                     device=args.device)
    else:
        print(f"Error: Unknown model name '{args.model_name}'")
        sys.exit(1)

    # --- Train Model ---
    print(f"Starting training for {args.model_name}...")
    try:
        if args.model_name == 'Hybrid':
            model.fit(interactions_df, item_features_df)
        else:
            model.fit(interactions_df)
        print("Training complete.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise # Re-raise the exception for debugging

    # --- Save Model ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct filename with key hyperparameters
    # Example: NCF_epochs10_lr0.001.pt or ALS_factors50.pkl
    filename_parts = [args.model_name]
    if args.model_name == 'ALS': filename_parts.append(f"factors{args.factors}")
    elif args.model_name in ['NCF', 'Hybrid']:
        filename_parts.extend([f"epochs{args.epochs}", f"lr{args.lr}"])
        # Add more hyperparams if needed for uniqueness

    save_path = None
    if args.model_name in ['NCF', 'Hybrid']:
        filename = "_".join(filename_parts) + ".pt"
        save_path = output_dir / filename
        print(f"Saving model to: {save_path}")
        model.save_model(str(save_path)) # NCF/Hybrid wrappers have save_model
    else:
        # Save simple models using pickle
        filename = "_".join(filename_parts) + ".pkl"
        save_path = output_dir / filename
        print(f"Saving model to: {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    end_time = time.time()
    print(f"--- Training Pipeline Finished ---")
    print(f"Model saved to: {save_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Add dummy config values if running directly without full setup
    # (Better to run after full setup)
    if not hasattr(config, 'USER_COL'): config.USER_COL = 'id_student'
    if not hasattr(config, 'ITEM_COL'): config.ITEM_COL = 'presentation_id'
    if not hasattr(config, 'SCORE_COL'): config.SCORE_COL = 'implicit_feedback'
    if not hasattr(config, 'PROCESSED_INTERACTIONS'): config.PROCESSED_INTERACTIONS = Path('data/processed/interactions_final.parquet')
    if not hasattr(config, 'PROCESSED_ITEMS'): config.PROCESSED_ITEMS = Path('data/processed/items_final.parquet')
    if not hasattr(config, 'SAVED_MODELS_DIR'): config.SAVED_MODELS_DIR = Path('saved_models')
    if not hasattr(config, 'RANDOM_SEED'): config.RANDOM_SEED = 42

    main()