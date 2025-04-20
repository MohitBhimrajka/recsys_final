# src/pipelines/evaluate.py

import argparse
import pandas as pd
from pathlib import Path
import pickle
import json
import sys
import time
import torch # Needed for loading PyTorch models

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
from src.data import preprocess # For time_based_split
from src.evaluation.evaluator import RecEvaluator
# Import all model classes that might need to be loaded
from src.models.popularity import PopularityRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.matrix_factorization import ImplicitALSWrapper
from src.models.ncf import NCFRecommender
from src.models.hybrid import HybridNCFRecommender


def load_model(model_path: Path):
    """Loads a saved model artifact, trying different loading methods."""
    print(f"Loading model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = None
    # Try loading as PyTorch model first (NCF/Hybrid)
    if model_path.suffix == '.pt':
        try:
            # Determine device based on availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Try loading NCF first, then Hybrid (could be improved with metadata)
            try:
                 print("Attempting to load as NCFRecommender...")
                 model = NCFRecommender.load_model(str(model_path), device=device)
                 print("Loaded as NCFRecommender.")
            except Exception as e_ncf:
                 print(f"Failed to load as NCF: {e_ncf}. Trying Hybrid...")
                 try:
                      model = HybridNCFRecommender.load_model(str(model_path), device=device)
                      print("Loaded as HybridNCFRecommender.")
                 except Exception as e_hybrid:
                      print(f"Failed to load as Hybrid: {e_hybrid}. Cannot load as known PyTorch wrapper.")
                      raise ValueError(f"Could not load PyTorch model from {model_path}") from e_hybrid
        except Exception as e:
             print(f"Error loading PyTorch model: {e}")
             raise
    elif model_path.suffix == '.pkl':
        # Try loading using pickle for simpler models
        try:
            print("Attempting to load using pickle...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded model of type: {type(model).__name__}")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise
    else:
        raise ValueError(f"Unknown model file extension: {model_path.suffix}. Expected .pt or .pkl")

    if model is None:
         raise RuntimeError(f"Failed to load model from {model_path}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained recommendation model.")
    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the saved model artifact (.pkl or .pt).')
    parser.add_argument('--interactions-path', type=str,
                        default=str(config.PROCESSED_INTERACTIONS),
                        help='Path to the processed interactions Parquet file (for splitting).')
    parser.add_argument('--item-features-path', type=str,
                        default=str(config.PROCESSED_ITEMS),
                        help='Path to the processed item features Parquet file (needed by evaluator).')
    parser.add_argument('--metrics-output-path', type=str, default=None,
                        help='Optional path to save evaluation metrics as a JSON file.')
    parser.add_argument('--k', type=int, default=config.TOP_K,
                        help='Value of K for metrics@K.')
    parser.add_argument('--time-split-threshold', type=int, default=config.TIME_SPLIT_THRESHOLD,
                        help='Timestamp threshold for time-based split.')
    parser.add_argument('--neg-samples', type=int, default=100,
                        help='Number of negative samples for evaluation (0 for full evaluation).')

    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    print(f"--- Starting Evaluation Pipeline ---")
    print(f"Args: {vars(args)}")

    # --- Load Model ---
    model_path = Path(args.model_path)
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Load Data ---
    print(f"Loading interactions data from: {args.interactions_path}")
    interactions_path = Path(args.interactions_path)
    if not interactions_path.exists():
        print(f"Error: Interactions file not found at {args.interactions_path}")
        sys.exit(1)
    interactions_df = pd.read_parquet(interactions_path)

    print(f"Loading item features from: {args.item_features_path}")
    item_features_path = Path(args.item_features_path)
    if not item_features_path.exists():
        print(f"Error: Item features file not found at {args.item_features_path}")
        sys.exit(1)
    item_features_df = pd.read_parquet(item_features_path)

    # Ensure item features df is indexed correctly for RecEvaluator
    # (Logic assumes the parquet file has an 'item_id' like column)
    potential_id_cols = [col for col in item_features_df.columns if 'id' in col.lower()]
    if config.ITEM_COL in item_features_df.columns:
         item_id_col_in_parquet = config.ITEM_COL
    elif len(potential_id_cols) == 1:
         item_id_col_in_parquet = potential_id_cols[0]
         print(f"Warning: Assuming item ID column in items parquet is '{item_id_col_in_parquet}'.")
    else:
         # Check if index name is already correct
         if item_features_df.index.name == config.ITEM_COL:
              item_id_col_in_parquet = None # Already indexed
         else:
              raise ValueError(f"Could not reliably determine item ID column/index in {args.item_features_path}. Found columns: {item_features_df.columns}, Index: {item_features_df.index.name}")

    if item_id_col_in_parquet:
         item_features_df = item_features_df.set_index(item_id_col_in_parquet)
         item_features_df.index.name = config.ITEM_COL # Standardize name

    # --- Time-Based Split ---
    print(f"Performing time-based split using threshold: {args.time_split_threshold}")
    train_df, test_df = preprocess.time_based_split(
        interactions_df=interactions_df,
        user_col=config.USER_COL,
        item_col=config.ITEM_COL,
        time_col=config.TIME_COL, # Make sure TIME_COL is defined in config.py
        time_unit_threshold=args.time_split_threshold
    )

    if test_df.empty:
        print("Warning: Test set is empty after time split. Evaluation results will be zero.")
        metrics = {f'Precision@{args.k}': 0.0, f'Recall@{args.k}': 0.0, f'NDCG@{args.k}': 0.0, 'n_users_evaluated': 0}
    else:
        # --- Evaluate ---
        print("Initializing evaluator...")
        evaluator = RecEvaluator(
            train_df=train_df,
            test_df=test_df,
            item_features_df=item_features_df, # Pass indexed df
            user_col=config.USER_COL,
            item_col=config.ITEM_COL,
            k=args.k
        )

        print(f"Evaluating model {type(model).__name__} with K={args.k} and neg_samples={args.neg_samples if args.neg_samples > 0 else 'Full'}...")
        n_neg = args.neg_samples if args.neg_samples > 0 else None
        metrics = evaluator.evaluate_model(model, n_neg_samples=n_neg)

    # --- Output Results ---
    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=4))

    if args.metrics_output_path:
        output_path = Path(args.metrics_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving metrics to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    end_time = time.time()
    print(f"\n--- Evaluation Pipeline Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Add dummy config values if running directly
    if not hasattr(config, 'USER_COL'): config.USER_COL = 'id_student'
    if not hasattr(config, 'ITEM_COL'): config.ITEM_COL = 'presentation_id'
    if not hasattr(config, 'SCORE_COL'): config.SCORE_COL = 'implicit_feedback'
    if not hasattr(config, 'TIME_COL'): config.TIME_COL = 'last_interaction_date' # Example, ensure this exists
    if not hasattr(config, 'PROCESSED_INTERACTIONS'): config.PROCESSED_INTERACTIONS = Path('data/processed/interactions_final.parquet')
    if not hasattr(config, 'PROCESSED_ITEMS'): config.PROCESSED_ITEMS = Path('data/processed/items_final.parquet')
    if not hasattr(config, 'TOP_K'): config.TOP_K = 10
    if not hasattr(config, 'TIME_SPLIT_THRESHOLD'): config.TIME_SPLIT_THRESHOLD = 250 # Make sure this matches preprocessing notebook

    main()