# api/app/model_loader.py
import pickle
import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import torch # Needed for loading PyTorch models
from typing import Dict, Any, Set
import time

# --- Adjust path to import from parent directory ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config # Import from main src directory
from src.data import preprocess # Import preprocess (needed for train map)

# Import all potential model classes
from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.matrix_factorization import ImplicitALSWrapper
from src.models.ncf import NCFRecommender
from src.models.hybrid import HybridNCFRecommender

# --- Globals ---
# Store loaded models by their display name
loaded_models: Dict[str, BaseRecommender] = {}
all_items_set: Set[str] = set()
train_user_item_map: Dict[int, Set[str]] = defaultdict(set) # Ensure keys are int
items_df_details: pd.DataFrame = None

# --- Configuration ---
# Define the models to load based on your saved files
MODELS_TO_LOAD = [
    {'name': 'Popularity', 'filename': 'Popularity.pkl'},
    {'name': 'ItemCF', 'filename': 'ItemCF.pkl'},
    {'name': 'ALS (f=100)', 'filename': 'ALS_factors100.pkl'}, # Use descriptive name
    {'name': 'NCF (e=15)', 'filename': 'NCF_epochs15_lr0.001.pt'},
    {'name': 'Hybrid (e=15)', 'filename': 'Hybrid_epochs15_lr0.001.pt'},
]

INTERACTIONS_PATH = config.PROCESSED_INTERACTIONS
ITEMS_PATH = config.PROCESSED_ITEMS

def _fix_numpy_int_keys(mapping: Dict) -> Dict:
    """
    Converts potential numpy integer keys in mappings to standard Python int.
    Leaves other key types (like strings for item IDs) unchanged.
    """
    if not mapping:
        return {}

    converted_mapping = {}
    needs_conversion = False
    for k in mapping.keys():
        # Check specifically for numpy integer types
        if isinstance(k, np.integer):
            needs_conversion = True
            break
        # If we hit a non-int key early, assume it's mixed or non-integer keys primarily
        elif not isinstance(k, int):
             break # Assume keys are correct type (e.g., strings for items)

    if not needs_conversion:
        # print(" Mapping keys do not appear to be numpy integers, no conversion needed.")
        return mapping

    print(" Converting numpy integer keys in mapping to standard int...")
    for k, v in mapping.items():
        if isinstance(k, np.integer):
            converted_mapping[int(k)] = v
        else:
            converted_mapping[k] = v # Keep other types (like strings) as is
    return converted_mapping

# --- _load_single_model (Keep As Is from Previous Corrected Version) ---
def _load_single_model(model_path: Path) -> BaseRecommender:
    """Loads a single model artifact, handling .pkl and .pt."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = None
    if model_path.suffix == '.pt':
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f" Attempting to load PyTorch model '{model_path.name}' to device '{device}'...")
            # Determine if it's NCF or Hybrid based on filename pattern or try/except
            if 'NCF' in model_path.name:
                 model = NCFRecommender.load_model(str(model_path), device=device)
                 print(" Loaded as NCFRecommender.")
            elif 'Hybrid' in model_path.name:
                 model = HybridNCFRecommender.load_model(str(model_path), device=device)
                 print(" Loaded as HybridNCFRecommender.")
            else:
                 # Fallback: Try loading as NCF, then Hybrid if naming unknown
                 try:
                     print(" Unknown PyTorch model type from filename, trying NCF...")
                     model = NCFRecommender.load_model(str(model_path), device=device)
                 except:
                     print(" Failed loading as NCF, trying Hybrid...")
                     model = HybridNCFRecommender.load_model(str(model_path), device=device)
            print(f" Successfully loaded PyTorch model: {type(model).__name__}")
        except Exception as e:
            print(f"Error loading PyTorch model from {model_path}: {e}")
            raise
    elif model_path.suffix == '.pkl':
        try:
            print(f" Attempting to load pickle model '{model_path.name}'...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f" Successfully loaded pickle model: {type(model).__name__}")

            # Apply key type fix only where keys are expected to be integers (user_id_to_idx)
            if hasattr(model, 'user_id_to_idx'):
                 print(" Applying key type fix for user_id_to_idx...")
                 model.user_id_to_idx = _fix_numpy_int_keys(model.user_id_to_idx)
                 if hasattr(model, 'idx_to_user_id'):
                     model.idx_to_user_id = {v: k for k, v in model.user_id_to_idx.items()}

            # Item IDs are strings, so no conversion needed for item_id_to_idx
            # if hasattr(model, 'item_id_to_idx'):
            #    print(" Applying key type fix for item_id_to_idx...") # No longer needed
            #    model.item_id_to_idx = _fix_numpy_int_keys(model.item_id_to_idx) # Don't call fix here
            #    if hasattr(model, 'idx_to_item_id'):
            #         model.idx_to_item_id = {v: k for k, v in model.item_id_to_idx.items()}

        except Exception as e:
            print(f"Error loading pickle file from {model_path}: {e}")
            raise
    else:
        raise ValueError(f"Unsupported model file extension: {model_path.suffix}")

    if model is None:
        raise RuntimeError(f"Model loading failed for {model_path}")

    return model


# --- load_model_and_data (Keep As Is from Previous Corrected Version) ---
def load_model_and_data():
    """Loads all specified models, item data, and training interactions map."""
    global loaded_models, all_items_set, train_user_item_map, items_df_details
    print("--- Loading models and data for API ---")
    load_start_time = time.time()
    models_loaded_count = 0
    loaded_models.clear() # Clear previously loaded models if reloading

    # 1. Load Models
    for model_config in MODELS_TO_LOAD:
        model_name = model_config['name']
        filename = model_config['filename']
        model_path = config.SAVED_MODELS_DIR / filename
        try:
            print(f"\nLoading model '{model_name}' from {filename}...")
            start_time = time.time()
            model_instance = _load_single_model(model_path)
            loaded_models[model_name] = model_instance
            end_time = time.time()
            print(f"Successfully loaded '{model_name}' in {end_time - start_time:.2f} seconds.")
            models_loaded_count += 1
        except Exception as e:
            print(f"ERROR: Failed to load model '{model_name}' from {filename}. Skipping. Error: {e}")

    if not loaded_models:
        raise RuntimeError("CRITICAL: No models could be loaded. API cannot function.")
    print(f"\nSuccessfully loaded {models_loaded_count} models: {list(loaded_models.keys())}")

    # 2. Load Items Data (same as before)
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Items file not found: {ITEMS_PATH}")
    try:
        print(f"\nLoading item details from {ITEMS_PATH}...")
        items_df = pd.read_parquet(ITEMS_PATH)
        if config.ITEM_COL not in items_df.columns:
             raise ValueError(f"'{config.ITEM_COL}' column not found in items file.")
        all_items_set = set(items_df[config.ITEM_COL].unique())
        items_df[['module_id', 'presentation_code']] = items_df[config.ITEM_COL].str.split('_', expand=True)
        items_df_details = items_df.set_index(config.ITEM_COL)
        print(f"Loaded {len(all_items_set)} unique items.")
        print(f"Item details DataFrame shape: {items_df_details.shape}")
    except Exception as e:
        print(f"Error loading or processing items file: {e}"); raise

    # 3. Load Training Interactions Map (same as before, ensures int keys)
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Interactions file not found: {INTERACTIONS_PATH}")
    try:
        print(f"\nLoading interactions from {INTERACTIONS_PATH} to build train map...")
        interactions_df = pd.read_parquet(INTERACTIONS_PATH)
        train_df, _ = preprocess.time_based_split(
            interactions_df,
            user_col=config.USER_COL,
            item_col=config.ITEM_COL,
            time_col=config.TIME_COL,
            time_unit_threshold=config.TIME_SPLIT_THRESHOLD
        )
        temp_map = train_df.groupby(config.USER_COL)[config.ITEM_COL].agg(set).to_dict()
        train_user_item_map.clear()
        train_user_item_map.update({int(k): v for k, v in temp_map.items()})
        print(f"Built training interaction map for {len(train_user_item_map)} users.")
    except Exception as e:
        print(f"Error loading or processing interactions file: {e}"); raise

    load_end_time = time.time()
    print(f"\n--- Model and data loading complete ({load_end_time - load_start_time:.2f} sec) ---")


# --- Accessor functions (Keep As Is) ---
def get_models() -> Dict[str, BaseRecommender]:
    if not loaded_models:
        raise RuntimeError("Models are not loaded. Call load_model_and_data() first.")
    return loaded_models

def get_all_items() -> set:
    if not all_items_set:
        raise RuntimeError("Items are not loaded. Call load_model_and_data() first.")
    return all_items_set

def get_train_map() -> dict:
    if not train_user_item_map:
         print("Warning: Training interaction map is empty.")
    return train_user_item_map

def get_item_details() -> pd.DataFrame:
     if items_df_details is None:
         raise RuntimeError("Item details not loaded. Call load_model_and_data() first.")
     return items_df_details