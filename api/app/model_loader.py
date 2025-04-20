# api/app/model_loader.py
import pickle
import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np # Import numpy

# --- Adjust path to import from parent directory ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config # Import from main src directory
from src.data import preprocess # Import preprocess
from src.models.item_cf import ItemCFRecommender # Import the specific model class

# --- Globals ---
loaded_model: ItemCFRecommender = None
all_items_set: set = set()
train_user_item_map: dict = defaultdict(set)
items_df_details: pd.DataFrame = None

# --- Configuration ---
MODEL_FILENAME = "ItemCF.pkl"
MODEL_PATH = config.SAVED_MODELS_DIR / MODEL_FILENAME
INTERACTIONS_PATH = config.PROCESSED_INTERACTIONS
ITEMS_PATH = config.PROCESSED_ITEMS

def load_model_and_data():
    """Loads the ItemCF model, item data, and training interactions map."""
    global loaded_model, all_items_set, train_user_item_map, items_df_details
    print("--- Loading model and data for API ---")

    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        print(f"Loading model from {MODEL_PATH}...")
        with open(MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"Model {type(loaded_model).__name__} loaded successfully.")
        if not isinstance(loaded_model, ItemCFRecommender):
             print(f"Warning: Loaded model is type {type(loaded_model)} not ItemCFRecommender")

        # --- FIX: Convert user mapping keys to standard int ---
        if hasattr(loaded_model, 'user_id_to_idx') and loaded_model.user_id_to_idx:
            print("Converting loaded model user ID mapping keys to standard int...")
            loaded_model.user_id_to_idx = {int(k): v for k, v in loaded_model.user_id_to_idx.items()}
            # Also update the inverse mapping if it exists
            if hasattr(loaded_model, 'idx_to_user_id'):
                 loaded_model.idx_to_user_id = {v: int(k) for k, v in loaded_model.user_id_to_idx.items()} # Rebuild from corrected map
            print("User ID mapping conversion complete.")
        # -------------------------------------------------------

    except Exception as e:
        print(f"Error loading model pickle file: {e}")
        raise

    # 2. Load Items Data (for candidate generation and details)
    if not ITEMS_PATH.exists():
        print(f"Error: Items file not found at {ITEMS_PATH}")
        raise FileNotFoundError(f"Items file not found: {ITEMS_PATH}")
    try:
        print(f"Loading item details from {ITEMS_PATH}...")
        items_df = pd.read_parquet(ITEMS_PATH)
        if config.ITEM_COL not in items_df.columns:
             raise ValueError(f"'{config.ITEM_COL}' column not found in items file.")

        all_items_set = set(items_df[config.ITEM_COL].unique())
        items_df[['module_id', 'presentation_code']] = items_df[config.ITEM_COL].str.split('_', expand=True)
        items_df_details = items_df.set_index(config.ITEM_COL)

        print(f"Loaded {len(all_items_set)} unique items.")
        print(f"Item details DataFrame shape: {items_df_details.shape}")

    except Exception as e:
        print(f"Error loading or processing items file: {e}")
        raise

    # 3. Load Training Interactions Map (for filtering seen items)
    if not INTERACTIONS_PATH.exists():
        print(f"Error: Interactions file not found at {INTERACTIONS_PATH}")
        raise FileNotFoundError(f"Interactions file not found: {INTERACTIONS_PATH}")
    try:
        print(f"Loading interactions from {INTERACTIONS_PATH} to build train map...")
        interactions_df = pd.read_parquet(INTERACTIONS_PATH)
        train_df, _ = preprocess.time_based_split(
            interactions_df,
            user_col=config.USER_COL,
            item_col=config.ITEM_COL,
            time_col=config.TIME_COL,
            time_unit_threshold=config.TIME_SPLIT_THRESHOLD
        )
        # --- Ensure keys in train_map are standard int ---
        temp_map = train_df.groupby(config.USER_COL)[config.ITEM_COL].agg(set).to_dict()
        train_user_item_map.update({int(k): v for k, v in temp_map.items()})
        # ------------------------------------------------
        print(f"Built training interaction map for {len(train_user_item_map)} users.")

    except Exception as e:
        print(f"Error loading or processing interactions file: {e}")
        raise

    print("--- Model and data loading complete ---")

# --- Accessor functions remain the same ---
def get_model() -> ItemCFRecommender:
    if loaded_model is None:
        raise RuntimeError("Model is not loaded. Call load_model_and_data() first.")
    return loaded_model

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