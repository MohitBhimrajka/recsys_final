# api/app/services.py
import pandas as pd
import numpy as np
from typing import List, Set, Optional, Dict, Tuple
from fastapi import Depends, HTTPException
import random
import time
from collections import defaultdict

# --- Adjust path to import from parent directory ---
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
from src.models.base import BaseRecommender
from api.app import schemas, model_loader

# --- Configuration for Ensemble ---
# Define weights based on your evaluation (e.g., NDCG@10)
MODEL_WEIGHTS = {
    'Popularity': 0.05, # Lowest weight
    'ItemCF': 0.40,     # Highest weight based on evaluation
    'ALS (f=100)': 0.10,# Low weight
    'NCF (e=15)': 0.25, # Second highest
    'Hybrid (e=15)': 0.20 # Third highest (adjust based on comparable evaluation)
}
print(f"Using Ensemble Weights: {MODEL_WEIGHTS}")
# Normalize weights to ensure they sum to 1
total_weight = sum(MODEL_WEIGHTS.values())
if total_weight > 0:
    MODEL_WEIGHTS = {name: w / total_weight for name, w in MODEL_WEIGHTS.items()}
else:
     print("Warning: All model weights are zero. Using equal weighting.")
     num_models = len(MODEL_WEIGHTS)
     MODEL_WEIGHTS = {name: 1.0 / num_models for name in MODEL_WEIGHTS} if num_models > 0 else {}
print(f"Normalized Ensemble Weights: {MODEL_WEIGHTS}")


# --- Helper Functions ---
# These are internal helpers called by the main service functions
def _get_candidate_items_helper(user_id: int, all_items: Set[str], train_map: dict) -> Set[str]:
    """Helper to get candidate items (all - seen in train)."""
    items_seen_by_user = train_map.get(user_id, set())
    candidate_items = all_items - items_seen_by_user
    return candidate_items

def _format_recommendations(top_k_items: List[Tuple[str, float]],
                           item_details: pd.DataFrame # Pass this explicitly now
                           ) -> List[schemas.RecommendationItem]:
    """Helper to format item list into RecommendationItem schema."""
    results = []
    for item_id, score in top_k_items:
        try:
            # Use .get(col, default) for safety if details are missing
            details = item_details.loc[item_id] if item_id in item_details.index else pd.Series()
            results.append(schemas.RecommendationItem(
                presentation_id=item_id,
                score=float(score),
                module_id=details.get('module_id', 'N/A'),
                presentation_code=details.get('presentation_code', 'N/A'),
                module_presentation_length=details.get('module_presentation_length') # Will be None if missing
            ))
        except Exception as detail_error:
            print(f"Warning: Error formatting details for item {item_id}: {detail_error}")
            # Append with placeholder details if lookup fails
            results.append(schemas.RecommendationItem(
                presentation_id=item_id,
                score=float(score),
                module_id='Error',
                presentation_code='Error',
                module_presentation_length=None
            ))
    return results


# --- Service to get recommendations from ALL models ---
# Dependencies are injected by FastAPI when this function is called by the router
def get_all_model_recommendations(
    user_id: int,
    k: int,
    models: Dict[str, BaseRecommender] = Depends(model_loader.get_models),
    all_items: Set[str] = Depends(model_loader.get_all_items),
    train_map: dict = Depends(model_loader.get_train_map),
    item_details: pd.DataFrame = Depends(model_loader.get_item_details),
) -> Dict[str, List[schemas.RecommendationItem]]:
    """Generates top-k recommendations from each loaded model for a given user."""
    print(f"Generating recommendations from all models for user: {user_id}, k={k}")
    start_time = time.time()
    all_results: Dict[str, List[schemas.RecommendationItem]] = {}

    # ** Call helper function explicitly, passing dependencies **
    candidate_items = _get_candidate_items_helper(user_id, all_items, train_map)

    if not candidate_items:
        print(f"User {user_id} has no candidate items. Returning empty results for all models.")
        return {name: [] for name in models.keys()}

    candidate_items_list = list(candidate_items)

    for model_name, model in models.items():
        known_users = model.get_known_users()
        if user_id not in known_users:
            all_results[model_name] = []
            continue

        try:
            scores = model.predict(user_id, candidate_items_list)
            if len(scores) != len(candidate_items_list):
                print(f"  Error: Score prediction length mismatch for model {model_name}, user {user_id}. Skipping.")
                all_results[model_name] = []
                continue

            item_score_pairs = list(zip(candidate_items_list, scores))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_k_pairs = item_score_pairs[:k]

            # ** Pass item_details explicitly to helper **
            formatted_recs = _format_recommendations(top_k_pairs, item_details)
            all_results[model_name] = formatted_recs

        except Exception as e:
            print(f"  Error during prediction for model {model_name}, user {user_id}: {e}")
            all_results[model_name] = []

    end_time = time.time()
    print(f"Generated all model recs in {end_time - start_time:.3f} seconds.")
    return all_results

# --- Service for Ensemble Recommendations ---
# Dependencies are injected by FastAPI when this function is called by the router
def get_ensemble_recommendations(
    user_id: int,
    k: int,
    models: Dict[str, BaseRecommender] = Depends(model_loader.get_models),
    all_items: Set[str] = Depends(model_loader.get_all_items),
    train_map: dict = Depends(model_loader.get_train_map),
    item_details: pd.DataFrame = Depends(model_loader.get_item_details),
) -> List[schemas.RecommendationItem]:
    """
    Generates top-k recommendations using a weighted average ensemble
    of scores from all loaded models.
    """
    print(f"Generating ENSEMBLE recommendations for user: {user_id}, k={k}")
    start_time = time.time()

    # ** Call helper function explicitly, passing dependencies **
    candidate_items = _get_candidate_items_helper(user_id, all_items, train_map)

    if not candidate_items:
        print(f"User {user_id} has no candidate items. Returning empty ensemble.")
        return []

    candidate_items_list = list(candidate_items)
    item_scores_all_models = defaultdict(dict)
    valid_models_for_user = []

    # 1. Get Raw Scores from Each Model
    for model_name, model in models.items():
        if model_name not in MODEL_WEIGHTS or MODEL_WEIGHTS[model_name] == 0: continue
        if user_id not in model.get_known_users(): continue
        valid_models_for_user.append(model_name)
        try:
            scores = model.predict(user_id, candidate_items_list)
            if len(scores) != len(candidate_items_list): continue
            for item_id, score in zip(candidate_items_list, scores): item_scores_all_models[item_id][model_name] = score
        except Exception as e: print(f"Warning: Error predicting with {model_name}: {e}")

    if not item_scores_all_models:
        print("No valid scores obtained from any model for this user. Returning empty ensemble.")
        return []

    # 2. Normalize Scores (Min-Max per item) and Calculate Weighted Score
    final_item_scores = {}
    for item_id, model_scores in item_scores_all_models.items():
        scores_list = list(model_scores.values())
        min_score, max_score = min(scores_list), max(scores_list)
        weighted_score = 0.0
        for model_name, raw_score in model_scores.items():
            normalized_score = 0.0
            if max_score > min_score: normalized_score = (raw_score - min_score) / (max_score - min_score)
            elif max_score == min_score and max_score > 0: normalized_score = 1.0
            weighted_score += normalized_score * MODEL_WEIGHTS.get(model_name, 0)
        final_item_scores[item_id] = weighted_score

    # 3. Sort by Final Weighted Score
    sorted_items = sorted(final_item_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_ensemble = sorted_items[:k]

    # 4. Format Results - ** Pass item_details explicitly **
    results = _format_recommendations(top_k_ensemble, item_details)

    end_time = time.time()
    print(f"Generated ensemble in {end_time - start_time:.3f}s using models: {valid_models_for_user}.")
    return results

# --- Primary Recommendation Service Function ---
# This is called by the main '/recommendations/{user_id}' endpoint
def get_recommendations_for_user(
    user_id: int,
    k: int,
    # FastAPI injects dependencies here
    models: Dict[str, BaseRecommender] = Depends(model_loader.get_models),
    all_items: Set[str] = Depends(model_loader.get_all_items),
    train_map: dict = Depends(model_loader.get_train_map),
    item_details: pd.DataFrame = Depends(model_loader.get_item_details),
) -> List[schemas.RecommendationItem]:
    """
    Generates top-k ensemble recommendations for a given user.
    Dependencies are injected here by FastAPI and passed down to the actual ensemble logic.
    """
    # Call the ensemble function, passing the dependencies received here
    return get_ensemble_recommendations(
        user_id=user_id,
        k=k,
        models=models,
        all_items=all_items,
        train_map=train_map,
        item_details=item_details
    )


# --- Metadata Services ---
# These remain the same as the previous corrected version
def get_users_list(search: Optional[str] = None, limit: int = 50) -> List[schemas.User]:
    """
    Loads and returns a list of unique user IDs from the processed user file,
    optionally filtered by a search query and limited.
    Ensures the correct user ID column is used.
    """
    users_path = config.PROCESSED_DATA_DIR / "users_final.parquet"
    if not users_path.exists():
        print(f"Error: Processed users file not found at {users_path}")
        raise HTTPException(status_code=500, detail=f"User data source file not found: {users_path}")

    try:
        users_df = pd.read_parquet(users_path)
        if 'student_id' in users_df.columns: user_id_col_in_df = 'student_id'
        elif config.USER_COL in users_df.columns: user_id_col_in_df = config.USER_COL; print(f"Note: User ID column in parquet is '{user_id_col_in_df}'.")
        else: raise HTTPException(status_code=500, detail=f"Cannot find required user ID column ('student_id' or '{config.USER_COL}') in {users_path}")

        # print(f"Using column '{user_id_col_in_df}' for user IDs from parquet.")
        if search:
            search_str = str(search).lower()
            filtered_df = users_df[users_df[user_id_col_in_df].astype(str).str.startswith(search_str)]
        else: filtered_df = users_df

        unique_user_ids = filtered_df.sort_values(by=user_id_col_in_df)[user_id_col_in_df].unique()[:limit]
        user_list = [schemas.User(student_id=int(uid)) for uid in unique_user_ids]
        # print(f"Returning {len(user_list)} users. (Search: '{search}', Limit: {limit})")
        return user_list
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"Error reading or processing users file {users_path}: {e}"); raise HTTPException(status_code=500, detail=f"Failed to retrieve user list: {e}")

def get_all_presentations() -> List[schemas.Presentation]:
    """Loads and returns details of all known presentations."""
    try:
        item_details = model_loader.get_item_details()
        presentations = []
        for pres_id, row in item_details.iterrows():
            presentations.append(schemas.Presentation(
                presentation_id=pres_id,
                module_id=row.get('module_id', 'N/A'),
                presentation_code=row.get('presentation_code', 'N/A'),
                module_presentation_length=row.get('module_presentation_length')
            ))
        return presentations
    except Exception as e: print(f"Error getting presentation details: {e}"); raise HTTPException(status_code=500, detail=f"Failed to retrieve presentation details: {e}")

def get_random_user() -> Optional[schemas.User]:
    """Selects a random user ID from the known processed users."""
    users_path = config.PROCESSED_DATA_DIR / "users_final.parquet"
    if not users_path.exists(): raise HTTPException(status_code=500, detail=f"User data source file not found: {users_path}")
    try:
        users_df = pd.read_parquet(users_path)
        if users_df.empty: return None
        if 'student_id' in users_df.columns: user_id_col_in_df = 'student_id'
        elif config.USER_COL in users_df.columns: user_id_col_in_df = config.USER_COL; print(f"Note: User ID column in parquet for random selection is '{user_id_col_in_df}'.")
        else: raise HTTPException(status_code=500, detail=f"Cannot find required user ID column ('student_id' or '{config.USER_COL}') in {users_path}")
        random_student_id = int(random.choice(users_df[user_id_col_in_df].unique()))
        return schemas.User(student_id=random_student_id)
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"Error getting random user from {users_path}: {e}"); raise HTTPException(status_code=500, detail=f"Failed to retrieve random user: {e}")