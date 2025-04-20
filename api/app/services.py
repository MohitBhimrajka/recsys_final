# api/app/services.py
import pandas as pd
from typing import List, Set, Optional # Added Optional
from fastapi import Depends, HTTPException
import random

# --- Adjust path to import from parent directory ---
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
from src.models.base import BaseRecommender
from api.app import schemas, model_loader # Import schemas and loader module

# --- Keep get_recommendations_for_user unchanged ---
def get_recommendations_for_user(
    user_id: int,
    k: int,
    model: BaseRecommender = Depends(model_loader.get_model), # Dependency inject model
    all_items: Set[str] = Depends(model_loader.get_all_items), # Dependency inject items
    train_map: dict = Depends(model_loader.get_train_map), # Dependency inject train map
    item_details: pd.DataFrame = Depends(model_loader.get_item_details) # Dependency inject details
) -> List[schemas.RecommendationItem]:
    """Generates top-k recommendations for a given user."""
    print(f"Generating recommendations for user: {user_id}, k={k}")
    # print(f"Type of user_id received: {type(user_id)}")

    known_users = model.get_known_users()
    # if known_users:
    #     sample_known_user = next(iter(known_users))
    #     print(f"Type of known users in model: {type(sample_known_user)}")
    # else:
    #      print("Model knows 0 users!")

    # 1. Check if user is known
    if user_id not in known_users:
        print(f"User {user_id} not found in model training data (Set contains {len(known_users)} users).")
        return [] # Return empty list

    # 2. Get items user interacted with in training
    items_seen_by_user = train_map.get(user_id, set())
    # print(f"User {user_id} has seen {len(items_seen_by_user)} items in training.")

    # 3. Determine candidate items (all items minus seen items)
    candidate_items = list(all_items - items_seen_by_user)
    if not candidate_items:
        # print(f"No candidate items left for user {user_id} after filtering seen items.")
        return []

    # print(f"Predicting scores for {len(candidate_items)} candidate items.")
    # 4. Predict scores for candidate items
    try:
        scores = model.predict(user_id, candidate_items)
    except Exception as e:
        print(f"Error during model prediction for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error predicting scores.")

    if len(scores) != len(candidate_items):
        print(f"Error: Score prediction length mismatch for user {user_id}.")
        raise HTTPException(status_code=500, detail="Prediction length mismatch.")

    # 5. Combine items and scores, sort, and take top K
    item_score_pairs = list(zip(candidate_items, scores))
    item_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_k_pairs = item_score_pairs[:k]

    # 6. Format results using the schema, enriching with details
    results = []
    for item_id, score in top_k_pairs:
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
             print(f"Warning: Error fetching details for item {item_id}: {detail_error}")
             # Append with placeholder details if lookup fails
             results.append(schemas.RecommendationItem(
                 presentation_id=item_id,
                 score=float(score),
                 module_id='Error',
                 presentation_code='Error',
                 module_presentation_length=None
             ))


    # print(f"Generated {len(results)} recommendations for user {user_id}.")
    return results

# --- MODIFIED: get_all_users becomes get_users_list with filtering ---
def get_users_list(search: Optional[str] = None, limit: int = 50) -> List[schemas.User]:
    """
    Loads and returns a list of unique user IDs from the processed user file,
    optionally filtered by a search query and limited.
    """
    users_path = config.PROCESSED_DATA_DIR / "users_final.parquet"
    if not users_path.exists():
        print(f"Error: Processed users file not found at {users_path}")
        # Return empty list or raise HTTPException, depending on desired API behavior
        # raise HTTPException(status_code=404, detail="User data source not found.")
        return []
    try:
        users_df = pd.read_parquet(users_path)

        # Determine the user ID column name (check for rename)
        # The User schema expects 'student_id'
        if 'student_id' in users_df.columns:
            user_id_col = 'student_id'
        elif config.USER_COL in users_df.columns:
            user_id_col = config.USER_COL # Use original if rename wasn't persisted/done yet
            print(f"Warning: Using original user column '{config.USER_COL}'. Ensure it matches the User schema or rename in preprocessing/loading.")
        else:
             print(f"Error: Cannot find user ID column ('student_id' or '{config.USER_COL}') in {users_path}")
             return []

        # --- Filtering Logic ---
        if search:
            # Convert search query and user IDs to string for consistent matching
            search_str = str(search).lower()
            # Filter users whose ID *starts with* the search string
            filtered_df = users_df[users_df[user_id_col].astype(str).str.startswith(search_str)]
            print(f"Filtered users by query '{search}'. Found {len(filtered_df)} matches before limit.")
        else:
            # No search query, return all users (up to the limit)
            filtered_df = users_df

        # Apply limit and get unique IDs
        # Ensure sorting for predictable results when no search term is provided
        unique_user_ids = filtered_df.sort_values(by=user_id_col)[user_id_col].unique()[:limit]

        # Create list of Pydantic models
        user_list = [schemas.User(student_id=int(uid)) for uid in unique_user_ids]

        return user_list

    except Exception as e:
        print(f"Error reading or processing users file {users_path}: {e}")
        # Depending on the error, might want to raise HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user list: {e}")
        # return []


# --- Keep get_all_presentations unchanged ---
def get_all_presentations() -> List[schemas.Presentation]:
    """Loads and returns details of all known presentations."""
    try:
        item_details = model_loader.get_item_details() # Get from loaded data
        presentations = []
        # item_details index is presentation_id
        for pres_id, row in item_details.iterrows():
            presentations.append(schemas.Presentation(
                presentation_id=pres_id,
                module_id=row.get('module_id', 'N/A'),
                presentation_code=row.get('presentation_code', 'N/A'),
                module_presentation_length=row.get('module_presentation_length')
            ))
        return presentations
    except Exception as e:
        print(f"Error getting presentation details: {e}")
        return []
    
def get_random_user() -> Optional[schemas.User]:
    """Selects a random user ID from the known processed users."""
    users_path = config.PROCESSED_DATA_DIR / "users_final.parquet"
    if not users_path.exists():
        print(f"Error: Processed users file not found at {users_path}")
        return None # Or raise appropriate exception
    try:
        users_df = pd.read_parquet(users_path)
        user_id_col = 'student_id' if 'student_id' in users_df.columns else config.USER_COL
        if user_id_col not in users_df.columns:
             print(f"Error: Cannot find user ID column in {users_path}")
             return None

        if users_df.empty:
            return None

        # Select a random student ID from the correct column
        random_student_id = int(random.choice(users_df[user_id_col].unique()))
        return schemas.User(student_id=random_student_id)

    except Exception as e:
        print(f"Error getting random user from {users_path}: {e}")
        return None # Or raise exception