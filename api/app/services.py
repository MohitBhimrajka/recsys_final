# api/app/services.py
import pandas as pd
from typing import List, Set
from fastapi import Depends, HTTPException

# --- Adjust path to import from parent directory ---
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config
from src.models.base import BaseRecommender
from api.app import schemas, model_loader # Import schemas and loader module

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
    print(f"Type of user_id received: {type(user_id)}") # Add this

    known_users = model.get_known_users()
    if known_users: # Add check if set is not empty
        sample_known_user = next(iter(known_users)) # Get one user from the set
        print(f"Type of known users in model: {type(sample_known_user)}") # Add this
    else:
         print("Model knows 0 users!")


    # 1. Check if user is known
    if user_id not in known_users: # Check against the loaded set
        print(f"User {user_id} not found in model training data (Set contains {len(known_users)} users).")
        # Return empty list or raise HTTPException
        # raise HTTPException(status_code=404, detail=f"User {user_id} not found in model training data.")
        return [] # Return empty list for now

    # 2. Get items user interacted with in training
    items_seen_by_user = train_map.get(user_id, set())
    print(f"User {user_id} has seen {len(items_seen_by_user)} items in training.")

    # 3. Determine candidate items (all items minus seen items)
    candidate_items = list(all_items - items_seen_by_user)
    if not candidate_items:
        print(f"No candidate items left for user {user_id} after filtering seen items.")
        return []

    print(f"Predicting scores for {len(candidate_items)} candidate items.")
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
        details = item_details.loc[item_id] if item_id in item_details.index else {}
        results.append(schemas.RecommendationItem(
            presentation_id=item_id,
            score=float(score),
            module_id=details.get('module_id', 'N/A'),
            presentation_code=details.get('presentation_code', 'N/A'),
            module_presentation_length=details.get('module_presentation_length') # Will be None if missing
        ))

    print(f"Generated {len(results)} recommendations for user {user_id}.")
    return results

def get_all_users() -> List[schemas.User]:
    """Loads and returns a list of unique user IDs from the processed user file."""
    users_path = config.PROCESSED_DATA_DIR / "users_final.parquet"
    if not users_path.exists():
        print(f"Warning: Processed users file not found at {users_path}")
        return []
    try:
        users_df = pd.read_parquet(users_path)
        # Ensure the column name matches the one *after* rename in load_to_db
        # Assuming 'id_student' is the original and it gets loaded into the User model
        user_id_col = 'student_id' if 'student_id' in users_df.columns else config.USER_COL
        if user_id_col not in users_df.columns:
             print(f"Warning: Cannot find user ID column ({user_id_col} or {config.USER_COL}) in {users_path}")
             return []

        user_list = [schemas.User(student_id=uid) for uid in users_df[user_id_col].unique()]
        return user_list
    except Exception as e:
        print(f"Error reading or processing users file {users_path}: {e}")
        return []

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