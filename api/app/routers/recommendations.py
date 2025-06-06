# api/app/routers/recommendations.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict # Added Dict

from api.app import schemas, services # Import schemas and services

router = APIRouter()

# --- UPDATED: /recommendations/{user_id} now returns ENSEMBLE results ---
@router.get(
    "/recommendations/{user_id}",
    response_model=schemas.RecommendationResponse,
    summary="Get ENSEMBLE course recommendations for a user", # Updated summary
    tags=["Recommendations"]
)
async def get_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    # Depends on the updated service function which now calls the ensemble logic
    recs: List[schemas.RecommendationItem] = Depends(services.get_recommendations_for_user)
):
    """
    Retrieves the top-k course presentation recommendations for a specific student ID,
    generated by combining results from multiple underlying models (ensemble).
    """
    # Note: Checking if user is known by *a specific* model isn't straightforward here,
    # as the ensemble might produce results even if one model doesn't know the user.
    # The service layer handles cases where *no* model produces results.
    if not recs and user_id: # Check if user_id is defined
         # Simplified check: If no recs, it means either user unknown to *all* models,
         # or saw all items across all models, or an error occurred upstream.
         # The service layer should log specifics. Here, just return empty.
         print(f"No ensemble recommendations generated for user {user_id}.")
         # Optionally raise 404, but empty list is valid if user saw everything
         # raise HTTPException(status_code=404, detail=f"No recommendations could be generated for user {user_id}.")
         pass # Return empty list via the response model

    return schemas.RecommendationResponse(recommendations=recs)

# --- NEW: Endpoint to get results from ALL models individually ---
@router.get(
    "/recommendations/{user_id}/all_models",
    response_model=schemas.AllModelsRecommendationResponse, # Use the new schema
    summary="Get recommendations from each individual model",
    tags=["Recommendations"]
)
async def get_all_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations per model"),
    # Depends on the new service function
    all_recs_dict: Dict[str, List[schemas.RecommendationItem]] = Depends(services.get_all_model_recommendations)
):
    """
    Retrieves the top-k course presentation recommendations from *each*
    individual model loaded in the backend (e.g., ItemCF, NCF, ALS).
    Useful for comparison and understanding individual model behavior.
    """
    if not all_recs_dict:
         # This case means the service function returned an empty dict,
         # likely because the user had no candidate items or an error occurred.
         raise HTTPException(status_code=404, detail=f"Could not retrieve recommendations from any model for user {user_id}.")

    # Check if *any* model returned recommendations
    has_recommendations = any(bool(recs) for recs in all_recs_dict.values())
    if not has_recommendations:
        print(f"User {user_id} is known, but no individual model generated recommendations (potentially saw all items or user unknown to specific models).")
        # Return the dictionary with empty lists, which is valid
        pass

    return schemas.AllModelsRecommendationResponse(results=all_recs_dict)


# --- Metadata Endpoints (Unchanged) ---
@router.get(
    "/users",
    response_model=List[schemas.User],
    summary="Get list of users, optionally filtered by search query",
    tags=["Metadata"]
)
async def get_user_list(
    search: Optional[str] = Query(None, description="Filter users whose student ID starts with this query string."),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of users to return.")
):
    """
    Retrieves a list of student IDs, optionally filtered by the beginning of the ID.
    Returns a limited number of results. Useful for populating asynchronous selectors.
    """
    try:
        users = services.get_users_list(search=search, limit=limit)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error getting user list: {e}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving user list.")
    return users

@router.get(
    "/presentations",
    response_model=List[schemas.Presentation],
    summary="Get list of presentations known to the system",
    tags=["Metadata"]
)
async def get_presentations():
    """
    Retrieves details about the course presentations known to the recommender models.
    """
    presentations = services.get_all_presentations()
    if not presentations:
         raise HTTPException(status_code=404, detail="No presentation data found.")
    return presentations

@router.get(
    "/users/random",
    response_model=Optional[schemas.User],
    summary="Get a random user ID from the system",
    tags=["Metadata"]
)
async def get_a_random_user():
    """
    Retrieves a single random student ID from the processed user data.
    Useful for quickly populating the demo. Returns null if no users are available.
    """
    user = services.get_random_user()
    if user is None:
        raise HTTPException(status_code=404, detail="No users found in the dataset.")
    return user