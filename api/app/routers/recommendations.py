# api/app/routers/recommendations.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List

from api.app import schemas, services # Import schemas and services

router = APIRouter()

@router.get(
    "/recommendations/{user_id}",
    response_model=schemas.RecommendationResponse,
    summary="Get course recommendations for a user",
    tags=["Recommendations"]
)
async def get_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    recs: List[schemas.RecommendationItem] = Depends(services.get_recommendations_for_user) # Use dependency injection
):
    """
    Retrieves the top-k course presentation recommendations for a specific student ID.
    """
    if not recs and user_id in services.model_loader.get_model().get_known_users():
         # Handle case where user is known but no recs generated (e.g., saw all items)
         print(f"User {user_id} is known, but no recommendations generated (potentially saw all items).")
    elif not recs:
         # Case where user is unknown or other issue occurred in service handled by empty list return
         print(f"No recommendations returned for user {user_id}.")
         # Optionally raise 404 here if you prefer that over an empty list for unknown users
         # raise HTTPException(status_code=404, detail=f"User {user_id} not found or no recommendations available.")

    return schemas.RecommendationResponse(recommendations=recs)


@router.get(
    "/users",
    response_model=List[schemas.User],
    summary="Get list of users known to the system",
    tags=["Metadata"]
)
async def get_users():
    """
    Retrieves a list of student IDs present in the processed user data.
    Useful for populating selectors in a UI.
    """
    users = services.get_all_users()
    if not users:
         raise HTTPException(status_code=404, detail="No user data found.")
    return users


@router.get(
    "/presentations",
    response_model=List[schemas.Presentation],
    summary="Get list of presentations known to the system",
    tags=["Metadata"]
)
async def get_presentations():
    """
    Retrieves details about the course presentations known to the recommender model.
    """
    presentations = services.get_all_presentations()
    if not presentations:
         raise HTTPException(status_code=404, detail="No presentation data found.")
    return presentations