# api/app/routers/recommendations.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional # Added Optional

from api.app import schemas, services # Import schemas and services

router = APIRouter()

# --- Keep /recommendations/{user_id} endpoint unchanged ---
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
    if not recs:
        try:
            # Attempt to check if user is known only if recs list is empty
            known_users = services.model_loader.get_model().get_known_users()
            if user_id in known_users:
                print(f"User {user_id} is known, but no recommendations generated (potentially saw all items).")
            else:
                # User is unknown to the model
                print(f"User {user_id} not found in model. Returning empty recommendations.")
                # You could optionally raise 404 here instead of returning empty
                # raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
        except Exception as e:
            # Handle cases where model loading might have failed earlier
            print(f"Could not check if user {user_id} is known due to error: {e}")

    return schemas.RecommendationResponse(recommendations=recs)


# --- MODIFIED: Update /users endpoint ---
@router.get(
    "/users",
    response_model=List[schemas.User],
    summary="Get list of users, optionally filtered by search query",
    tags=["Metadata"]
)
async def get_user_list(
    search: Optional[str] = Query(None, description="Filter users whose student ID starts with this query string."),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of users to return.")
    # Inject the MODIFIED service function
    # users: List[schemas.User] = Depends(services.get_users_list) <- Incorrect DI for params
):
    """
    Retrieves a list of student IDs, optionally filtered by the beginning of the ID.
    Returns a limited number of results. Useful for populating asynchronous selectors.
    """
    # Call the service function directly, passing the query parameters
    try:
        users = services.get_users_list(search=search, limit=limit)
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions from the service layer
        raise http_exc
    except Exception as e:
        # Catch other unexpected errors
        print(f"Unexpected error getting user list: {e}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving user list.")

    # Note: We don't raise 404 here if the list is empty,
    # an empty list is a valid response (e.g., no users match the search).
    # The frontend can handle the empty list display.
    return users


# --- Keep /presentations endpoint unchanged ---
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
         # Can keep 404 here if no presentation data is truly a failure state
         raise HTTPException(status_code=404, detail="No presentation data found.")
    return presentations

@router.get(
    "/users/random",
    response_model=Optional[schemas.User], # Can return null if no users found
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
        # Raise 404 if no users exist in the source file at all
        raise HTTPException(status_code=404, detail="No users found in the dataset.")
    return user