# api/app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict # Added Dict

class RecommendationItem(BaseModel):
    """Schema for a single recommendation item."""
    presentation_id: str
    score: float
    module_id: str
    presentation_code: str
    module_presentation_length: Optional[int] = None

    class Config:
        from_attributes = True

class RecommendationResponse(BaseModel):
    """Schema for the list of recommendations (typically ensemble)."""
    recommendations: List[RecommendationItem]

# --- NEW SCHEMA for All Models Endpoint ---
class AllModelsRecommendationResponse(BaseModel):
    """Schema for the dictionary of recommendations from all models."""
    # The key will be the model name (e.g., "ItemCF", "NCF (e=15)")
    results: Dict[str, List[RecommendationItem]]

class User(BaseModel):
    """Basic User schema for listing."""
    student_id: int

    class Config:
        from_attributes = True

class Presentation(BaseModel):
    """Schema for Presentation details."""
    presentation_id: str
    module_id: str
    presentation_code: str
    module_presentation_length: Optional[int] = None

    class Config:
        from_attributes = True