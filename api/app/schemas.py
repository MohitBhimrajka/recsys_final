# api/app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class RecommendationItem(BaseModel):
    """Schema for a single recommendation item."""
    presentation_id: str
    score: float
    module_id: str
    presentation_code: str
    module_presentation_length: Optional[int] = None # Make optional if length might be missing

    class Config:
        from_attributes = True # Changed in Pydantic v2 from orm_mode=True

class RecommendationResponse(BaseModel):
    """Schema for the list of recommendations."""
    recommendations: List[RecommendationItem]

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