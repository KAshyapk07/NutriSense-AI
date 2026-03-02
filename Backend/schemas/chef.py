from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChefParseRequest(BaseModel):
    recipe_name: str = Field(..., min_length=1, description="Name of the dish")
    instructions: Optional[str] = Field(
        None,
        description="Raw cooking instructions text. If not provided, the AI will generate steps from the dish name and ingredients.",
    )
    ingredients: Optional[str] = Field(None, description="Raw ingredients list (optional)")


class MiseEnPlaceItem(BaseModel):
    id: int
    text: str
    duration_minutes: Optional[int] = None


class CookStep(BaseModel):
    id: int
    action: str
    timer_seconds: Optional[int] = None
    tool: Optional[str] = None
    tip: Optional[str] = None


class ChefParseResponse(BaseModel):
    recipe_name: str
    mise_en_place: List[MiseEnPlaceItem] = Field(default_factory=list)
    steps: List[CookStep] = Field(default_factory=list)
    tools_required: List[str] = Field(default_factory=list)
    estimated_total_minutes: Optional[int] = None
    parse_error: Optional[str] = None
