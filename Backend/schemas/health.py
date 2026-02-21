from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    recipes: int
    ingredients: int
    cuisines: int
    image_classes: int
    image_model_loaded: bool
    num_classes: int
    data_source: str = "neo4j"
