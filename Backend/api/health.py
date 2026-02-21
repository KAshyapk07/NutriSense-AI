import logging

from fastapi import APIRouter, Depends, HTTPException

from Backend.dependencies.neo4j import get_neo4j_client
from Backend.dependencies.model import get_image_model
from Backend.schemas.health import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    neo4j_client=Depends(get_neo4j_client),
    image_model=Depends(get_image_model),
):
    try:
        stats = neo4j_client.get_stats()
    except Exception as exc:
        logger.error("Health check — Neo4j error: %s", exc)
        raise HTTPException(status_code=503, detail="Neo4j unavailable.")

    return HealthResponse(
        status="ok",
        recipes=stats.get("recipes", 0),
        ingredients=stats.get("ingredients", 0),
        cuisines=stats.get("cuisines", 0),
        image_classes=stats.get("image_classes", 0),
        image_model_loaded=image_model is not None,
        num_classes=image_model.num_classes,
    )
