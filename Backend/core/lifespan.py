import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from Backend.core.config import settings
from Backend.dependencies import neo4j as neo4j_dep
from Backend.dependencies import model as model_dep
from Backend.dependencies import router as router_dep
from Backend.dependencies import graph_rag as graph_rag_dep

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting NutriSense-AI backend...")

    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info("Upload directory ready: %s", settings.upload_dir)

    neo4j_dep.init()
    model_dep.init()
    router_dep.init()

    # Phase 3 — GraphRAG: initialise embedding model + vector search service
    graph_rag_dep.init(neo4j_dep.get_neo4j_client())

    logger.info("Startup complete. Ready to serve requests.")

    yield

    # Shutdown
    logger.info("Shutting down NutriSense-AI backend...")
    neo4j_dep.close()
    logger.info("Shutdown complete.")
