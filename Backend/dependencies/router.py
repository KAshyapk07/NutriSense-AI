from __future__ import annotations

import logging
from typing import Optional

from Src.LLM.llm_client import OllamaLLMClient
from Src.LLM.llm_engine import LLMEngine
from Src.Router.Router import NutriSenseRouter
from Backend.dependencies.neo4j import get_neo4j_client
from Backend.dependencies.model import get_image_model

logger = logging.getLogger(__name__)

_router: Optional[NutriSenseRouter] = None


def init() -> None:
    global _router
    neo4j_client = get_neo4j_client()
    image_model = get_image_model()

    llm_client = OllamaLLMClient()
    llm_engine = LLMEngine(llm_client)
    _router = NutriSenseRouter(
        neo4j_client=neo4j_client,
        llm_engine=llm_engine,
        image_model=image_model,
    )
    logger.info("NutriSenseRouter initialized.")


def get_router() -> NutriSenseRouter:
    if _router is None:
        raise RuntimeError("Router is not initialized.")
    return _router
