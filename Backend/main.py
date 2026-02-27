import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from Backend.api import health as health_api
from Backend.api import process as process_api
from Backend.api import search as search_api
from Backend.core.config import settings
from Backend.core.lifespan import lifespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="NutriSense-AI",
    description="AI nutrition intelligence for Indian cuisine.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(process_api.router)
app.include_router(health_api.router)
app.include_router(search_api.router)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."},
    )


# Static files
# Set SERVE_STATIC=false to disable when using Nginx or CDN in production.
if settings.serve_static and os.path.isdir(settings.frontend_dir):
    app.mount(
        "/",
        StaticFiles(directory=settings.frontend_dir, html=True),
        name="frontend",
    )
    logger.info("Serving static files from '%s'", settings.frontend_dir)
else:
    logger.info(
        "Static file serving disabled (SERVE_STATIC=false or dir not found)."
    )

# Entry point
if __name__ == "__main__":
    import pathlib
    import sys
    # Add project root to sys.path so Backend package is resolvable
    # when this file is run directly (python Backend/main.py).
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    uvicorn.run(
        "Backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
