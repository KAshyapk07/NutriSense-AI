import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from Backend.core.config import settings
from Backend.dependencies.router import get_router
from Backend.schemas.process import ProcessResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Process"])


def _validate_extension(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.allowed_extensions


@router.post("/process", response_model=ProcessResponse)
async def process(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    nutri_router=Depends(get_router),
):
    has_query = query and query.strip()
    has_image = image is not None and image.filename

    if not has_query and not has_image:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: 'query' (text) or 'image' (file).",
        )

    image_path: Optional[str] = None

    try:
        if has_image:
            if not _validate_extension(image.filename):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type. "
                        f"Allowed: {', '.join(sorted(settings.allowed_extensions))}"
                    ),
                )

            contents = await image.read()

            if len(contents) > settings.max_content_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.max_content_mb} MB limit.",
                )

            safe_name = f"{uuid.uuid4().hex}_{os.path.basename(image.filename)}"
            image_path = os.path.join(settings.upload_dir, safe_name)

            with open(image_path, "wb") as f:
                f.write(contents)

        result = await nutri_router.execute_async(
            text_query=query or "",
            image_input=image_path,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /process: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", image_path, exc)
