import base64
import csv
import io
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from Backend.dependencies.auth_user import _decode_access_token
from Backend.dependencies.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Report"])

ADMIN_TOKEN = os.getenv("REPORT_ADMIN_TOKEN", "")
_CSV_HEADERS = ["id", "timestamp", "user_email", "query", "response_type", "description", "has_image"]

_IMAGE_LIMIT = 2 * 1024 * 1024  # 2 MB


def _user_email_from_header(authorization: str | None) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        payload = _decode_access_token(authorization[7:])
        if payload:
            return payload.get("email", "") or payload.get("sub", "")
    return ""


class ReportRequest(BaseModel):
    description: str
    query: str | None = None
    response_type: str | None = None


@router.post("/report", status_code=201)
async def submit_report(
    body: ReportRequest,
    authorization: str | None = Header(default=None),
):
    user_email = _user_email_from_header(authorization)
    get_neo4j_client().save_report(
        report_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_email=user_email,
        query=(body.query or "")[:500],
        response_type=(body.response_type or ""),
        description=body.description[:2000],
    )
    logger.info("Report saved — user=%s type=%s", user_email or "anon", body.response_type)
    return {"status": "received"}


@router.post("/report/feedback", status_code=201)
async def submit_feedback_report(
    description: str = Form(...),
    query: str = Form(default=""),
    response_type: str = Form(default="settings"),
    image: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
):
    user_email = _user_email_from_header(authorization)

    image_b64 = ""
    if image and image.filename:
        content = await image.read(_IMAGE_LIMIT + 1)
        if len(content) > _IMAGE_LIMIT:
            raise HTTPException(status_code=413, detail="Image must be under 2 MB.")
        image_b64 = base64.b64encode(content).decode()

    get_neo4j_client().save_report(
        report_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_email=user_email,
        query=query[:500],
        response_type=response_type,
        description=description[:2000],
        image_b64=image_b64,
    )
    logger.info("Feedback report saved — user=%s has_image=%s", user_email or "anon", bool(image_b64))
    return {"status": "received"}


@router.get("/report/download")
async def download_reports(token: str = ""):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden.")

    reports = get_neo4j_client().get_all_reports()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_CSV_HEADERS)
    for r in reports:
        writer.writerow([
            r.get("id", ""),
            r.get("timestamp", ""),
            r.get("user_email", ""),
            r.get("query", ""),
            r.get("response_type", ""),
            r.get("description", ""),
            "yes" if r.get("image_b64") else "no",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nutriverse_reports.csv"},
    )
