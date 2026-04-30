import csv
import logging
import os
import pathlib
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from Backend.dependencies.auth_user import _decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Report"])

_REPORT_DIR = pathlib.Path(os.getenv("REPORT_DIR", "data/reports"))
_REPORT_FILE = _REPORT_DIR / "issue_reports.csv"
_CSV_HEADERS = ["id", "timestamp", "user_email", "query", "response_type", "description"]

ADMIN_TOKEN = os.getenv("REPORT_ADMIN_TOKEN", "")


def _ensure_csv():
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not _REPORT_FILE.exists():
        with open(_REPORT_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_HEADERS)


class ReportRequest(BaseModel):
    description: str
    query: str | None = None
    response_type: str | None = None


@router.post("/report", status_code=201)
async def submit_report(
    body: ReportRequest,
    authorization: str | None = Header(default=None),
):
    user_email = ""
    if authorization and authorization.lower().startswith("bearer "):
        payload = _decode_access_token(authorization[7:])
        if payload:
            user_email = payload.get("email", "") or payload.get("sub", "")

    _ensure_csv()
    row = [
        str(uuid.uuid4()),
        datetime.now(timezone.utc).isoformat(),
        user_email,
        (body.query or "")[:500],
        (body.response_type or ""),
        body.description[:2000],
    ]
    with open(_REPORT_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    logger.info("Issue report saved — user=%s type=%s", user_email or "anon", body.response_type)
    return {"status": "received"}


@router.get("/report/download")
async def download_reports(token: str = ""):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden.")
    _ensure_csv()
    return FileResponse(
        path=str(_REPORT_FILE),
        media_type="text/csv",
        filename="nutriverse_issue_reports.csv",
    )
