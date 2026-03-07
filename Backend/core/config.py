import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    serve_static: bool = field(
        default_factory=lambda: os.getenv("SERVE_STATIC", "true").lower() == "true"
    )
    upload_dir: str = field(
        default_factory=lambda: os.getenv("UPLOAD_DIR", "temp_uploads")
    )
    max_content_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTENT_MB", "16"))
    )
    frontend_dir: str = field(
        default_factory=lambda: os.getenv("FRONTEND_DIR", "frontend/dist")
    )
    host: str = field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "8000"))
    )
    allowed_extensions: frozenset = field(
        default_factory=lambda: frozenset({"png", "jpg", "jpeg", "gif", "bmp", "webp"})
    )
    # ── Public URL for tunnelling / production deployment ──────────────────
    # Set this to the externally reachable base URL of the app.
    #
    #   Local ngrok testing:   PUBLIC_URL=https://<id>.ngrok-free.app
    #   Production (Render):   PUBLIC_URL=https://nutrisense.onrender.com
    #   Production (custom):   PUBLIC_URL=https://nutrisense.yourdomain.com
    #
    # When set, GET /config returns this URL so the React frontend can build
    # the correct QR code URL for the P2P Kitchen Remote feature without
    # needing a Vite rebuild.  Leave unset during plain localhost development
    # (the frontend falls back to window.location.origin automatically).
    public_url: Optional[str] = field(
        default_factory=lambda: os.getenv("PUBLIC_URL") or None
    )

    @property
    def max_content_bytes(self) -> int:
        return self.max_content_mb * 1024 * 1024


settings = Settings()
