import os
from dataclasses import dataclass, field


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
        default_factory=lambda: os.getenv("FRONTEND_DIR", "Frontend")
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

    @property
    def max_content_bytes(self) -> int:
        return self.max_content_mb * 1024 * 1024


settings = Settings()
