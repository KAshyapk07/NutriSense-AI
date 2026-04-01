# ══════════════════════════════════════════════════════════════════════
# NutriSense-AI Backend — Production Docker Image
# ══════════════════════════════════════════════════════════════════════
# Single container packaging:
#   - FastAPI backend
#   - ConvNeXt-Small image classification model
#   - Faster-Whisper STT (small.en model auto-downloaded on first run)
#   - All Python dependencies
#
# External cloud services (configured via environment variables):
#   - Groq (LLM inference)
#   - Neo4j AuraDB (knowledge graph)
#   - Firebase (authentication)
# ══════════════════════════════════════════════════════════════════════

# ── Base stage: Python 3.11 slim ──────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies for PyTorch, PyAV (bundled with faster-whisper), and Neo4j driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies stage ────────────────────────────────────────────────
FROM base AS dependencies

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Production stage ──────────────────────────────────────────────────
FROM base AS production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY Backend/ ./Backend/
COPY Src/ ./Src/
COPY run.py .

# Ensure ConvNeXt model is present
# (Required file: Src/Image_classifier/models/nutrisense_convnext_small_best.pth)
RUN test -f Src/Image_classifier/models/nutrisense_convnext_small_best.pth || \
    (echo "ERROR: ConvNeXt model file not found. Please ensure Src/Image_classifier/models/nutrisense_convnext_small_best.pth exists." && exit 1)

# Create temp_uploads directory for image processing
RUN mkdir -p temp_uploads

# Expose FastAPI port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Environment defaults (override in deployment platform)
ENV MODEL_PATH=Src/Image_classifier/models/nutrisense_convnext_small_best.pth \
    STT_MODEL_SIZE=small.en \
    STT_DEVICE=cpu \
    STT_COMPUTE_TYPE=int8 \
    STT_BACKEND=faster-whisper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run FastAPI with uvicorn
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
