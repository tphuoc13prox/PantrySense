FROM python:3.12-slim

# libgomp is needed by lightgbm and faiss at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer is cached across code changes. The CPU-only
# torch index avoids the CUDA wheels that sentence-transformers would pull in.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend

# The dataset and built indices are not baked into the image. Mount them at
# /app/data (see .dockerignore). A browser cannot be opened from a container,
# the idle shutdown would stop it when no tab is polling, and the automatic
# index build would try to download and process the full dataset on startup.
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

ENV PANTRYSENSE_AUTO_OPEN_BROWSER=false \
    PANTRYSENSE_AUTO_SHUTDOWN=false \
    PANTRYSENSE_AUTO_BUILD_INDEX=false \
    HF_HOME=/app/data/.hf-cache

VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
