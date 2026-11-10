# fedcrop: results dashboard + live CSV prediction in one container.
#
#   docker build -t fedcrop .
#   docker run --rm -p 8000:8000 fedcrop      ->  http://localhost:8000
#
# Build from your own checkout: the seed-42 checkpoints and scalers.json are
# gitignored, so a fresh clone from GitHub does not contain them.

# ---- stage 1: build the React app ------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: Python service --------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FEDCROP_FRONTEND_DIR=/app/frontend/dist \
    PORT=8000

WORKDIR /app

# CPU-only PyTorch: about 200 MB instead of several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY src/ src/
COPY config/ config/
COPY backend/ backend/
COPY artifacts/results/ artifacts/results/
COPY artifacts/scalers/ artifacts/scalers/
COPY artifacts/models/ artifacts/models/
COPY --from=frontend /fe/dist frontend/dist

# Run as an unprivileged user (uid 1000 also suits Hugging Face Spaces).
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)" || exit 1

# PORT is read at start-up so a host that assigns its own port can set it.
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
