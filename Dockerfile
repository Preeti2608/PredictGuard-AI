# ==============================================================================
# PredictGuard AI — Production Multi-Stage Dockerfile
# Optimized for minimal attack surface, CPU inference, and non-root execution
# ==============================================================================

# Stage 1: Build stage for dependency compilation
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Substitute full tensorflow with lightweight tensorflow-cpu for production container
RUN sed -i 's/^tensorflow==/tensorflow-cpu==/' requirements.txt && \
    pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Minimal Production Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install curl for container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application assets and models with unprivileged ownership
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser sample_images/ ./sample_images/
COPY --chown=appuser:appuser index.html ./index.html

# Run as non-root user
USER appuser

EXPOSE 8000

# Container readiness healthcheck polling /health/ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health/ready || exit 1

# Start Uvicorn bound to runtime host and port
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
