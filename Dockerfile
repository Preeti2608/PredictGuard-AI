# Python 3.13 slim image for minimal attack surface and lightweight deployment
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies if required by Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files, trained models, and sample assets
COPY main.py index.html ./
COPY models/ ./models/
COPY sample_images/ ./sample_images/

# Expose standard port
EXPOSE 8000

# Health check using the /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start Uvicorn bound to dynamic platform port
CMD ["sh", "-c", "uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
