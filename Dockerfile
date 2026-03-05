# Pin base image to specific version for reproducibility
FROM python:3.11.8-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 mlbuser && \
  mkdir -p /app/artifacts && \
  chown -R mlbuser:mlbuser /app

# Copy dependency files
COPY --chown=mlbuser:mlbuser pyproject.toml README.md /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir ".[all]"

# Copy application code
COPY --chown=mlbuser:mlbuser src /app/src
COPY --chown=mlbuser:mlbuser scripts /app/scripts

# Set Python path
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER mlbuser

# Expose port for health checks
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run behind gunicorn for production traffic handling
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "300", "scripts.server:app"]
