# Pin base image to specific version for reproducibility
FROM python:3.12-slim AS base

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
COPY --chown=mlbuser:mlbuser pyproject.toml README.md alembic.ini /app/

# Copy application code needed to build/install package
COPY --chown=mlbuser:mlbuser src /app/src
COPY --chown=mlbuser:mlbuser scripts /app/scripts

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir .

# Set Python path
ENV PYTHONPATH=/app/src

# Switch to non-root user BEFORE bootstrapping models so files are owned by mlbuser
USER mlbuser

# Copy pre-trained model artifacts (overwrite bootstrap if they exist locally)
COPY --chown=mlbuser:mlbuser artifacts/models/ /app/artifacts/models/

# Ensure model artifacts exist and are writable by mlbuser
RUN mkdir -p /app/artifacts/models \
  && python /app/scripts/bootstrap_models.py --if-missing

# Expose port for health checks
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run behind gunicorn for production traffic handling
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "300", "scripts.server:app"]
