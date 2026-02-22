FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir ".[all]"

COPY src /app/src
COPY scripts /app/scripts

ENV PYTHONPATH=/app/src

CMD ["python", "scripts/daily_run.py"]
