FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir .

COPY src /app/src
COPY scripts /app/scripts

ENV PYTHONPATH=/app/src

CMD ["python", "scripts/train.py", "--help"]
