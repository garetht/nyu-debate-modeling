FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libffi-dev \
    libssl-dev \
    openssh-client \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-group cuda

COPY explorer /app/explorer
COPY run_orchestrator /app/run_orchestrator
COPY models /app/models
COPY schemas /app/schemas
COPY utils /app/utils
COPY data /app/data
COPY prompts /app/prompts
COPY scripts /app/scripts
COPY tests /app/tests
COPY train /app/train

ENV PATH="/app/.venv/bin:$PATH" \
    SRC_ROOT=/app/ \
    INPUT_ROOT=/app/data/ \
    PYTHONPATH=/app \
    EXPLORER_SSH_IDENTITY=/root/.ssh/lambda-labs.pem

EXPOSE 8067

CMD ["uvicorn", "explorer.explorer_backend.server:app", "--host", "0.0.0.0", "--port", "8067"]
