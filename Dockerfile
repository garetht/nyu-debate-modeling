# Build stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies

RUN uv venv
RUN uv sync

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV SRC_ROOT=/app/
ENV INPUT_ROOT=/app/data/

CMD ["/bin/bash"]
