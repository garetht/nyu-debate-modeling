# Build stage
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

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

# install everything else
RUN uv sync --no-group cuda

ENV MAX_JOBS=14

RUN uv pip install --force-reinstall --upgrade flash-attn==2.8.0.post2 --no-build-isolation

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV SRC_ROOT=/app/
ENV INPUT_ROOT=/app/data/
ENV OPENAI_ORGANIZATION="org-vt1Xse3GWhm4hiaZutOuXVbn"
# ENV OPENAI_API_KEY
# ENV

ENTRYPOINT ["python"]
