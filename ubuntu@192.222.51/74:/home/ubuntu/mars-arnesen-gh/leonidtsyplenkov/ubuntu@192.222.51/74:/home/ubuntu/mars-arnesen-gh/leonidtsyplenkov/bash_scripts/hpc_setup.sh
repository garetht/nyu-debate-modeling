#!/bin/bash
set -e

pip install uv==0.7.19

uv venv --allow-existing
uv sync --no-group cuda

# Installing Python dependencies that depend on the runtime GPU environment
uv pip install bitsandbytes==0.46.1;
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128;

# increasing MAX_JOBS to more than this will cause processes to be killed due to
# likely out of memory errors, even with 200+GB of memory on the machine
CUDA_HOME=/usr/local/cuda MAX_JOBS=24 uv pip install --force-reinstall -v --upgrade flash-attn==2.8.0.post2 --no-build-isolation;

source .venv/bin/activate

python scripts/huggingface_downloader.py gradientai/Llama-3-8B-Instruct-262k ./downloaded-models/gradientai/Llama-3-8B-Instruct-262k

set +e
