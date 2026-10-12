# ==========================================
# STAGE 1: BUILDER (Includes C++ compiler)
# ==========================================
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    FLASH_ATTN_CUDA_ARCHS="89" \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=10

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git python3 python3-pip python3-venv python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

WORKDIR /app

# 1. Compile FlashAttention first in isolation for layer caching
RUN uv venv /app/.venv && \
    uv pip install setuptools wheel ninja packaging && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install flash-attn --no-build-isolation

# 2. Copy dependency manifests
COPY pyproject.toml uv.lock ./

# 3. Install remaining dependencies
RUN uv sync

# ==========================================
# STAGE 2: RUNTIME (Minimal, without compiler)
# ==========================================
# Use lightweight Ubuntu image.
# No CUDA base image required as PyTorch (cu128) bundles its own CUDA runtime libraries.
FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python, SoX, and FFmpeg runtime audio libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv sox libsox-fmt-all ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Prepend virtualenv to PATH for direct execution
ENV PATH="/app/.venv/bin:$PATH"

# Copy default server implementation into container
COPY legacy_scripts/server_failed.py ./server.py

# Default entrypoint for Uvicorn service
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
