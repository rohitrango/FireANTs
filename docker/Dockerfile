FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# System dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    build-essential \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment for all Python deps
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

# Upgrade pip and build tooling
RUN python -m pip install --upgrade pip setuptools wheel ninja

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Build context (pruned via .dockerignore)
WORKDIR /build
COPY . .

# Install the main package into the venv
RUN pip install --no-cache-dir .

# Build and install fused ops into the same venv
RUN cd fused_ops && python setup.py build_ext && python setup.py install && cd ..

# --------------------------
# Minimal runtime image
# --------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Only runtime necessities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Bring the prebuilt virtualenv from the base stage
COPY --from=base /opt/venv /opt/venv

# Use the venv by default
ENV PATH=/opt/venv/bin:${PATH}
ENV VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Default: print versions (override with your own CMD/ENTRYPOINT)
ENTRYPOINT ["/bin/bash"]
