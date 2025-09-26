FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Set CUDA architecture flags for PyTorch
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install python 3.9 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9

# Upgrade pip and install torch build dependencies
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install setuptools wheel ninja

# Checkpoint this layer
FROM base AS pytorch

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /fireants

# Copy the code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Install fused ops
RUN pip install ./fused_ops