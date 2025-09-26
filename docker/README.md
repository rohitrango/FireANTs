# Docker Instructions

This directory contains the Dockerfile for setting up the FireANTs environment with CUDA support.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Building the Docker Image

First, clone the repository:

```bash
git clone https://github.com/rohitrango/fireants
cd fireants
```

Then, from the root directory of the project, run:

```bash
docker buildx build -t fireants -f docker/Dockerfile .
```

This will create a Docker image named `fireants` with all the necessary dependencies installed.

## Running the Container

To run the container with GPU support:

```bash
docker run --gpus all --shm-size=40g -it -v /data:/data fireants
```

To mount your local code directory for development:

```bash
docker run --gpus all --shm-size=40g -it -v $(pwd):/fireants -v /data:/data fireants
```

You may need to rebuild the `fused_ops` extension after mounting the code directory if you are making changes to the `fused_ops` kernels.
