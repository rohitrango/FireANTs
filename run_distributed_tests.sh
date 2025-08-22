#!/bin/bash
# Run distributed tests with torchrun
NCCL_DEBUG=INFO torchrun --nproc-per-node=4 -m pytest tests/test_parallel_state.py -v
