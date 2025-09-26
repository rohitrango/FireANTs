#!/bin/bash
set -e
set -x

# Function to run tests and check exit status
pytest -v -x tests/test*py

# Run single GPU tests
echo "Running single GPU tests..."
torchrun --nproc-per-node=1 -m pytest -v -x tests/distributed/test_image_io.py

# Run parallel state tests with 4 GPUs
echo "Running parallel state tests with 4 GPUs..."
torchrun --nproc-per-node=4 -m pytest -v -x tests/distributed/test_parallel_state.py

echo "Running parallel state tests with 4 GPUs..."
torchrun --nproc-per-node=4 -m pytest -v -x tests/distributed/test_distributed_greedy.py

# Run ring sampler tests with different GPU configurations
echo "Running ring sampler tests with multiple GPU configurations..."
for n in 2 3 4 7 8; do
    echo "Testing with $n GPUs..."
    torchrun --nproc-per-node=$n -m pytest -v -x tests/distributed/test_ring_sampler.py
done

echo "All tests completed successfully!"
