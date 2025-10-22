#!/bin/bash
set -e
set -x

# Parse command line arguments
MODE="default"
if [[ "$1" == "--all" ]]; then
    MODE="all"
elif [[ "$1" == "--distributed-only" ]]; then
    MODE="distributed"
fi

# Function to run basic tests
run_basic_tests() {
    echo "Running basic tests..."
    pytest -v -x tests/test*py
}

# Function to run distributed tests
run_distributed_tests() {
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
}

# Run tests based on mode
if [[ "$MODE" == "all" ]]; then
    run_basic_tests
    run_distributed_tests
    echo "All tests completed successfully!"
elif [[ "$MODE" == "distributed" ]]; then
    run_distributed_tests
    echo "Distributed tests completed successfully!"
else
    run_basic_tests
    echo ""
    echo "Basic tests completed successfully!"
    echo ""
    echo "Note: Distributed tests were not run."
    echo "Usage:"
    echo "  ./run_tests.sh                 - Run basic tests only (default)"
    echo "  ./run_tests.sh --distributed-only - Run distributed tests only"
    echo "  ./run_tests.sh --all           - Run all tests (basic + distributed)"
    echo ""
fi
