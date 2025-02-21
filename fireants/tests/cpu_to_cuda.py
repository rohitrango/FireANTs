import torch
import time
import numpy as np

def benchmark_transfer(size, use_pin_memory, non_blocking, direction="cpu_to_cuda", runs=5):
    """Benchmark CPU to CUDA or CUDA to CPU transfer speeds and compute average over multiple runs."""
    times = []
    
    for _ in range(runs):
        # Create a tensor on CPU
        tensor_cpu = torch.randn(size, dtype=torch.float32)

        # Use pinned memory if specified
        if use_pin_memory:
            tensor_cpu = tensor_cpu.pin_memory()

        # Warm-up CUDA to avoid first-time transfer overhead
        _ = torch.randn(1).cuda()
        torch.cuda.synchronize()

        # Measure transfer time
        if direction == "cpu_to_cuda":
            start_time = time.time()
            tensor_cuda = tensor_cpu.to("cuda", non_blocking=non_blocking)
            torch.cuda.synchronize()
            end_time = time.time()
        
        elif direction == "cuda_to_cpu":
            tensor_cuda = tensor_cpu.to("cuda")  # Move it to CUDA first
            torch.cuda.synchronize()
            start_time = time.time()
            tensor_cpu_out = tensor_cuda.to("cpu", non_blocking=non_blocking)
            torch.cuda.synchronize()
            end_time = time.time()
        
        else:
            raise ValueError("Invalid direction. Use 'cpu_to_cuda' or 'cuda_to_cpu'.")

        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    return np.mean(times), np.std(times)  # Return mean and standard deviation

# Tensor size (512³)
size = (512, 512, 512)
runs = 5  # Number of runs per config

# Configurations to test
configs = [
    (False, False),  # Non-pinned, blocking
    (False, True),   # Non-pinned, non-blocking
    (True, False),   # Pinned, blocking
    (True, True),    # Pinned, non-blocking
]

# Run benchmarks
print(f"{'Pinned':<10}{'Non-Blocking':<15}{'CPU->CUDA (ms)':<20}{'CUDA->CPU (ms)':<20}")
print("="*65)

for use_pin_memory, non_blocking in configs:
    cpu_to_cuda_avg, cpu_to_cuda_std = benchmark_transfer(size, use_pin_memory, non_blocking, "cpu_to_cuda", runs)
    cuda_to_cpu_avg, cuda_to_cpu_std = benchmark_transfer(size, use_pin_memory, non_blocking, "cuda_to_cpu", runs)
    print(f"{str(use_pin_memory):<10}{str(non_blocking):<15}{cpu_to_cuda_avg:<10.2f} ± {cpu_to_cuda_std:<8.2f} {cuda_to_cpu_avg:<10.2f} ± {cuda_to_cpu_std:<8.2f}")


