import torch
from fireants.registration.distributed.parallel_state import (
    initialize_parallel_state,
    cleanup_parallel_state,
    all_reduce_across_dp_ranks,
    get_parallel_state,
)

def main():
    # Initialize a 2x2 mesh (data_parallel_size=2, grid_parallel_size=2)
    initialize_parallel_state(grid_parallel_size=2, data_parallel_size=2)
    
    # Get the parallel state to know our rank
    parallel_state = get_parallel_state()
    rank = parallel_state.get_rank()
    dp_rank = parallel_state.data_parallel_rank
    gp_rank = parallel_state.grid_parallel_rank
    
    # Create a test tensor with value based on rank
    tensor = torch.tensor([float(rank)], device=parallel_state.device)
    print(f"Rank {rank} (DP={dp_rank}, GP={gp_rank}) before all_reduce: {tensor.item()}")
    
    # Perform all-reduce across data parallel ranks (SUM operation)
    all_reduce_across_dp_ranks(tensor, op=torch.distributed.ReduceOp.SUM)
    
    # Print result after reduction
    print(f"Rank {rank} (DP={dp_rank}, GP={gp_rank}) after all_reduce: {tensor.item()}")
    
    # Cleanup
    cleanup_parallel_state()

if __name__ == "__main__":
    main()
