# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


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
