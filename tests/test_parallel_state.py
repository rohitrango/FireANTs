import os
import pytest
import torch
import numpy as np
from fireants.registration.distributed import parallel_state
from fireants.registration.distributed.parallel_state import (
    initialize_parallel_state,
    cleanup_parallel_state,
    all_reduce_across_dp_ranks,
    all_reduce_across_gp_ranks,
    get_parallel_state,
    get_device,
    get_grid_parallel_size,
    get_data_parallel_size,
)


@pytest.fixture(scope="module")
def distributed_env():
    """Setup distributed environment variables for testing. Using module scope to initialize only once."""
    # Save original environment
    orig_env = {}
    for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
        orig_env[key] = os.environ.get(key)
    
    # Set master address and port for NCCL
    yield
    
    # Restore original environment
    for key, value in orig_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

@pytest.fixture(scope="module")
def parallel_setup(distributed_env):
    """Initialize parallel state for testing. Using module scope to initialize only once."""
    initialize_parallel_state(grid_parallel_size=2, data_parallel_size=2, backend='nccl', wait=2)
    yield
    cleanup_parallel_state(wait=4)

@pytest.mark.distributed
def test_initialization(distributed_env):
    """Test parallel state initialization."""
    initialize_parallel_state(grid_parallel_size=2, data_parallel_size=2)
    
    state = get_parallel_state()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    assert state is not None
    assert state.world_size == 4
    assert state.grid_parallel_size == 2
    assert state.data_parallel_size == 2
    assert state.current_rank == rank
    assert state.local_rank == local_rank
    assert state.device == torch.device(f'cuda:{local_rank}')
    
    # Verify mesh positions
    dp_rank = rank // 2  # 0,0,1,1
    gp_rank = rank % 2   # 0,1,0,1
    assert state.data_parallel_rank == dp_rank
    assert state.grid_parallel_rank == gp_rank
    
    cleanup_parallel_state()

@pytest.mark.distributed
def test_dp_reduction(parallel_setup):
    """Test data parallel reduction."""
    # Create tensor with rank-specific values
    rank = get_parallel_state().current_rank
    x = torch.tensor([float(rank)], device=get_device())
    print(f"x: {x}, rank: {rank}, device: {get_device()}, dprank: {get_parallel_state().data_parallel_rank}, gprank: {get_parallel_state().grid_parallel_rank}")

    dp_group = get_parallel_state().get_current_dp_group()
    print(f"dp_group: {dp_group}, rank: {rank}")
    assert rank in dp_group
    if rank == 0 or rank == 2:
        assert dp_group == [0, 2]
    else:
        assert dp_group == [1, 3]
    
    # Perform reduction
    all_reduce_across_dp_ranks(x, torch.distributed.ReduceOp.SUM)
    
    # For 2x2 mesh, each DP group should sum to either 0+2=2 or 1+3=4
    gp_rank = get_parallel_state().grid_parallel_rank
    expected_sum = 2.0 if gp_rank == 0 else 4.0
    torch.cuda.synchronize()
    torch.distributed.barrier()
    assert torch.allclose(x, torch.tensor([expected_sum], device=get_device())), f"x: {x}, expected_sum: {expected_sum}, rank: {rank}, gp_rank: {gp_rank}, dp_rank: {get_parallel_state().data_parallel_rank}"

@pytest.mark.distributed
def test_gp_reduction(parallel_setup):
    """Test grid parallel reduction."""
    # Create tensor with rank-specific values
    rank = get_parallel_state().current_rank
    x = torch.tensor([float(rank)], device=get_device())
    print(f"x: {x}, rank: {rank}, device: {get_device()}, dprank: {get_parallel_state().data_parallel_rank}, gprank: {get_parallel_state().grid_parallel_rank}")
    
    # Perform reduction
    all_reduce_across_gp_ranks(x, torch.distributed.ReduceOp.SUM)
    
    # For 2x2 mesh, each GP group should sum to either 0+1=1 or 2+3=5
    dp_rank = get_parallel_state().data_parallel_rank
    expected_sum = 1.0 if dp_rank == 0 else 5.0
    torch.cuda.synchronize()
    torch.distributed.barrier()
    assert torch.allclose(x, torch.tensor([expected_sum], device=get_device())), f"x: {x}, expected_sum: {expected_sum}, rank: {rank}"

@pytest.mark.distributed
def test_device_mesh_properties(parallel_setup):
    """Test device mesh properties and group formation."""
    state = get_parallel_state()
    rank = state.current_rank
    
    # Test mesh shape
    assert state.mesh.shape == (2, 2)  # 2x2 mesh
    
    # Test DP group formation
    dp_group = state.get_current_dp_group()
    gp_rank = state.grid_parallel_rank
    expected_dp_group = [0, 2] if gp_rank == 0 else [1, 3]
    assert dp_group == expected_dp_group
    
    # Test GP group formation
    gp_group = state.get_current_gp_group()
    dp_rank = state.data_parallel_rank
    expected_gp_group = [0, 1] if dp_rank == 0 else [2, 3]
    assert gp_group == expected_gp_group
    
    # Test next/previous GP ranks
    if gp_rank == 0:
        assert state.get_previous_gp_rank() is None
        assert state.get_next_gp_rank() == rank + 1
    elif gp_rank == 1:
        assert state.get_previous_gp_rank() == rank - 1
        assert state.get_next_gp_rank() is None

@pytest.mark.distributed
def test_parallel_sizes(parallel_setup):
    """Test parallel size getters."""
    assert get_grid_parallel_size() == 2
    assert get_data_parallel_size() == 2

@pytest.mark.distributed
def test_send_receive(parallel_setup):
    """Test send/receive operations between ranks."""
    state = get_parallel_state()
    rank = state.current_rank
    
    # Test sending between adjacent GP ranks within each DP group
    send_tensor = torch.tensor([float(rank)], device=get_device())
    
    # Get next/previous ranks in GP group
    next_rank = state.get_next_gp_rank()
    prev_rank = state.get_previous_gp_rank()
    
    # Send to next rank if we're not the last in GP group
    if next_rank is not None:
        send_req = parallel_state.isend(send_tensor, dst=next_rank)
        send_req.wait()
    
    # Receive from previous rank if we're not the first in GP group
    if prev_rank is not None:
        recv_tensor = torch.zeros_like(send_tensor)
        recv_req = parallel_state.irecv(recv_tensor, src=prev_rank)
        recv_req.wait()
        
        # Verify received value
        expected = torch.tensor([float(prev_rank)], device=get_device())
        assert torch.allclose(recv_tensor, expected)
