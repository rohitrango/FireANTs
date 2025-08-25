

# script contains the parallel state for the registration pipeline
# currently, FireANTs supports two forms of parallelization:
#  1. Grid Parallel (for distributing the same image across multiple GPUs)
#  2. Data Parallel (for distributing different images across multiple GPUs)
# (1) is typically used for distributed greedy registration, where the same image is distributed across multiple GPUs
# (2) is typically used for template building and data-parallel registration for large datasets

# TODO (low priority): can add virtual tp to the device mesh to support offloaded registration (for gpu-poor scenarios)

import torch
import os
from typing import Optional, Union, List
import logging
from logging import getLogger
import numpy as np
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from time import sleep

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

_PLATFORM_NAME = os.name
_PARALLEL_STATE = None   # probably a device mesh
_BACKEND = None

class FireANTsDeviceMesh:
    def __init__(self, mesh: np.ndarray, backend: str):
        # mesh is a nD array
        self.mesh = mesh
        self.backend = backend
        # sizes
        self.world_size = mesh.size
        self.num_devices = torch.cuda.device_count() 
        self.data_parallel_size = mesh.shape[0]
        self.grid_parallel_size = mesh.shape[1]
        self.current_rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank % self.num_devices}')
        # get current data_parallel_rank and grid_parallel_rank
        dp_rank, gp_rank = np.where(mesh == self.current_rank)
        assert len(dp_rank) == 1 and len(gp_rank) == 1, f"Rank {self.current_rank} is in the mesh {len(dp_rank)} times"
        dp_rank, gp_rank = dp_rank[0], gp_rank[0]
        self.data_parallel_rank = dp_rank
        self.grid_parallel_rank = gp_rank

        # get device mesh
        assert self.current_rank in self.mesh[:, self.grid_parallel_rank], f"Rank {self.current_rank} is not in the grid parallel group"
        assert self.current_rank in self.mesh[self.data_parallel_rank, :], f"Rank {self.current_rank} is not in the data parallel group"
        # create new groups
        self.gp_pg_groups = {}
        self.dp_pg_groups = {}
        print("data parallel size", self.data_parallel_size)
        print("grid parallel size", self.grid_parallel_size)
        for row in range(self.data_parallel_size):
            ranks = list(self.mesh[row, :])
            group = torch.distributed.new_group(ranks=ranks)
            self.gp_pg_groups[tuple(ranks)] = group
            if self.current_rank in ranks:
                print(f"Rank {self.current_rank} is in the grid parallel group {ranks}")
                self.gp_group_obj = group

        for col in range(self.grid_parallel_size):
            ranks = list(self.mesh[:, col])
            group = torch.distributed.new_group(ranks=ranks)
            self.dp_pg_groups[tuple(ranks)] = group
            if self.current_rank in ranks:
                print(f"Rank {self.current_rank} is in the data parallel group {ranks}")
                self.dp_group_obj = group

        torch.distributed.barrier()
    
    def get_rank(self):
        '''
        get the current rank
        '''
        return self.current_rank
    
    def get_current_dp_group(self):
        '''
        get all the data parallel ranks of the current process
        '''
        return list(self.mesh[:, self.grid_parallel_rank])
    
    def get_current_gp_group(self):
        '''
        get all the grid parallel ranks of the current process
        '''
        return list(self.mesh[self.data_parallel_rank, :])
    
    def get_previous_gp_rank(self):
        '''
        get the previous grid parallel rank
        '''
        if self.grid_parallel_rank == 0:
            return None
        return self.mesh[self.data_parallel_rank, self.grid_parallel_rank - 1]
    
    def get_next_gp_rank(self):
        '''
        get the next grid parallel rank
        '''
        if self.grid_parallel_rank == self.grid_parallel_size - 1:
            return None
        return self.mesh[self.data_parallel_rank, self.grid_parallel_rank + 1]

def ring_collect_op(send_tensors: List[torch.Tensor], recv_tensors: List[torch.Tensor], src: int, dst: int):
    '''
    perform a ring collect operation

    ### Note that doing send-recv in a ring with isend/irecv hangs, so we use the batch isend/irecv ops instead
    '''
    assert len(send_tensors) == len(recv_tensors), "Number of send and recv tensors must be the same"
    if len(send_tensors) == 0:
        return
    send_ops = [torch.distributed.P2POp(torch.distributed.isend, tensor, peer=dst) for tensor in send_tensors]
    recv_ops = [torch.distributed.P2POp(torch.distributed.irecv, tensor, peer=src) for tensor in recv_tensors]
    reqs = torch.distributed.batch_isend_irecv(send_ops + recv_ops)
    [r.wait() for r in reqs]


def all_reduce_across_ranks(tensor: torch.Tensor, op: torch.distributed.ReduceOp, group: Optional[torch.distributed.ProcessGroup] = None):
    '''
    perform all reduce across given ranks
    '''
    torch.distributed.all_reduce(tensor, op=op, group=group)

def all_reduce_across_dp_ranks(tensor: torch.Tensor, op: torch.distributed.ReduceOp):
    '''
    all reduce the tensor across given ranks
    '''
    for ranks, group in _PARALLEL_STATE.dp_pg_groups.items():
        # if _PARALLEL_STATE.current_rank in ranks:
        all_reduce_across_ranks(tensor, op, group)
        # torch.distributed.barrier()

def all_reduce_across_gp_ranks(tensor: torch.Tensor, op: torch.distributed.ReduceOp):
    '''
    all reduce the tensor across given ranks
    '''
    for ranks, group in _PARALLEL_STATE.gp_pg_groups.items():
        # if _PARALLEL_STATE.current_rank in ranks:
        all_reduce_across_ranks(tensor, op, group)
        # torch.distributed.barrier()

def get_parallel_state():
    '''
    get the parallel state object
    '''
    global _PARALLEL_STATE
    return _PARALLEL_STATE

def get_device():
    '''
    get the current device
    '''
    return _PARALLEL_STATE.device

def is_initialized():
    '''
    check if the parallel state is initialized
    '''
    return _PARALLEL_STATE is not None

def get_current_device():
    '''
    get the current device
    '''
    return _PARALLEL_STATE.device

def get_default_backend():
    '''
    get the default backend for the platform
    '''
    if _PLATFORM_NAME == 'nt':
        return 'gloo'
    return 'nccl'

def get_grid_parallel_size():
    return _PARALLEL_STATE.grid_parallel_size

def get_data_parallel_size():
    return _PARALLEL_STATE.data_parallel_size

def launched_with_torchrun():
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "LOCAL_RANK" in os.environ
    )

def initialize_parallel_state(
        grid_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None,
        backend: Optional[str] = None,
        wait: Optional[float] = None,
):
    '''
    '''
    assert not torch.distributed.is_initialized(), "Distributed training already initialized"
    assert launched_with_torchrun(), "Distributed training must be launched with torchrun"

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    current_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    try:
        torch.cuda.set_device(local_rank)
    except:
        logger.warning(f"Cannot set device to {local_rank}, using CPU")
    # check correctness
    if data_parallel_size is None:
        data_parallel_size = world_size // grid_parallel_size
    assert data_parallel_size * grid_parallel_size == world_size, f"Cannot divide world size={world_size} into grid={grid_parallel_size} and data={data_parallel_size} parallel sizes"
    
    global _BACKEND, _PARALLEL_STATE
    # set backend and launch
    _BACKEND = get_default_backend() if backend is None else backend
    torch.distributed.init_process_group(backend=_BACKEND)
    if wait is not None:
        sleep(wait)

    # manually setting up the ranks instead of using device mesh (for windows support later in case device mesh doesn't support all gloo functions)
    _PARALLEL_STATE = FireANTsDeviceMesh(np.arange(world_size).reshape(data_parallel_size, grid_parallel_size), _BACKEND)

def isend(tensor: torch.Tensor, dst: int, tag: int = 0, group: Optional[torch.distributed.ProcessGroup] = None) -> torch.distributed.Work:
    '''
    Asynchronously send a tensor to a destination rank.
    
    Args:
        tensor: Tensor to send (assumed to be on CUDA by default)
        dst: Destination rank
        tag: Tag to identify the message (default: 0)
        group: Process group for the send operation (default: None, uses world group)
    
    Returns:
        A distributed request object that can be used to query the status or wait for completion.
        For GLOO backend, a CPU copy of the tensor is used for sending while original tensor remains on GPU.
    '''
    # For GLOO backend, create a CPU copy for sending while keeping original on GPU
    if _BACKEND == 'gloo':
        cpu_tensor = tensor.cpu().detach().clone()
        return torch.distributed.isend(cpu_tensor, dst=dst, tag=tag, group=group)
    
    print("sending tensor to ", dst)
    
    return torch.distributed.isend(tensor, dst=dst, tag=tag, group=group)

def irecv(tensor: torch.Tensor, src: int, tag: int = 0, group: Optional[torch.distributed.ProcessGroup] = None) -> torch.distributed.Work:
    '''
    Asynchronously receive a tensor from a source rank.
    
    Args:
        tensor: Empty tensor to receive data into (assumed to be on CUDA by default)
        src: Source rank
        tag: Tag to identify the message (default: 0)
        group: Process group for the receive operation (default: None, uses world group)
    
    Returns:
        A distributed request object that can be used to query the status or wait for completion.
        Note: For GLOO backend, the received tensor will be on CPU and needs to be moved back to CUDA after wait().
    '''
    original_device = tensor.device
    # For GLOO backend, we need to receive on CPU
    if _BACKEND == 'gloo':
        tensor = tensor.cpu()
        
    req = torch.distributed.irecv(tensor, src=src, tag=tag, group=group)
    print("received tensor from ", src)
    
    if _BACKEND == 'gloo':
        # Create a wrapper around the request that moves tensor back to original device after completion
        original_wait = req.wait
        def wait_and_move():
            original_wait()
            tensor.data = tensor.data.to(original_device)
        req.wait = wait_and_move
    
    return req


def cleanup_parallel_state(wait: Optional[float] = None):
    '''
    '''
    global _PARALLEL_STATE, _BACKEND
    torch.distributed.destroy_process_group()
    _PARALLEL_STATE = None
    _BACKEND = None
    if wait is not None:
        sleep(wait)

