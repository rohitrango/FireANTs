'''
Utilities for distributed training
'''
import numpy as np
import torch
from torch import distributed as dist
from fireants.interpolator.fused_grid_sample import get_min_coords3d, get_min_coords2d, get_max_coords3d, get_max_coords2d
import subprocess
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ['get_dim_to_shard', 'gather_and_concat', 'add_distributed_padding', 'calculate_bbox_from_gather_stats']

def get_dim_to_shard(dims, world_size, fixed_shape, moving_shape):
    '''
    dims: number of spatial dims
    world_size: number of processes
    fixed_shape: shape of fixed image
    moving_shape: shape of moving image
    '''
    fixed_shape = list(fixed_shape)[2:]
    moving_shape = list(moving_shape)[2:]
    shape = [min(fixed_shape[i], moving_shape[i]) for i in range(dims)]
    return np.argmax(shape)

def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    free_mem = [int(x) for x in result.stdout.strip().split('\n')]
    return free_mem  # list of free memory per GPU in MiB

def should_shard_problem(base_shape, scale, offload):
    working_mem_mult = 20 if not offload else 14
    dims = len(base_shape) - 2
    working_mem_required = np.prod(base_shape) * 4 * working_mem_mult / 1024**2 / (scale*1.0)**dims  # in MB
    # get remaining memory torch
    remaining_mem = get_gpu_memory()[0]
    return working_mem_required > remaining_mem

def async_send_tensor_ack(tensor, otherrank, myrank, gather_stats_only):
    shape = torch.tensor(tensor.shape, device=tensor.device).long()
    req = dist.isend(shape, dst=otherrank, tag=myrank)
    req.wait()
    # send the tensor now if given
    if gather_stats_only:
        return None
    else:
        req = dist.isend(tensor, dst=otherrank, tag=myrank * 10000)
        return req

def async_recv_tensor_ack(ref_tensor, otherrank, myrank, gather_stats_only):
    # receive the shape first
    # use reference tensor for shape, dtype, etc
    sz = torch.zeros(len(ref_tensor.shape), device=torch.cuda.current_device()).long()
    req = dist.irecv(sz, src=otherrank, tag=otherrank)
    req.wait()
    if gather_stats_only:
        return None, None, sz.tolist()
    else: 
        # receive the tensor now
        tensor = torch.empty(sz.tolist(), dtype=ref_tensor.dtype, device=ref_tensor.device)
        req = dist.irecv(tensor, src=otherrank, tag=otherrank * 100)
        return tensor, req, sz.tolist()

def refactor_grid_to_image_stats(stats):
    ''' 
    given stats containing grid shapes, refactor into image shapes
    '''
    stats['dim_to_shard'] += 1  # we did a -1 during gather and concat
    for k in stats:
        if k == 'dim_to_shard':
            continue
        v = stats[k][:-1]       # remove last channel dimension
        v = [v[0], 1] + list(v[1:])   # insert dummy channel dimension
        stats[k] = list(v)
    return stats

def gather_and_concat(tensor, rank, world_size, master_rank, is_state_sharded, dim_to_shard, gather_stats_only=False):
    '''
    utility function to gather subtensors from different processes and then concatenate them along "dim_to_shard"
    if "is_state_sharded" is False, we only need to gather them up to rank 0
    '''
    stats = {}
    stats['dim_to_shard'] = dim_to_shard
    if is_state_sharded:
        # keep track of shape
        stats[rank] = list(tensor.shape)
        # everyone needs to have all copies (this is useful to keep a copy of the moving image in all gpus for example)
        # send shape and tensor to all other ranks
        tensors = [None for _ in range(world_size)]
        tensors[rank] = tensor
        reqs = []

        for poll_rank in range(world_size):
            # send info to all other ranks
            if poll_rank == rank:
                send_reqs = []
                for nbrrank in range(world_size):
                    if nbrrank == rank:
                        continue
                    # returns none if gather_stats_only is True (we have sent tensor size)
                    req = async_send_tensor_ack(tensor, nbrrank, rank, gather_stats_only) 
                    if req is not None:
                        send_reqs.append(req)
                [r.wait() for r in send_reqs]
            # receive from poll_rank
            else: 
                recv_tensor, req, recv_shape = async_recv_tensor_ack(tensor, poll_rank, rank, gather_stats_only)
                tensors[poll_rank] = recv_tensor
                stats[poll_rank] = recv_shape
                if req is not None:
                    reqs.append(req)

        [r.wait() for r in reqs]

        # if gather_stats_only, we only need to return the stats, otherwise return the concatenated tensor too
        if not gather_stats_only:
            return torch.cat(tensors, dim=dim_to_shard+2), stats
        else:
            return tensor, stats
    else:
        # gather all of them to rank 0
        raise NotImplementedError("TODO: Implement this function")

def crop_distributed_padding(tensor, rank, world_size, image_padding, dim_to_shard):
    ''' undo the effects of add_distributed_padding '''
    shape = list(tensor.shape)
    # if rank = 0, didnt get any padding from previous slice
    # if rank = world_size - 1, didnt get any padding from next slice
    start_crop_idx = 0 if rank == 0 else image_padding 
    end_crop_idx = shape[dim_to_shard+2] if rank == world_size - 1 else shape[dim_to_shard+2] - image_padding
    return tensor.narrow_copy(dim_to_shard+2, start_crop_idx, end_crop_idx-start_crop_idx).contiguous()


def add_distributed_padding(tensor, rank, world_size, image_padding, dim_to_shard):
    '''
    utility to add some padding to the tensor depending on its rank

    protocol: 
        - receive from previous slice, send to next slice
        - receive from next slice, send to previous slice
    '''
    if image_padding <= 0:
        return tensor
    # get target shape to pad
    tgt_shape = list(tensor.shape)
    tgt_shape[dim_to_shard+2] = image_padding
    tgt_shape = tuple(tgt_shape)

    # receive slices
    reqs = []
    slice_before, slice_after = [], []

    # receive from previous 
    if (rank-1) >= 0:
        slice_before = [torch.empty(tgt_shape, dtype=tensor.dtype, device=tensor.device)]
        req = dist.irecv(slice_before[0], src=rank-1)
        reqs.append(req)

    # send to next
    if (rank+1) < world_size:
        req = dist.isend(tensor.narrow_copy(dim_to_shard+2, -image_padding, image_padding), dst=rank+1)
        reqs.append(req)
    
    [r.wait() for r in reqs[::-1]]
    reqs = []
    
    # receive from next
    if (rank+1) < world_size:
        slice_after = [torch.empty(tgt_shape, dtype=tensor.dtype, device=tensor.device)]
        req = dist.irecv(slice_after[0], src=rank+1)
        reqs.append(req)
    
    # send to previous
    if (rank-1) >= 0:
        req = dist.isend(tensor.narrow_copy(dim_to_shard+2, 0, image_padding), dst=rank-1)
        reqs.append(req)

    [r.wait() for r in reqs[::-1]]
    tensor = torch.cat(slice_before + [tensor] + slice_after, dim=dim_to_shard+2)
    return tensor

def calculate_bbox_from_gather_stats(moving_gather_stats, rank, world_size, dims, align_corners=True):
    ''' given the results of gather stats and current rank, worldsize, and dims, return the coordinates
    as per the result of align_corners'''
    # store size to be used later
    sz = moving_gather_stats[rank][2:]
    dim2shard = moving_gather_stats['dim_to_shard']
    # get total size
    total_size = 0
    start_size = 0
    end_size = -1   # to make it inclusive
    for i in range(world_size):
        total_size += moving_gather_stats[i][dim2shard+2]
        if i < rank:
            start_size += moving_gather_stats[i][dim2shard+2]
        if i <= rank:
            end_size += moving_gather_stats[i][dim2shard+2]
    # get actual total size of sharded dim
    sz[dim2shard] = total_size
    # get min and max coords in (z, y, x) format
    mincoords = (get_min_coords3d if dims == 3 else get_min_coords2d)(*sz, align_corners=align_corners)[::-1]
    maxcoords = (get_max_coords3d if dims == 3 else get_max_coords2d)(*sz, align_corners=align_corners)[::-1]
    mincoords, maxcoords = list(mincoords), list(maxcoords)
    # change the dim based on dim_to_shard
    if align_corners:
        mincoords[dim2shard] = 2*start_size/(total_size-1) - 1
        maxcoords[dim2shard] = 2*end_size/(total_size-1) - 1
    else:
        mincoords[dim2shard] = (2*start_size+1)/total_size - 1
        maxcoords[dim2shard] = (2*end_size+1)/total_size - 1
    return mincoords[::-1], maxcoords[::-1]
    