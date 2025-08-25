import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from fireants.registration.distributed.ringsampler import distributed_grid_sampler_3d
from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d
from fireants.registration.distributed import parallel_state as ps
import pytest
import os
from typing import Tuple, Optional
import logging
import scipy

logger = logging.getLogger(__name__)

def create_random_noise_3d(depth: int, height: int, width: int, channels: int = 1, rng_seed: int = 42) -> np.ndarray:
    """Create random noise and upsample to target size using bicubic interpolation."""
    # Base size for the random noise
    base_size = 8
    
    def create_single_channel(seed_offset: int = 0):
        # Create small random noise
        rng = np.random.default_rng(rng_seed + seed_offset)
        small_noise = rng.random((base_size, base_size, base_size))    # Range [-1, 1]
        
        # Convert to torch tensor for upsampling
        small_tensor = torch.from_numpy(small_noise).float()
        small_tensor = small_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Calculate scale factors
        z_scale = depth / base_size
        y_scale = height / base_size
        x_scale = width / base_size
        
        # Upsample using trilinear interpolation
        upsampled = F.interpolate(
            small_tensor, 
            size=(depth, height, width),
            mode='trilinear',
            align_corners=True
        )
        
        return upsampled.squeeze().numpy()
    
    # Generate noise for each channel
    if channels == 1:
        return create_single_channel()
    else:
        noise_channels = [create_single_channel(_) for _ in range(channels)]
        return np.stack(noise_channels, axis=-1)

def test_ring_sampler_3d():
    # Skip if not running with torchrun
    if not ps.launched_with_torchrun():
        pytest.skip("This test requires running with torchrun")
    
    # Initialize parallel state
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ps.initialize_parallel_state(grid_parallel_size=world_size)  # Using 2 for grid parallel size
    
    # Get rank info from parallel state
    rank = ps.get_parallel_state().get_rank()
    grid_parallel_size = ps.get_grid_parallel_size()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create perlin noise image
    img_size = 128
    device = ps.get_device()  # Get device from parallel state
    
    # Create random noise image
    random_img = create_random_noise_3d(img_size, img_size, img_size)
    random_img = torch.from_numpy(random_img).float().unsqueeze(0).unsqueeze(0) * 0 + 1
    random_img = random_img.to(device)

    # Create displacement field
    grid_shape = (1, 196, 160, 224, 3)
    displacement = create_random_noise_3d(grid_shape[1], grid_shape[2], grid_shape[3], channels=3)
    displacement = torch.from_numpy(displacement).float()
    displacement = displacement.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, 196, 160, 224]
    displacement = displacement.permute(0, 2, 3, 4, 1)  # [1, 196, 160, 224, 3]
    displacement = displacement.contiguous().to(device)
    displacement = displacement * 0.1
    print(f"displacement min: {displacement.min()}, max: {displacement.max()}, rank: {rank}")

    # create PSD affine matrix
    rng = np.random.default_rng(411)
    affine = rng.random((3, 3))
    affine /= np.linalg.norm(affine)
    affine = scipy.linalg.expm(affine)
    # tr = rng.random((3, 1)) * 0.1 - 0.05
    tr = np.zeros((3, 1))
    print(f"affine = {affine}, tr = {tr}, rank = {rank}")
    affine = np.concatenate([affine, tr], axis=1)  # (3, 4)
    affine_tensor = torch.eye(4, 4, device=device)
    affine_tensor[:3, :] = torch.from_numpy(affine)
    affine_tensor = affine_tensor.unsqueeze(0).contiguous()  # (1, 4, 4)
    print(f"affine tensor det =  {torch.linalg.det(affine_tensor[0])}, shape = {affine_tensor.shape}, rank = {rank}")
    # affine_tensor = None
    
    # Run baseline fused grid sample
    baseline_output = fused_grid_sampler_3d(
        random_img.contiguous(),
        affine=affine_tensor[:, :3, :].contiguous() if affine_tensor is not None else None,
        grid=displacement.contiguous(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        is_displacement=True
    )
    
    # Shard the image along spatial dim 0 (Z)
    shard_size = (img_size + grid_parallel_size - 1) // grid_parallel_size
    img_start = rank * shard_size
    img_end = min(img_start + shard_size, img_size)

    img_shard = random_img[:, :, img_start:img_end, :, :]
    img_min_coords = (-1.0, -1.0, -1.0)  # align_corners=True
    img_max_coords = (1.0, 1.0, 1.0)     # align_corners=True
    # Adjust Z coordinates based on shard position
    z_min = -1.0 + 2.0 * (img_start / (img_size - 1.0))
    z_max = -1.0 + 2.0 * ((img_end - 1.0) / (img_size - 1.0))
    img_min_coords = (-1.0, -1.0, z_min)
    img_max_coords = (1.0, 1.0, z_max)
    
    # Convert to tensors on correct device
    img_min_coords = torch.tensor(img_min_coords, device=device)
    img_max_coords = torch.tensor(img_max_coords, device=device)
    
    # Shard the grid along spatial dim 1 (Z)
    grid_shard_size = (grid_shape[2] + grid_parallel_size - 1) // grid_parallel_size
    grid_start = rank * grid_shard_size
    grid_end = min(grid_start + grid_shard_size, grid_shape[2])

    print(f"img_start: {img_start}, img_end: {img_end}, rank: {rank}, shard_size: {shard_size}, img_size: {img_size}")
    print(f"grid_start: {grid_start}, grid_end: {grid_end}, rank: {rank}, grid_shard_size: {grid_shard_size}, grid_shape: {grid_shape}")
    torch.distributed.barrier()
    if rank == 0:
        print()
    
    grid_shard = displacement[:, :, grid_start:grid_end, :, :]
    y_min = -1.0 + 2.0 * (grid_start / (grid_shape[2] - 1.0))
    y_max = -1.0 + 2.0 * ((grid_end - 1.0) / (grid_shape[2] - 1.0))
    grid_min_coords = (-1.0, y_min, -1.0)  # align_corners=True
    grid_max_coords = (1.0, y_max, 1.0)     # align_corners=True

    grid_min_coords = torch.tensor(grid_min_coords, device=device)
    grid_max_coords = torch.tensor(grid_max_coords, device=device)

    
    for i in range(grid_parallel_size):
        if i == rank:
            logger.info(f"img_shard shape: {img_shard.shape}, rank={rank}")
            logger.info(f"grid_shard shape: {grid_shard.shape}, rank={rank}")
            logger.info(f"img_min_coords shape: {img_min_coords}, rank={rank}")
            logger.info(f"img_max_coords shape: {img_max_coords}, rank={rank}")
            logger.info(f"grid_min_coords shape: {grid_min_coords}, rank={rank}")
            logger.info(f"grid_max_coords shape: {grid_max_coords}, rank={rank}")
        torch.distributed.barrier()
        print()

    
    # Run distributed grid sampler
    distributed_output = distributed_grid_sampler_3d(
        img_shard.contiguous(),
        img_min_coords,
        img_max_coords,
        affine_tensor,
        grid_shard.contiguous(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        min_coords=grid_min_coords,
        max_coords=grid_max_coords,
        is_displacement=True
    )

    logger.info(f"distributed_output shape: {distributed_output.shape}")
    
    # Compare with corresponding shard of baseline output
    baseline_shard = baseline_output[:, :, :, grid_start:grid_end, :]
    assert baseline_shard.shape == distributed_output.shape, f"Baseline shard shape {baseline_shard.shape} does not match distributed output shape {distributed_output.shape}"

    print(f"baseline_shard shape: {baseline_shard.shape}")
    print(f"baseline min: {baseline_shard.min()}, max: {baseline_shard.max()}, rank: {rank}")
    print(f"distributed_output min: {distributed_output.min()}, max: {distributed_output.max()}, rank: {rank}")
    print()
    
    # Compute relative error
    # error = torch.norm(distributed_output - baseline_shard) 
    error = torch.abs(distributed_output - baseline_shard).mean()
    rel_error = (torch.abs(distributed_output - baseline_shard) / (1e-2 + torch.abs(baseline_shard))).mean()
    # Use all_reduce to get max error across ranks
    max_error = torch.tensor([error], device=device)
    max_rel_error = torch.tensor([rel_error], device=device)
    print(f"rank: {rank}, max_error: {max_error}, max_rel_error: {max_rel_error}")
    ps.all_reduce_across_gp_ranks(max_error, op=dist.ReduceOp.AVG)
    ps.all_reduce_across_gp_ranks(max_rel_error, op=dist.ReduceOp.AVG)

    if rank == 0:
        print(f"Maximum relative error across ranks: {max_error.item()}")
        assert max_error.item() < 1e-5, f"Relative error {max_error.item()} is too large"
        assert max_rel_error.item() < 1e-3, f"Relative error {max_rel_error.item()} is too large"
    
    # Cleanup parallel state
    ps.cleanup_parallel_state()


if __name__ == "__main__":
    test_ring_sampler_3d()