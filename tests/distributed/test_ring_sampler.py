import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from time import time
from fireants.registration.distributed.ring_sampler import distributed_grid_sampler_3d_fwd, fireants_ringsampler_interpolator
from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d
from fireants.registration.distributed import parallel_state as ps
import pytest
import os
from typing import Tuple, Optional, Dict, Any
import logging
import scipy
from dataclasses import dataclass

logger = logging.getLogger(__name__)

### Run this test with `torchrun --nproc_per_node=2/3/4/7/8 test_ring_sampler.py`

@dataclass
class RingSamplerTestEnv:
    """Test environment for ring sampler tests"""
    rank: int
    grid_parallel_size: int
    device: torch.device
    gp_group: Any

@pytest.fixture(scope="module")
def ring_sampler_env():
    """
    Fixture to set up the distributed environment for ring sampler tests.
    Handles initialization and cleanup of parallel state.
    
    Returns:
        RingSamplerTestEnv: Test environment object containing rank, world size, etc.
    """
    # Skip if not running with torchrun
    if not ps.launched_with_torchrun():
        pytest.skip("This test requires running with torchrun")
    
    # Initialize parallel state
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ps.initialize_parallel_state(grid_parallel_size=world_size)
    
    # Get rank info and device from parallel state
    rank = ps.get_parallel_state().get_rank()
    grid_parallel_size = ps.get_grid_parallel_size()
    device = ps.get_device()
    gp_group = ps.get_parallel_state().get_current_gp_group()
    local_rank = gp_group.index(rank)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create and return test environment
    env = RingSamplerTestEnv(
        rank=local_rank,
        grid_parallel_size=grid_parallel_size,
        device=device,
        gp_group=gp_group,
    )
    
    yield env
    
    # Cleanup parallel state after test
    ps.cleanup_parallel_state()

def create_random_noise_3d(depth: int, height: int, width: int, channels: int = 1, rng_seed: int = 42) -> np.ndarray:
    """Create random noise and upsample to target size using bicubic interpolation."""
    # Base size for the random noise
    base_size = 8
    
    def create_single_channel(seed_offset: int = 0):
        # Create small random noise
        rng = np.random.default_rng(rng_seed + seed_offset)
        small_noise = rng.random((base_size, base_size, base_size)) - 0.5   # Range [-1, 1]
        small_noise = small_noise * 2
        
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

def test_ring_sampler_3d(ring_sampler_env: RingSamplerTestEnv):
    """Test forward pass of ring sampler"""
    # Create perlin noise image
    img_size = 128
    device = ring_sampler_env.device
    
    # Create random noise image
    random_img = create_random_noise_3d(img_size, img_size, img_size)
    random_img = torch.from_numpy(random_img).float().unsqueeze(0).unsqueeze(0) * 0 + 1
    random_img = random_img.to(ring_sampler_env.device)

    # Create displacement field
    grid_shape = (1, 196, 160, 224, 3)
    displacement = create_random_noise_3d(grid_shape[1], grid_shape[2], grid_shape[3], channels=3) 
    displacement = torch.from_numpy(displacement).float()
    displacement = displacement.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, 196, 160, 224]
    displacement = displacement.permute(0, 2, 3, 4, 1)  # [1, 196, 160, 224, 3]
    displacement = displacement.contiguous().to(ring_sampler_env.device)
    displacement = displacement * 0.1
    logger.info(f"displacement min: {displacement.min()}, max: {displacement.max()}, rank: {ring_sampler_env.rank}")

    # create PSD affine matrix
    rng = np.random.default_rng(411)
    affine = rng.random((3, 3))
    affine /= np.linalg.norm(affine)
    affine = scipy.linalg.expm(affine)
    # tr = rng.random((3, 1)) * 0.1 - 0.05
    tr = np.zeros((3, 1))
    affine = np.concatenate([affine, tr], axis=1)  # (3, 4)
    affine_tensor = torch.eye(4, 4, device=ring_sampler_env.device)
    affine_tensor[:3, :] = torch.from_numpy(affine)
    affine_tensor = affine_tensor.unsqueeze(0).contiguous()  # (1, 4, 4)
    
    # Reset memory stats and measure baseline implementation
    torch.cuda.reset_peak_memory_stats()
    start = time()
    
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
    torch.cuda.synchronize()
    baseline_time = time() - start
    baseline_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Shard the image along spatial dim 0 (Z)
    shard_size = (img_size + ring_sampler_env.grid_parallel_size - 1) // ring_sampler_env.grid_parallel_size
    img_start = ring_sampler_env.rank * shard_size
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
    img_min_coords = torch.tensor(img_min_coords, device=ring_sampler_env.device)
    img_max_coords = torch.tensor(img_max_coords, device=ring_sampler_env.device)
    
    # Shard the grid along spatial dim 1 (Z)
    grid_shard_size = (grid_shape[2] + ring_sampler_env.grid_parallel_size - 1) // ring_sampler_env.grid_parallel_size
    grid_start = ring_sampler_env.rank * grid_shard_size
    grid_end = min(grid_start + grid_shard_size, grid_shape[2])

    logger.info(f"img_start: {img_start}, img_end: {img_end}, rank: {ring_sampler_env.rank}, shard_size: {shard_size}, img_size: {img_size}")
    logger.info(f"grid_start: {grid_start}, grid_end: {grid_end}, rank: {ring_sampler_env.rank}, grid_shard_size: {grid_shard_size}, grid_shape: {grid_shape}")
    torch.distributed.barrier()
    if ring_sampler_env.rank == 0:
        logger.info("--------------------------------")
    
    grid_shard = displacement[:, :, grid_start:grid_end, :, :]
    y_min = -1.0 + 2.0 * (grid_start / (grid_shape[2] - 1.0))
    y_max = -1.0 + 2.0 * ((grid_end - 1.0) / (grid_shape[2] - 1.0))
    grid_min_coords = (-1.0, y_min, -1.0)  # align_corners=True
    grid_max_coords = (1.0, y_max, 1.0)     # align_corners=True

    grid_min_coords = torch.tensor(grid_min_coords, device=ring_sampler_env.device)
    grid_max_coords = torch.tensor(grid_max_coords, device=ring_sampler_env.device)

    
    for i in range(ring_sampler_env.grid_parallel_size):
        if i == ring_sampler_env.rank:
            logger.info(f"img_shard shape: {img_shard.shape}, rank={ring_sampler_env.rank}")
            logger.info(f"grid_shard shape: {grid_shard.shape}, rank={ring_sampler_env.rank}")
            logger.info(f"img_min_coords shape: {img_min_coords}, rank={ring_sampler_env.rank}")
            logger.info(f"img_max_coords shape: {img_max_coords}, rank={ring_sampler_env.rank}")
            logger.info(f"grid_min_coords shape: {grid_min_coords}, rank={ring_sampler_env.rank}")
            logger.info(f"grid_max_coords shape: {grid_max_coords}, rank={ring_sampler_env.rank}")
        torch.distributed.barrier()
        logger.info("--------------------------------")

    
    # Reset memory stats and measure distributed implementation
    torch.cuda.reset_peak_memory_stats()
    start = time()
    
    # Run distributed grid sampler
    distributed_output = distributed_grid_sampler_3d_fwd(
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
    torch.cuda.synchronize()
    distributed_time = time() - start
    distributed_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    logger.info(f"distributed_output shape: {distributed_output.shape}")
    
    # Compare with corresponding shard of baseline output
    baseline_shard = baseline_output[:, :, :, grid_start:grid_end, :]
    assert baseline_shard.shape == distributed_output.shape, f"Baseline shard shape {baseline_shard.shape} does not match distributed output shape {distributed_output.shape}"

    logger.info(f"baseline_shard shape: {baseline_shard.shape}")
    logger.info(f"baseline min: {baseline_shard.min()}, max: {baseline_shard.max()}, rank: {ring_sampler_env.rank}")
    logger.info(f"distributed_output min: {distributed_output.min()}, max: {distributed_output.max()}, rank: {ring_sampler_env.rank}")
    logger.info("--------------------------------")
    
    # Compute relative error
    # error = torch.norm(distributed_output - baseline_shard) 
    error = torch.abs(distributed_output - baseline_shard).mean()
    rel_error = (torch.abs(distributed_output - baseline_shard) / (1e-2 + torch.abs(baseline_shard))).mean()
    # Use all_reduce to get max error across ranks
    max_error = torch.tensor([error], device=ring_sampler_env.device)
    max_rel_error = torch.tensor([rel_error], device=ring_sampler_env.device)
    logger.info(f"rank: {ring_sampler_env.rank}, max_error: {max_error}, max_rel_error: {max_rel_error}")
    ps.all_reduce_across_gp_ranks(max_error, op=dist.ReduceOp.AVG)
    ps.all_reduce_across_gp_ranks(max_rel_error, op=dist.ReduceOp.AVG)

    # Gather performance metrics across ranks
    max_memory = torch.tensor([distributed_memory], device=ring_sampler_env.device)
    max_time = torch.tensor([distributed_time], device=ring_sampler_env.device)
    ps.all_reduce_across_gp_ranks(max_memory, op=dist.ReduceOp.MAX)
    ps.all_reduce_across_gp_ranks(max_time, op=dist.ReduceOp.MAX)

    if ring_sampler_env.rank == 0:
        logger.info(f"Maximum relative error across ranks: {max_error.item()}")
        assert max_error.item() < 1e-5, f"Relative error {max_error.item()} is too large"
        assert max_rel_error.item() < 1e-3, f"Relative error {max_rel_error.item()} is too large"
        
        # Print performance metrics
        logger.info("\nPerformance Metrics:")
        logger.info(f"Time: {distributed_time:.4f}s (distributed) vs {baseline_time:.4f}s (baseline)")
        logger.info(f"Memory per GPU: {distributed_memory:.2f}MB (distributed) vs {baseline_memory:.2f}MB (baseline)")
        logger.info(f"Speedup: {baseline_time/distributed_time:.2f}x")
        logger.info(f"Memory reduction per GPU: {(baseline_memory - distributed_memory)/baseline_memory*100:.2f}%")
    
def test_ring_sampler_backward_3d(ring_sampler_env: RingSamplerTestEnv):
    """Test backward pass of ring sampler"""
    
    # Create test tensors with requires_grad=True
    # Image tensor
    img_size = 128
    random_img = create_random_noise_3d(img_size, img_size, img_size)
    random_img = torch.from_numpy(random_img).float().unsqueeze(0).unsqueeze(0)
    random_img = random_img.to(ring_sampler_env.device).detach().requires_grad_(True)

    # Grid tensor (displacement field)
    grid_shape = (1, 196, 160, 224, 3)
    displacement = create_random_noise_3d(grid_shape[1], grid_shape[2], grid_shape[3], channels=3)
    displacement = torch.from_numpy(displacement).float()
    displacement = displacement.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, 196, 160, 224]
    displacement = displacement.permute(0, 2, 3, 4, 1)  # [1, 196, 160, 224, 3]
    displacement = (displacement * 0.1).contiguous().to(ring_sampler_env.device).detach().requires_grad_(True)

    # Affine tensor
    rng = np.random.default_rng(411)
    affine = rng.random((3, 3))
    affine /= np.linalg.norm(affine)
    affine = scipy.linalg.expm(affine * 0.1)
    tr = np.zeros((3, 1))
    affine = np.concatenate([affine, tr], axis=1)  # (3, 4)
    affine_tensor = torch.eye(4, 4, device=ring_sampler_env.device)
    affine_tensor[:3, :] = torch.from_numpy(affine)
    affine_tensor = affine_tensor.unsqueeze(0).contiguous().detach().requires_grad_(True)

    # Run baseline forward pass
    baseline_output = fused_grid_sampler_3d(
        random_img.contiguous(),
        affine=affine_tensor[:, :3, :].contiguous(),
        grid=displacement,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        is_displacement=True
    )

    # Run baseline backward pass
    (baseline_output**2).sum().backward()

    # Store baseline gradients
    baseline_img_grad = random_img.grad.clone().detach()
    baseline_grid_grad = displacement.grad.clone().detach()
    baseline_affine_grad = affine_tensor.grad.clone().detach()

    # Reset gradients for distributed test
    random_img.grad = None
    displacement.grad = None
    affine_tensor.grad = None

    # Shard the image along spatial dim 0 (Z)
    shard_size = (img_size + ring_sampler_env.grid_parallel_size - 1) // ring_sampler_env.grid_parallel_size
    img_start = ring_sampler_env.rank * shard_size
    img_end = min(img_start + shard_size, img_size)

    img_shard = random_img[:, :, img_start:img_end, :, :].contiguous().detach().requires_grad_(True)
    z_min = -1.0 + 2.0 * (img_start / (img_size - 1.0))
    z_max = -1.0 + 2.0 * ((img_end - 1.0) / (img_size - 1.0))
    img_min_coords = torch.tensor((-1.0, -1.0, z_min), device=ring_sampler_env.device)
    img_max_coords = torch.tensor((1.0, 1.0, z_max), device=ring_sampler_env.device)

    # Shard the grid along spatial dim 1 (Z)
    grid_shard_size = (grid_shape[2] + ring_sampler_env.grid_parallel_size - 1) // ring_sampler_env.grid_parallel_size
    grid_start = ring_sampler_env.rank * grid_shard_size
    grid_end = min(grid_start + grid_shard_size, grid_shape[2])

    grid_shard = displacement[:, :, grid_start:grid_end, :, :].contiguous().detach().requires_grad_(True)
    y_min = -1.0 + 2.0 * (grid_start / (grid_shape[2] - 1.0))
    y_max = -1.0 + 2.0 * ((grid_end - 1.0) / (grid_shape[2] - 1.0))
    grid_min_coords = torch.tensor((-1.0, y_min, -1.0), device=ring_sampler_env.device)
    grid_max_coords = torch.tensor((1.0, y_max, 1.0), device=ring_sampler_env.device)

    output_sharded = fireants_ringsampler_interpolator(
        img_shard.contiguous(),
        affine=affine_tensor[:, :3, :].contiguous(),
        grid=grid_shard,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        is_displacement=True,
        min_img_coords=img_min_coords,
        max_img_coords=img_max_coords,
        min_coords=grid_min_coords,
        max_coords=grid_max_coords,
    )

    loss = (output_sharded ** 2).sum()
    ps.all_reduce_across_gp_ranks(loss, op=dist.ReduceOp.SUM)
    loss.backward()

    img_grad = img_shard.grad.clone().detach()
    grid_grad = grid_shard.grad.clone().detach()
    affine_grad = affine_tensor.grad.clone().detach()

    # Compare gradients
    # For image gradient, compare only the sharded portion
    img_grad_error = (torch.abs(img_grad - baseline_img_grad[:, :, img_start:img_end, :, :]) / (torch.abs(baseline_img_grad[:, :, img_start:img_end, :, :]) + 1e-8)).mean()
    grid_grad_error = (torch.abs(grid_grad - baseline_grid_grad[:, :, grid_start:grid_end, :, :]) / (torch.abs(baseline_grid_grad[:, :, grid_start:grid_end, :, :]) + 1e-8)).mean()
    affine_grad_error = (torch.abs(affine_grad - baseline_affine_grad) / (torch.abs(baseline_affine_grad) + 1e-8)).mean()

    # Use all_reduce to get max errors across ranks
    max_errors = torch.tensor([img_grad_error, grid_grad_error, affine_grad_error], device=ring_sampler_env.device)
    ps.all_reduce_across_gp_ranks(max_errors, op=dist.ReduceOp.MAX)

    if ring_sampler_env.rank == 0:
        logger.info(f"Maximum gradient errors across ranks:")
        logger.info(f"Image gradient error: {max_errors[0].item()}")
        logger.info(f"Grid gradient error: {max_errors[1].item()}")
        logger.info(f"Affine gradient error: {max_errors[2].item()}")
        
        # Assert that errors are within acceptable bounds
        assert max_errors[0].item() < 1e-3, f"Image gradient error {max_errors[0].item()} is too large"
        assert max_errors[1].item() < 1e-3, f"Grid gradient error {max_errors[1].item()} is too large"
        assert max_errors[2].item() < 1e-3, f"Affine gradient error {max_errors[2].item()} is too large"


if __name__ == "__main__":
    test_ring_sampler_3d()
    test_ring_sampler_backward_3d()