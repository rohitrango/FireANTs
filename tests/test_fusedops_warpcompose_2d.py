# Copyright (c) 2026 Rohit Jena. All rights reserved.
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
import pytest
from time import time
import itertools
import gc

from fireants.interpolator.fused_grid_sample import fused_warp_composer_2d
from fireants.interpolator.grid_sample import torch_warp_composer_2d
import logging
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def rel_error(a, b, eps=1e-5):
    """Compute relative error between two tensors."""
    return (torch.abs(a - b) / (eps + torch.abs(b))).mean().item()


def get_memory_allocated():
    """Get current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1024**2


def test_fused_warp_composer_2d_correctness():
    """Test correctness of fused 2D warp composer against baseline implementation."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Test dimensions
    Hi, Wi = 196, 224
    H, W = 256, 194
    
    # Create test tensors
    u = torch.randn(1, Hi, Wi, 2, generator=rng).cuda()
    v = torch.randn(1, H, W, 2, generator=rng).cuda()
    
    # Create random affine transformation
    affine = torch.linalg.matrix_exp(torch.randn(2, 2, generator=rng))
    affine = torch.cat([affine, 0.1*torch.rand(2, 1, generator=rng)-0.05], dim=1)[None].cuda()
    affine = affine.detach().contiguous()
    
    # Test all combinations of requires_grad
    i = 0
    for ug, vg, ag in itertools.product([True, False], [True, False], [True, False]):
        # Skip case where nothing requires grad
        if ug == False and vg == False and ag == False:
            logger.info("Skipping case where nothing requires grad")
            continue
        
        i += 1
        logger.info(f"\n\n\n{i}) Requires grad: u={ug}, v={vg}, affine={ag}\n" + "-"*60)
        
        # Set requires_grad
        u = u.detach().requires_grad_(ug)
        v = v.detach().requires_grad_(vg)
        affine = affine.detach().requires_grad_(ag)
        
        # Test with/without affine and align_corners
        for use_affine, align_corners in itertools.product([True, False], [True, False]):
            logger.info(f"\nConfig: use_affine={use_affine}, align_corners={align_corners}")
            
            # Clear gradients
            if u.grad is not None:
                u.grad = None
            if v.grad is not None:
                v.grad = None
            if affine.grad is not None:
                affine.grad = None
            
            # Test fused implementation
            start_ours = time()
            composed_ours = fused_warp_composer_2d(
                u, affine if use_affine else None, v, align_corners=align_corners
            )
            if not composed_ours.requires_grad:
                logger.info(f"Skipping case where output does not require grad, use_affine={use_affine}, align_corners={align_corners}")
                continue
            
            (composed_ours.mean()).backward()
            torch.cuda.synchronize()
            end_ours = time()
            
            # Save gradients from fused implementation
            u_grad_ours = u.grad.clone() if u.grad is not None else torch.ones(2).cuda()
            v_grad_ours = v.grad.clone() if v.grad is not None else torch.ones(2).cuda()
            affine_grad_ours = affine.grad.clone() if (use_affine and affine.grad is not None) else torch.ones(1, 2, 3).cuda()
            
            # Clear gradients
            if u.grad is not None:
                u.grad = None
            if v.grad is not None:
                v.grad = None
            if affine.grad is not None:
                affine.grad = None
            
            # Test baseline implementation
            start_baseline = time()
            composed_baseline = torch_warp_composer_2d(
                u, affine if use_affine else None, v, align_corners=align_corners
            )
            (composed_baseline.mean()).backward()
            torch.cuda.synchronize()
            end_baseline = time()
            
            # Save gradients from baseline
            u_grad_baseline = u.grad.clone() if u.grad is not None else torch.ones(2).cuda()
            v_grad_baseline = v.grad.clone() if v.grad is not None else torch.ones(2).cuda()
            affine_grad_baseline = affine.grad.clone() if (use_affine and affine.grad is not None) else torch.ones(1, 2, 3).cuda()
            
            # Clear gradients
            if u.grad is not None:
                u.grad = None
            if v.grad is not None:
                v.grad = None
            if affine.grad is not None:
                affine.grad = None
            
            # Check correctness
            error_output = rel_error(composed_ours, composed_baseline)
            logger.info(f"  rel_error output: {error_output:.6e}")
            assert error_output < 1e-4, f"Output error too large: {error_output}"
            
            if u.requires_grad:
                error_u = rel_error(u_grad_ours, u_grad_baseline)
                logger.info(f"  rel_error u_grad: {error_u:.6e}")
                assert error_u < 1e-4, f"u gradient error too large: {error_u}"
            
            if v.requires_grad:
                error_v = rel_error(v_grad_ours, v_grad_baseline)
                logger.info(f"  rel_error v_grad: {error_v:.6e}")
                assert error_v < 1e-4, f"v gradient error too large: {error_v}"
            
            if affine.requires_grad and use_affine:
                error_affine = rel_error(affine_grad_ours, affine_grad_baseline, 1e-2)
                logger.info(f"  rel_error affine_grad: {error_affine:.6e}")
                assert error_affine < 5e-3, f"affine gradient error too large: {error_affine}"
            
            # Performance comparison
            speedup = (end_baseline - start_baseline) / (end_ours - start_ours)
            logger.info(f"  Runtime Ours: {end_ours - start_ours:.4f}s, Baseline: {end_baseline - start_baseline:.4f}s")
            logger.info(f"  Speedup: {speedup:.2f}x")


def test_fused_warp_composer_2d_memory():
    """Test memory usage of fused 2D warp composer vs baseline."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    # Larger test dimensions for memory testing
    Hi, Wi = 512, 512
    H, W = 512, 512
    batch_size = 4
    
    logger.info("\n" + "="*60)
    logger.info("Memory Usage Comparison")
    logger.info("="*60)
    
    # Test memory for fused implementation
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    u = torch.randn(batch_size, Hi, Wi, 2, requires_grad=True).cuda()
    v = torch.randn(batch_size, H, W, 2, requires_grad=True).cuda()
    affine = torch.randn(batch_size, 2, 3, requires_grad=True).cuda()
    
    mem_before_fused = get_memory_allocated()
    peak_before_fused = torch.cuda.max_memory_allocated() / 1024**2
    
    composed_ours = fused_warp_composer_2d(u, affine, v, align_corners=True)
    composed_ours.mean().backward()
    torch.cuda.synchronize()
    
    mem_after_fused = get_memory_allocated()
    peak_fused = torch.cuda.max_memory_allocated() / 1024**2
    mem_used_fused = mem_after_fused - mem_before_fused
    peak_mem_fused = peak_fused - peak_before_fused
    
    logger.info(f"\nFused Implementation:")
    logger.info(f"  Memory used: {mem_used_fused:.2f} MB")
    logger.info(f"  Peak memory: {peak_mem_fused:.2f} MB")
    
    # Clear
    del u, v, affine, composed_ours
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test memory for baseline implementation
    torch.cuda.reset_peak_memory_stats()
    
    u = torch.randn(batch_size, Hi, Wi, 2, requires_grad=True).cuda()
    v = torch.randn(batch_size, H, W, 2, requires_grad=True).cuda()
    affine = torch.randn(batch_size, 2, 3, requires_grad=True).cuda()
    
    mem_before_baseline = get_memory_allocated()
    peak_before_baseline = torch.cuda.max_memory_allocated() / 1024**2
    
    composed_baseline = torch_warp_composer_2d(u, affine, v, align_corners=True)
    composed_baseline.mean().backward()
    torch.cuda.synchronize()
    
    mem_after_baseline = get_memory_allocated()
    peak_baseline = torch.cuda.max_memory_allocated() / 1024**2
    mem_used_baseline = mem_after_baseline - mem_before_baseline
    peak_mem_baseline = peak_baseline - peak_before_baseline
    
    logger.info(f"\nBaseline Implementation:")
    logger.info(f"  Memory used: {mem_used_baseline:.2f} MB")
    logger.info(f"  Peak memory: {peak_mem_baseline:.2f} MB")
    
    logger.info(f"\nMemory Savings:")
    logger.info(f"  Memory used reduction: {((mem_used_baseline - mem_used_fused) / mem_used_baseline * 100):.1f}%")
    logger.info(f"  Peak memory reduction: {((peak_mem_baseline - peak_mem_fused) / peak_mem_baseline * 100):.1f}%")
    
    # Cleanup
    del u, v, affine, composed_baseline
    torch.cuda.empty_cache()
    gc.collect()


def test_fused_warp_composer_2d_batch_sizes():
    """Test fused 2D warp composer with different batch sizes."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    Hi, Wi = 128, 128
    H, W = 128, 128
    
    logger.info("\n" + "="*60)
    logger.info("Batch Size Tests")
    logger.info("="*60)
    
    for batch_size in [1, 2, 4, 8]:
        logger.info(f"\nBatch size: {batch_size}")
        
        u = torch.randn(batch_size, Hi, Wi, 2, generator=rng, requires_grad=True).cuda()
        v = torch.randn(batch_size, H, W, 2, generator=rng, requires_grad=True).cuda()
        affine = torch.randn(batch_size, 2, 3, generator=rng, requires_grad=True).cuda()
        
        # Fused implementation
        start = time()
        composed_ours = fused_warp_composer_2d(u, affine, v, align_corners=True)
        composed_ours.mean().backward()
        torch.cuda.synchronize()
        time_fused = time() - start
        
        u.grad = None
        v.grad = None
        affine.grad = None
        
        # Baseline implementation
        start = time()
        composed_baseline = torch_warp_composer_2d(u, affine, v, align_corners=True)
        composed_baseline.mean().backward()
        torch.cuda.synchronize()
        time_baseline = time() - start
        
        # Check correctness
        error = rel_error(composed_ours, composed_baseline)
        speedup = time_baseline / time_fused
        
        print(f"  Error: {error:.6e}")
        print(f"  Fused time: {time_fused:.4f}s")
        print(f"  Baseline time: {time_baseline:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert error < 1e-4, f"Error too large for batch size {batch_size}: {error}"
        
        del u, v, affine, composed_ours, composed_baseline
        torch.cuda.empty_cache()


def test_fused_warp_composer_2d_different_shapes():
    """Test fused 2D warp composer with different input/output shapes."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    print("\n" + "="*60)
    print("Shape Tests")
    print("="*60)
    
    test_configs = [
        ((128, 128), (128, 128)),  # Same size
        ((256, 256), (128, 128)),  # Downsample
        ((128, 128), (256, 256)),  # Upsample
        ((192, 256), (128, 160)),  # Different aspect ratio
    ]
    
    for (Hi, Wi), (H, W) in test_configs:
        print(f"\nInput shape: ({Hi}, {Wi}), Output shape: ({H}, {W})")
        
        u = torch.randn(1, Hi, Wi, 2, generator=rng, requires_grad=True).cuda()
        v = torch.randn(1, H, W, 2, generator=rng, requires_grad=True).cuda()
        affine = torch.randn(1, 2, 3, generator=rng, requires_grad=True).cuda()
        
        # Test both implementations
        composed_ours = fused_warp_composer_2d(u, affine, v, align_corners=True)
        composed_ours.mean().backward()
        
        u.grad = None
        v.grad = None
        affine.grad = None
        
        composed_baseline = torch_warp_composer_2d(u, affine, v, align_corners=True)
        composed_baseline.mean().backward()
        
        error = rel_error(composed_ours, composed_baseline)
        print(f"  Error: {error:.6e}")
        assert error < 1e-4, f"Error too large: {error}"
        
        del u, v, affine, composed_ours, composed_baseline
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Run all tests
    print("\n" + "="*60)
    print("Testing Fused Warp Composer 2D")
    print("="*60)
    
    test_fused_warp_composer_2d_correctness()
    test_fused_warp_composer_2d_memory()
    test_fused_warp_composer_2d_batch_sizes()
    test_fused_warp_composer_2d_different_shapes()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
