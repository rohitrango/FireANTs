import pytest
import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import logging
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Dict, Any, List, Tuple

from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.io.image import Image, BatchedImages
try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

# Set up logging
logger = logging.getLogger(__name__)

# Test data paths
test_data_dir = Path(__file__).parent / "test_data"
fixed_image_path = str(test_data_dir / "deformable_image_1.nii.gz")
moving_image_path = str(test_data_dir / "deformable_image_2.nii.gz")
fixed_seg_path = str(test_data_dir / "deformable_seg_1.nii.gz")
moving_seg_path = str(test_data_dir / "deformable_seg_2.nii.gz")

def get_memory_stats() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0

def run_registration_with_profiling(reg_class, fixed_batch: BatchedImages, 
                                  moving_batch: BatchedImages,
                                  fixed_seg_batch: BatchedImages,
                                  moving_seg_batch: BatchedImages,
                                  scales: List[int],
                                  iterations: List[int],
                                  **kwargs) -> Tuple[float, Dict[str, Any]]:
    """Run registration with minimal profiling and return dice score and performance metrics."""
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Profile only the registration part
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
        with_modules=False
    ) as prof:
        with record_function("registration"):
            # Create and run registration
            reg = reg_class(
                scales=scales,
                iterations=iterations,
                fixed_images=fixed_batch,
                moving_images=moving_batch,
                loss_type='cc',
                optimizer='Adam',
                optimizer_lr=0.2,
                smooth_warp_sigma=0.25,
                smooth_grad_sigma=0.5,
                progress_bar=True,
                **kwargs
            )
            reg.optimize()
    
    # Get results and compute dice score
    moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch).detach()
    dice_scores = 1 - dice_loss(moved_seg_batch, 
                              fixed_seg_batch(), 
                              reduce=False).mean(0)
    mean_dice = dice_scores.mean().item()

    # Calculate total FLOPs
    total_flops = 0
    events = prof.key_averages()
    for evt in events:
        if evt.flops:
            total_flops += evt.flops

    # Get stats
    stats = {
        'peak_memory_mb': get_memory_stats(),
        'total_flops': total_flops,
    }
    return mean_dice, stats

@pytest.fixture(scope="module")
def baseline_data():
    """Fixture to load and prepare test data."""
    # Load images
    fixed_image = Image.load_file(fixed_image_path)
    moving_image = Image.load_file(moving_image_path)
    fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True)
    moving_seg = Image.load_file(moving_seg_path, is_segmentation=True)
    
    # Create batched images
    fixed_batch = BatchedImages([fixed_image])
    moving_batch = BatchedImages([moving_image])
    fixed_seg_batch = BatchedImages([fixed_seg])
    moving_seg_batch = BatchedImages([moving_seg])
    
    # Get initial dice score
    dice_score_before = 1 - dice_loss(moving_seg_batch(), 
                                    fixed_seg_batch(), 
                                    reduce=False).mean(0)
    mean_dice_before = dice_score_before.mean().item()
    
    return {
        'fixed_batch': fixed_batch,
        'moving_batch': moving_batch,
        'fixed_seg_batch': fixed_seg_batch,
        'moving_seg_batch': moving_seg_batch,
        'initial_dice': mean_dice_before
    }

class TestPrecisionModes:
    """Test registration with different precision modes."""
    
    # Store results for comparison
    results = {}
    
    @pytest.mark.parametrize("reg_class", [GreedyRegistration, SyNRegistration])
    def test_mixed_precision(self, reg_class, baseline_data):
        """Test registration with bf16 images and fp32 warps."""
        # Convert images to bf16
        fixed_batch = baseline_data['fixed_batch'].to(torch.bfloat16)
        moving_batch = baseline_data['moving_batch'].to(torch.bfloat16)
        fixed_seg_batch = baseline_data['fixed_seg_batch'].to(torch.bfloat16)
        moving_seg_batch = baseline_data['moving_seg_batch'].to(torch.bfloat16)
        
        mean_dice, stats = run_registration_with_profiling(
            reg_class, fixed_batch, moving_batch,
            fixed_seg_batch, moving_seg_batch,
            scales=[4, 2, 1], iterations=[200, 100, 50],
            dtype=torch.float32  # Keep warps in fp32
        )
        
        logger.info(f"\n{reg_class.__name__} Mixed Precision Results (bf16 images, fp32 warps):")
        logger.info(f"Dice Score: {mean_dice:.3f}")
        logger.info(f"Peak GPU Memory: {stats['peak_memory_mb']:.2f} MB")
        logger.info(f"Total FLOPs: {stats['total_flops']:,}")
        
        # Store results for comparison
        TestPrecisionModes.results[f"{reg_class.__name__}_mixed"] = {
            'dice': mean_dice,
            'memory': stats['peak_memory_mb'],
            'flops': stats['total_flops']
        }
        
        assert mean_dice > baseline_data['initial_dice'], "Mixed precision dice score not better than initial"
    
    @pytest.mark.parametrize("reg_class", [GreedyRegistration, SyNRegistration])
    def test_bf16_precision(self, reg_class, baseline_data):
        """Test registration with bf16 images and bf16 warps."""
        # Convert images to bf16
        fixed_batch = baseline_data['fixed_batch'].to(torch.bfloat16)
        moving_batch = baseline_data['moving_batch'].to(torch.bfloat16)
        fixed_seg_batch = baseline_data['fixed_seg_batch'].to(torch.bfloat16)
        moving_seg_batch = baseline_data['moving_seg_batch'].to(torch.bfloat16)
        
        mean_dice, stats = run_registration_with_profiling(
            reg_class, fixed_batch, moving_batch,
            fixed_seg_batch, moving_seg_batch,
            scales=[4, 2, 1], iterations=[200, 100, 50],
            dtype=torch.bfloat16,  # Keep warps in bf16
        )
        
        logger.info(f"\n{reg_class.__name__} BF16 Results:")
        logger.info(f"Dice Score: {mean_dice:.3f}")
        logger.info(f"Peak GPU Memory: {stats['peak_memory_mb']:.2f} MB")
        logger.info(f"Total FLOPs: {stats['total_flops']:,}")
        
        # Store results for comparison
        TestPrecisionModes.results[f"{reg_class.__name__}_bf16"] = {
            'dice': mean_dice,
            'memory': stats['peak_memory_mb'],
            'flops': stats['total_flops']
        }
        
        # Print comparison after both tests complete
        if len(TestPrecisionModes.results) == 2:  # Both tests for this registration class done
            reg_name = reg_class.__name__
            mixed_res = TestPrecisionModes.results[f"{reg_name}_mixed"]
            bf16_res = TestPrecisionModes.results[f"{reg_name}_bf16"]
            
            logger.info(f"\n=== {reg_name} Precision Mode Comparison ===")
            logger.info(f"Initial Dice Score: {baseline_data['initial_dice']:.3f}")
            logger.info(f"Mixed Precision (bf16 img + fp32 warp) Dice: {mixed_res['dice']:.3f}")
            logger.info(f"Full BF16 Dice: {bf16_res['dice']:.3f}")
            logger.info(f"Difference (Mixed - BF16): {(mixed_res['dice'] - bf16_res['dice']):.3f}")
        
        assert mean_dice > baseline_data['initial_dice'], "BF16 dice score not better than initial"


class TestMultiScaleOptimization:
    """Test multi-scale optimization strategies."""
    
    @pytest.mark.parametrize("reg_class", [GreedyRegistration, SyNRegistration])
    def test_optimization_comparison(self, reg_class, baseline_data):
        """Compare partial vs full optimization strategies."""
        # Run partial optimization (coarse levels only)
        partial_dice, partial_stats = run_registration_with_profiling(
            reg_class, baseline_data['fixed_batch'], baseline_data['moving_batch'],
            baseline_data['fixed_seg_batch'], baseline_data['moving_seg_batch'],
            scales=[8, 4, 2], iterations=[200, 100, 50]  # Only optimize at coarse levels
        )
        
        # Run full optimization (including finest level)
        full_dice, full_stats = run_registration_with_profiling(
            reg_class, baseline_data['fixed_batch'], baseline_data['moving_batch'],
            baseline_data['fixed_seg_batch'], baseline_data['moving_seg_batch'],
            scales=[8, 4, 2, 1], iterations=[200, 100, 50, 25]  # Full optimization including level 1
        )
        
        # Log results
        logger.info(f"\n{reg_class.__name__} Optimization Comparison:")
        logger.info("Partial Optimization ([8,4,2]):")
        logger.info(f"- Dice Score: {partial_dice:.3f}")
        logger.info(f"- Peak GPU Memory: {partial_stats['peak_memory_mb']:.2f} MB")
        
        logger.info("\nFull Optimization ([8,4,2,1]):")
        logger.info(f"- Dice Score: {full_dice:.3f}")
        logger.info(f"- Peak GPU Memory: {full_stats['peak_memory_mb']:.2f} MB")
        logger.info(f"- Total FLOPs: {full_stats['total_flops']:,}")
        
        logger.info(f"\nComparison:")
        logger.info(f"- Dice Score Improvement: {(full_dice - partial_dice):.3f}")
        logger.info(f"- Memory Increase: {(full_stats['peak_memory_mb'] - partial_stats['peak_memory_mb']):.2f} MB")
        logger.info(f"- Additional FLOPs: {(full_stats['total_flops'] - partial_stats['total_flops']):,}")
        
        # Assertions
        assert partial_dice > baseline_data['initial_dice'], "Partial optimization dice score not better than initial"
        assert full_dice > baseline_data['initial_dice'], "Full optimization dice score not better than initial"
        assert full_dice >= partial_dice, f"Full optimization ({full_dice:.3f}) should perform at least as well as partial optimization ({partial_dice:.3f})"
