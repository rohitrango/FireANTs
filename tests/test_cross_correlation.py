import pytest
import torch
import numpy as np
import logging
from pathlib import Path
import time

# Import FireANTs components
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_relative_error(result1, result2, eps=1e-6):
    """Calculate relative error between two results."""
    return torch.abs(result1 - result2) / (torch.abs(result2) + eps)

@pytest.fixture(scope="class")
def test_data():
    """Fixture to generate test data for cross correlation tests."""
    # Create a directory for test results
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    # Generate test images with different sizes
    sizes = [16, 32, 64]
    kernel_sizes = [3, 5, 7]
    reduction_types = ["mean", "sum", "none"]
    
    # Create test data
    test_data = {}
    for size in sizes:
        for kernel_size in kernel_sizes:
            for reduction in reduction_types:
                key = f"size_{size}_kernel_{kernel_size}_reduction_{reduction}"
                # Create random images with shape [batch, channels, height, width, depth]
                img1 = torch.rand(1, 1, size, size, size)
                img2 = torch.rand(1, 1, size, size, size)
                test_data[key] = {
                    "img1": img1,
                    "img2": img2,
                    "kernel_size": kernel_size,
                    "reduction": reduction
                }
    
    return test_data

class TestCrossCorrelation:
    """Test suite for comparing cross correlation implementations."""
    
    def test_relative_error(self, test_data):
        """Test that the relative error between implementations is within acceptable bounds."""
        # Define acceptable relative error threshold
        error_threshold = 1e-5
        
        # Test results
        results = []
        
        for key, data in test_data.items():
            img1 = data["img1"]
            img2 = data["img2"]
            kernel_size = data["kernel_size"]
            reduction = data["reduction"]
            
            # Create instances of both implementations
            cc_loss = LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                kernel_type="rectangular",
                reduction=reduction
            )
            
            fused_cc_loss = FusedLocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                reduction=reduction
            )
            
            # Compute results
            cc_result = cc_loss(img1, img2)
            fused_cc_result = fused_cc_loss(img1, img2)
            
            # Calculate relative error
            rel_error = calculate_relative_error(cc_result, fused_cc_result)
            
            # Log results
            logger.info(f"Test: {key}")
            logger.info(f"CC result: {cc_result.item():.6f}")
            logger.info(f"Fused CC result: {fused_cc_result.item():.6f}")
            logger.info(f"Relative error: {rel_error.item():.6f}")
            
            # Store results
            results.append({
                "test": key,
                "cc_result": cc_result.item(),
                "fused_cc_result": fused_cc_result.item(),
                "relative_error": rel_error.item()
            })
            
            # Assert that relative error is within acceptable bounds
            assert rel_error.item() < error_threshold, \
                f"Relative error ({rel_error.item():.6f}) exceeds threshold ({error_threshold}) for {key}"
        
        # Print summary
        logger.info("\nCross Correlation Comparison Summary:")
        for result in results:
            logger.info(f"{result['test']}: CC={result['cc_result']:.6f}, Fused CC={result['fused_cc_result']:.6f}, Rel Error={result['relative_error']:.6f}")
    
    def test_performance_comparison(self, test_data):
        """Compare performance between the two implementations."""
        # Select a subset of test cases for performance comparison
        performance_tests = [
            k for k in test_data.keys() 
            if "size_64" in k and "kernel_3" in k and "reduction_mean" in k
        ]
        
        # Number of iterations for timing
        n_iterations = 10
        
        for key in performance_tests:
            data = test_data[key]
            img1 = data["img1"]
            img2 = data["img2"]
            kernel_size = data["kernel_size"]
            reduction = data["reduction"]
            
            # Create instances of both implementations
            cc_loss = LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                kernel_type="rectangular",
                reduction=reduction
            )
            
            fused_cc_loss = FusedLocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                reduction=reduction
            )
            
            # Time the original implementation
            start_time = time.time()
            for _ in range(n_iterations):
                cc_result = cc_loss(img1, img2)
            cc_time = (time.time() - start_time) / n_iterations
            
            # Time the fused implementation
            start_time = time.time()
            for _ in range(n_iterations):
                fused_cc_result = fused_cc_loss(img1, img2)
            fused_cc_time = (time.time() - start_time) / n_iterations
            
            # Calculate speedup
            speedup = cc_time / fused_cc_time
            
            # Log results
            logger.info(f"\nPerformance Test: {key}")
            logger.info(f"Original CC time: {cc_time:.6f} seconds")
            logger.info(f"Fused CC time: {fused_cc_time:.6f} seconds")
            logger.info(f"Speedup: {speedup:.2f}x")
            
            # Assert that fused implementation is faster
            assert fused_cc_time < cc_time, \
                f"Fused implementation ({fused_cc_time:.6f}s) is not faster than original ({cc_time:.6f}s) for {key}"
    
    def test_gradient_comparison(self, test_data):
        """Compare gradients between the two implementations."""
        # Select a subset of test cases for gradient comparison
        gradient_tests = [
            k for k in test_data.keys() 
            if "size_32" in k and "kernel_3" in k and "reduction_mean" in k
        ]
        
        # Define acceptable relative error threshold for gradients
        error_threshold = 1e-4
        
        for key in gradient_tests:
            data = test_data[key]
            img1 = data["img1"].requires_grad_(True)
            img2 = data["img2"].requires_grad_(True)
            kernel_size = data["kernel_size"]
            reduction = data["reduction"]
            
            # Create instances of both implementations
            cc_loss = LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                kernel_type="rectangular",
                reduction=reduction
            )
            
            fused_cc_loss = FusedLocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                reduction=reduction
            )
            
            # Compute results and gradients
            cc_result = cc_loss(img1, img2)
            cc_result.backward()
            cc_grad1 = img1.grad.clone()
            cc_grad2 = img2.grad.clone()
            
            # Reset gradients
            img1.grad = None
            img2.grad = None
            
            # Compute results and gradients for fused implementation
            fused_cc_result = fused_cc_loss(img1, img2)
            fused_cc_result.backward()
            fused_cc_grad1 = img1.grad.clone()
            fused_cc_grad2 = img2.grad.clone()
            
            # Calculate relative errors for gradients
            rel_error_grad1 = calculate_relative_error(cc_grad1, fused_cc_grad1)
            rel_error_grad2 = calculate_relative_error(cc_grad2, fused_cc_grad2)
            
            # Log results
            logger.info(f"\nGradient Test: {key}")
            logger.info(f"CC result: {cc_result.item():.6f}")
            logger.info(f"Fused CC result: {fused_cc_result.item():.6f}")
            logger.info(f"Relative error (grad1): {rel_error_grad1.mean().item():.6f}")
            logger.info(f"Relative error (grad2): {rel_error_grad2.mean().item():.6f}")
            
            # Assert that relative error is within acceptable bounds
            assert rel_error_grad1.mean().item() < error_threshold, \
                f"Relative error for grad1 ({rel_error_grad1.mean().item():.6f}) exceeds threshold ({error_threshold}) for {key}"
            assert rel_error_grad2.mean().item() < error_threshold, \
                f"Relative error for grad2 ({rel_error_grad2.mean().item():.6f}) exceeds threshold ({error_threshold}) for {key}"
    
    def test_different_kernel_types(self, test_data):
        """Test that different kernel types produce consistent results."""
        # Select a test case
        key = [k for k in test_data.keys() if "size_32" in k and "kernel_3" in k and "reduction_mean" in k][0]
        data = test_data[key]
        img1 = data["img1"]
        img2 = data["img2"]
        kernel_size = data["kernel_size"]
        reduction = data["reduction"]
        
        # Define acceptable relative error threshold
        error_threshold = 1e-5
        
        # Test different kernel types
        kernel_types = ["rectangular", "triangular", "gaussian"]
        
        for kernel_type in kernel_types:
            # Create instance of original implementation with specific kernel type
            cc_loss = LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                kernel_type=kernel_type,
                reduction=reduction
            )
            
            # Create instance of fused implementation
            fused_cc_loss = FusedLocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=kernel_size,
                reduction=reduction
            )
            
            # Compute results
            cc_result = cc_loss(img1, img2)
            fused_cc_result = fused_cc_loss(img1, img2)
            
            # Calculate relative error
            rel_error = calculate_relative_error(cc_result, fused_cc_result)
            
            # Log results
            logger.info(f"\nKernel Type Test: {kernel_type}")
            logger.info(f"CC result: {cc_result.item():.6f}")
            logger.info(f"Fused CC result: {fused_cc_result.item():.6f}")
            logger.info(f"Relative error: {rel_error.item():.6f}")
            
            # Assert that relative error is within acceptable bounds
            assert rel_error.item() < error_threshold, \
                f"Relative error ({rel_error.item():.6f}) exceeds threshold ({error_threshold}) for kernel type {kernel_type}" 