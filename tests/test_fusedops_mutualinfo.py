import torch
import pytest
from time import time
from fireants.losses.mi import GlobalMutualInformationLoss
from fireants.losses.fusedmi import FusedGlobalMutualInformationLoss
from torch.cuda import OutOfMemoryError
from logging import getLogger
logger = getLogger(__name__)

seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   

def test_fused_mi_correctness():
    """Test correctness of fused mutual information against baseline implementations"""
    # Test with small input size first
    N = 32
    img1 = torch.rand(1, 1, N, N, N).cuda()  # Values from 0 to 1
    img2 = ((img1 + 3*img1**2 + img1**3) / 5).cuda().requires_grad_(True)  # Nonlinear transform of img1
    
    # Test both Gaussian and B-spline kernels
    kernel_types = ['gaussian']
    sigma_ratios = [0.5, 1.0, 2.0]  # Test different sigma ratios
    
    for kernel_type in kernel_types:
        for sigma_ratio in sigma_ratios:
            # Initialize both implementations with same parameters
            loss = FusedGlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=32,
                reduction='mean',
                sigma_ratio=sigma_ratio,
                normalize_image_if_required=True,
                approximate_reduction=False
            ).cuda()
            
            loss_baseline = GlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=32,
                reduction='mean',
                sigma_ratio=sigma_ratio
            ).cuda()
            
            # Forward pass
            out = loss(img1, img2)
            out_baseline = loss_baseline(img1, img2)
            
            # Check forward pass results
            logger.info(f"out: {out}, {out_baseline}")
            assert torch.allclose(out, out_baseline, rtol=1e-4), \
                f"Forward pass results don't match baseline for {kernel_type} kernel with sigma_ratio={sigma_ratio}"
            
            # Backward pass
            out.backward()
            grad_ours = img2.grad.clone()
            
            img2.grad = None
            out_baseline.backward()
            grad_baseline = img2.grad.clone()
            
            # Check backward pass results
            assert torch.allclose(grad_ours, grad_baseline, rtol=1e-3), \
                f"Gradients don't match baseline for {kernel_type} kernel with sigma_ratio={sigma_ratio}"
            
            img2.grad = None


def test_fused_mi_memory_forward():
    """Test memory usage during forward pass"""
    kernel_types = ['gaussian']
    sigma_ratio = 1.0  # Use default sigma ratio for memory tests
    
    for kernel_type in kernel_types:
        logger.info(f"\nTesting {kernel_type} kernel:")
        for i in range(6, 10):
            N = 2 ** i
            img1 = torch.rand(1, 1, N, N, N).cuda()  # Values from 0 to 1
            img2 = ((img1 + 3*img1**2 + img1**3) / 5).cuda().requires_grad_(True)  # Nonlinear transform of img1
            
            torch.cuda.reset_peak_memory_stats()
            # # Calculate input tensor memory
            # input_memory = (img1.element_size() * img1.nelement() + 
            #               img2.element_size() * img2.nelement()) / 1024**2  # MB
            input_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            loss = FusedGlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=32,
                reduction='mean',
                sigma_ratio=sigma_ratio,
                normalize_image_if_required=True,
                approximate_reduction=False
            ).cuda()
            
            loss_baseline = GlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=32,
                reduction='mean',
                sigma_ratio=sigma_ratio
            ).cuda()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Measure fused implementation
            start = time()
            out = loss(img1, img2)
            torch.cuda.synchronize()
            out_time = time() - start
            fused_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Measure baseline implementation
            try:
                start = time()
                out_baseline = loss_baseline(img1, img2)
                torch.cuda.synchronize()
                out_baseline_time = time() - start
                baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
            except OutOfMemoryError:
                baseline_memory = baseline_memory * 8
                out_baseline_time = 100000
            
            # Verify results match
            assert torch.allclose(out, out_baseline, rtol=1e-4), f"Results don't match baseline for N={N}"
            
            # logger.info performance metrics
            logger.info(f"\nN: {N}")
            logger.info(f"Forward time: {out_time:.4f}s (fused) vs {out_baseline_time:.4f}s (baseline)")
            logger.info(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline)")
            logger.info(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}%")


def test_fused_mi_memory_backward():
    """Test memory usage during backward pass"""
    kernel_types = ['gaussian']
    sigma_ratio = 1.0  # Use default sigma ratio for memory tests
    
    for kernel_type in kernel_types:
        logger.info(f"\nTesting {kernel_type} kernel:")
        for i in range(6, 10):
            N = 2 ** i
            img1 = torch.rand(1, 1, N, N, N).cuda()  # Values from 0 to 1
            img2 = ((img1 + 3*img1**2 + img1**3) / 5).cuda().requires_grad_(True)  # Nonlinear transform of img1
            
            torch.cuda.reset_peak_memory_stats()
            # Calculate input tensor memory
            # input_memory = (img1.element_size() * img1.nelement() + 
            #               img2.element_size() * img2.nelement()) / 1024**2  # MB
            input_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            loss = FusedGlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=32,
                reduction='mean',
                sigma_ratio=sigma_ratio,
                normalize_image_if_required=True,
                approximate_reduction=False
            ).cuda()
            
            loss_baseline = GlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=32,
                reduction='mean',
                sigma_ratio=sigma_ratio
            ).cuda()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Measure fused implementation
            t0 = time()
            out = loss(img1, img2)
            torch.cuda.synchronize()
            t1 = time()
            out.backward()
            torch.cuda.synchronize()
            t2 = time()
            fwd_time_ours = t1 - t0
            bwd_time_ours = t2 - t1
            grad_ours = img2.grad.clone()
            fused_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
            
            # Reset memory stats
            img2.grad = None
            torch.cuda.reset_peak_memory_stats()
            
            # Measure baseline implementation
            try:
                t0 = time()
                out_baseline = loss_baseline(img1, img2)
                torch.cuda.synchronize()
                t1 = time()
                out_baseline.backward()
                torch.cuda.synchronize()
                t2 = time()
                fwd_time_baseline = t1 - t0
                bwd_time_baseline = t2 - t1
                grad_baseline = img2.grad.clone()
                baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
                # Verify results match
                assert torch.allclose(grad_ours, grad_baseline, rtol=1e-3), f"Gradients don't match baseline for N={N}"
            except OutOfMemoryError:
                baseline_memory = baseline_memory * 8
                fwd_time_baseline = 100000
                bwd_time_baseline = 100000
            
            # Print performance metrics
            logger.info(f"\nN: {N}")
            logger.info(f"Forward time: {fwd_time_ours:.4f}s (fused) vs {fwd_time_baseline:.4f}s (baseline)")
            logger.info(f"Backward time: {bwd_time_ours:.4f}s (fused) vs {bwd_time_baseline:.4f}s (baseline)")
            logger.info(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline)")
            logger.info(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}%")


def test_fused_mi_approximate_reduction():
    """Test that approximate reduction gives similar results"""
    N = 32
    img1 = torch.rand(1, 1, N, N, N).cuda()  # Values from 0 to 1
    img2 = ((img1 + 3*img1**2 + img1**3) / 5).cuda().requires_grad_(True)  # Nonlinear transform of img1
    
    kernel_types = ['gaussian']
    for kernel_type in kernel_types:
        # Initialize exact and approximate implementations
        loss_exact = FusedGlobalMutualInformationLoss(
            kernel_type=kernel_type,
            num_bins=32,
            reduction='mean',
            approximate_reduction=False
        ).cuda()
        
        loss_approx = FusedGlobalMutualInformationLoss(
            kernel_type=kernel_type,
            num_bins=32,
            reduction='mean',
            approximate_reduction=True
        ).cuda()
        
        # Forward pass
        out_exact = loss_exact(img1, img2)
        out_approx = loss_approx(img1, img2)
        
        # Check forward pass results - allow larger tolerance for approximate version
        assert out_approx.item() < out_exact.item() + 1e-2, \
            f"Approximate reduction results should be smaller than exact reduction results due to sharpening effect"


def test_fused_mi_num_bins_ablation():
    """Test effect of different numbers of bins on MI computation"""
    N = 128  # Use smaller size for bin ablation
    img1 = torch.rand(1, 1, N, N, N).cuda()  # Values from 0 to 1
    img2 = ((img1 + 3*img1**2 + img1**3) / 5).cuda().requires_grad_(True)  # Nonlinear transform of img1
    
    kernel_types = ['gaussian']
    num_bins_list = [8, 16, 32, 64]
    
    for kernel_type in kernel_types:
        logger.info(f"\nTesting {kernel_type} kernel with different numbers of bins:")
        for num_bins in num_bins_list:
            # Initialize both implementations
            loss = FusedGlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=num_bins,
                reduction='mean',
                approximate_reduction=False
            ).cuda()
            
            loss_baseline = GlobalMutualInformationLoss(
                kernel_type=kernel_type,
                num_bins=num_bins,
                reduction='mean'
            ).cuda()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            input_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            # Measure fused implementation
            t0 = time()
            out = loss(img1, img2)
            torch.cuda.synchronize()
            t1 = time()
            out.backward()
            torch.cuda.synchronize()
            t2 = time()
            fwd_time_ours = t1 - t0
            bwd_time_ours = t2 - t1
            grad_ours = img2.grad.clone()
            fused_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
            
            # Reset memory stats
            img2.grad = None
            torch.cuda.reset_peak_memory_stats()
            
            # Measure baseline implementation
            try:
                t0 = time()
                out_baseline = loss_baseline(img1, img2)
                torch.cuda.synchronize()
                t1 = time()
                out_baseline.backward()
                torch.cuda.synchronize()
                t2 = time()
                fwd_time_baseline = t1 - t0
                bwd_time_baseline = t2 - t1
                grad_baseline = img2.grad.clone()
                baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
                
                # Verify results match
                assert torch.allclose(out, out_baseline, rtol=1e-4), \
                    f"Results don't match baseline for num_bins={num_bins}"
                assert torch.allclose(grad_ours, grad_baseline, rtol=1e-3), \
                    f"Gradients don't match baseline for num_bins={num_bins}"
            except OutOfMemoryError:
                baseline_memory = baseline_memory * 8
                fwd_time_baseline = 100000
                bwd_time_baseline = 100000
            
            # Log performance metrics
            logger.info(f"\nNumber of bins: {num_bins}")
            logger.info(f"MI value: {out.item():.4f}, baseline: {out_baseline.item():.4f}")
            logger.info(f"Forward time: {fwd_time_ours:.4f}s (fused) vs {fwd_time_baseline:.4f}s (baseline)")
            logger.info(f"Backward time: {bwd_time_ours:.4f}s (fused) vs {bwd_time_baseline:.4f}s (baseline)")
            logger.info(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline)")
            logger.info(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}%")
            
            img2.grad = None


if __name__ == '__main__':
    pytest.main([__file__])