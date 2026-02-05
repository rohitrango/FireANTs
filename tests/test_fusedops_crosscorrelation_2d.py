import torch
import pytest
from time import time
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from fireants.tests.cc_mem_test import fast_lncc
from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss
seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   

from logging import getLogger
logger = getLogger(__name__)
print = logger.info

def rtol(a, b):
    return (torch.abs(a - b) / (1e-5 + torch.abs(b))).mean()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_cc2d_correctness():
    """Test correctness of fused 2D cross correlation against baseline implementations"""
    # Test with small input size first
    N = 128
    eps = 1e-1
    img1 = (torch.rand(1, 1, N, N) + eps).cuda()
    img2 = (torch.rand(1, 1, N, N)*0.5 + 0.5*img1.cpu()).cuda().requires_grad_(True)
    
    smooth_nr = 1e-5
    smooth_dr = 1e-5
    loss = FusedLocalNormalizedCrossCorrelationLoss(2, kernel_size=5, reduction='mean', use_ants_gradient=False, smooth_nr=smooth_nr, smooth_dr=smooth_dr).cuda()
    loss_baseline = LocalNormalizedCrossCorrelationLoss(2, kernel_size=5, reduction='mean', smooth_nr=smooth_nr, smooth_dr=smooth_dr).cuda()
    
    # Forward pass
    out = loss(img2, img1)
    out_baseline = loss_baseline(img1, img2)
    out_baseline2 = fast_lncc(img1, img2, 5)
    
    # Check forward pass results
    print(f"out: {out:.4f}, {out_baseline:.4f}, {out_baseline2:.4f}")
    assert torch.allclose(out, out_baseline, atol=1e-4) or torch.allclose(out, out_baseline2, atol=1e-4), "Forward pass results don't match baseline"
    
    # Backward pass
    out.backward()
    grad_ours = img2.grad / N**2
    
    img2.grad = None
    out_baseline.backward()
    grad_baseline = img2.grad
    
    img2.grad = None
    out_baseline2.backward()
    grad_baseline2 = img2.grad
    
    # Check backward pass results
    logger.info(f"grad_ours: {torch.abs(grad_ours).mean():.4e}, grad_baseline: {torch.abs(grad_baseline).mean():.4e}, grad_baseline2: {torch.abs(grad_baseline2).mean():.4e}")
    assert torch.allclose(grad_ours, grad_baseline, atol=1e-3) or torch.allclose(grad_ours, grad_baseline2, atol=1e-3), f"Gradients don't match baselines {rtol(grad_ours, grad_baseline)} {rtol(grad_ours, grad_baseline2)}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_cc2d_memory_forward():
    """Test memory usage during forward pass for 2D"""
    for i in range(7, 13):  # Smaller sizes for 2D
        N = 2 ** i
        img1 = torch.rand(1, 1, N, N).cuda()
        img2 = (torch.rand(1, 1, N, N)*0.5 + 0.5*img1.cpu()).cuda().requires_grad_(True)
        
        # Calculate input tensor memory
        input_memory = (img1.element_size() * img1.nelement() + 
                       img2.element_size() * img2.nelement()) / 1024**2  # MB
        
        loss = FusedLocalNormalizedCrossCorrelationLoss(2, kernel_size=5, reduction='mean', use_ants_gradient=False, smooth_nr=0).cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(2, kernel_size=5, reduction='mean', smooth_nr=0).cuda()
        
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
        start = time()
        out_baseline = loss_baseline(img1, img2)
        torch.cuda.synchronize()
        out_baseline_time = time() - start
        baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        # Measure baseline2 implementation
        torch.cuda.reset_peak_memory_stats()
        start = time()
        out_baseline2 = fast_lncc(img1, img2, 5)
        torch.cuda.synchronize()
        out_baseline2_time = time() - start
        baseline2_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        # Verify results match
        assert torch.allclose(out, out_baseline, atol=1e-3) or torch.allclose(out, out_baseline2, atol=1e-3), f"Results don't match baseline for N={N}"
        
        # Print performance metrics
        print(f"\nN: {N}")
        print(f"Forward time: {out_time:.4f}s (fused) vs {out_baseline_time:.4f}s (baseline) vs {out_baseline2_time:.4f}s (baseline2)")
        print(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline) vs {baseline2_memory:.2f}MB (baseline2)")
        if baseline_memory > 0:
            print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}% vs baseline, {(baseline2_memory - fused_memory)/baseline2_memory*100:.2f}% vs baseline2")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_cc2d_memory_backward():
    """Test memory usage during backward pass for 2D"""
    for i in range(7, 13):  # Smaller sizes for 2D
        N = 2 ** i
        eps = 1e-1
        img1 = (torch.rand(1, 1, N, N) + eps).cuda().requires_grad_(True)
        img2 = (torch.rand(1, 1, N, N) + eps).cuda().requires_grad_(True)
        
        # Calculate input tensor memory
        input_memory = (img1.element_size() * img1.nelement() + 
                       img2.element_size() * img2.nelement()) / 1024**2  # MB
        
        loss = FusedLocalNormalizedCrossCorrelationLoss(2, kernel_size=5, reduction='mean', use_ants_gradient=False, smooth_nr=0, smooth_dr=0).cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(2, kernel_size=5, reduction='mean', smooth_nr=0, smooth_dr=0).cuda()
        
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
        grad_ours = img2.grad / N**2
        fused_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        # Reset memory stats
        img2.grad = None
        torch.cuda.reset_peak_memory_stats()
        
        # Measure baseline implementation
        t0 = time()
        out_baseline = loss_baseline(img1, img2)
        torch.cuda.synchronize()
        t1 = time()
        out_baseline.backward()
        torch.cuda.synchronize()
        t2 = time()
        fwd_time_baseline = t1 - t0
        bwd_time_baseline = t2 - t1
        grad_baseline = img2.grad
        baseline_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        # Measure baseline2 implementation
        img2.grad = None
        torch.cuda.reset_peak_memory_stats()
        t0 = time()
        out_baseline2 = fast_lncc(img1, img2, 5)
        torch.cuda.synchronize()
        t1 = time()
        out_baseline2.backward()
        torch.cuda.synchronize()
        t2 = time()
        fwd_time_baseline2 = t1 - t0
        bwd_time_baseline2 = t2 - t1
        grad_baseline2 = img2.grad
        baseline2_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        logger.info(f"N={N} | grad_ours - grad_baseline mean abs diff: {(grad_ours - grad_baseline).abs().mean().item():.4e}")
        logger.info(f"N={N} | grad_ours - grad_baseline2 mean abs diff: {(grad_ours - grad_baseline2).abs().mean().item():.4e}")
        
        # Verify results match
        assert torch.allclose(grad_ours, grad_baseline, atol=1e-3) or torch.allclose(grad_ours, grad_baseline2, atol=1e-3), f"Gradients don't match baselines for N={N} {rtol(grad_ours, grad_baseline)} {rtol(grad_ours, grad_baseline2)}"
        
        # Print performance metrics
        print(f"\nN: {N}")
        print(f"Forward time: {fwd_time_ours:.4f}s (fused) vs {fwd_time_baseline:.4f}s (baseline) vs {fwd_time_baseline2:.4f}s (baseline2)")
        print(f"Backward time: {bwd_time_ours:.4f}s (fused) vs {bwd_time_baseline:.4f}s (baseline) vs {bwd_time_baseline2:.4f}s (baseline2)")
        print(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline) vs {baseline2_memory:.2f}MB (baseline2)")
        if baseline_memory > 0:
            print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}% vs baseline, {(baseline2_memory - fused_memory)/baseline2_memory*100:.2f}% vs baseline2")

if __name__ == '__main__':
    pytest.main([__file__])
