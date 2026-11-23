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
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from fireants.tests.cc_mem_test import fast_lncc
from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss, FusedNCC2d, FusedNCC2dGaussian, FusedNCC3d, FusedNCC3dGaussian
seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   

def test_fused_cc_correctness():
    """Test correctness of fused cross correlation against baseline implementations"""
    # Test with small input size first
    N = 64
    eps = 1e-0
    rtol_table = {
        torch.float32: 1e-3,
        torch.bfloat16: 1e-2,
    }

    def rel_err(a, b, eps_r=1e-3):
        return (torch.abs(a - b) / (eps_r + torch.abs(b))).mean()

    for dtype in [torch.float32, torch.bfloat16]:
        img1 = (torch.rand(1, 1, N, N, N) + eps).to(dtype).cuda()
        img2 = (torch.rand(1, 1, N, N, N) + eps).to(dtype).cuda().requires_grad_(True)
        
        smooth_dr = 1e-3
        kernel_size = 5
        loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=kernel_size, reduction='mean', smooth_dr=smooth_dr).cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(3, kernel_size=kernel_size, reduction='mean', smooth_dr=smooth_dr).cuda()
        
        # Forward pass
        out = loss(img1, img2)
        out_baseline = loss_baseline(img1, img2)
        out_baseline2 = fast_lncc(img1, img2, kernel_size)
        print(f"\ndtype: {dtype}: Output values: {out}, {out_baseline}, {out_baseline2}")
        
        # Check forward pass results
        rtol = rtol_table[dtype]
        if dtype == torch.float32:
            assert torch.allclose(out, out_baseline, rtol=rtol), f"Forward pass results don't match baseline for dtype {dtype}"
            assert torch.allclose(out, out_baseline2, rtol=rtol), f"Forward pass results don't match baseline2 for dtype {dtype}"
            assert rel_err(out, out_baseline,) < rtol, f"Forward pass results don't match baseline for dtype {dtype}"
            assert rel_err(out, out_baseline2) < rtol, f"Forward pass results don't match baseline2 for dtype {dtype}"
        
        # Backward pass
        out.backward()
        grad_ours = img2.grad
        
        img2.grad = None
        out_baseline.backward()
        grad_baseline = img2.grad
        
        img2.grad = None
        out_baseline2.backward()
        grad_baseline2 = img2.grad

        print(f"Abs error grad: {torch.abs(grad_ours - grad_baseline).mean()}")
        
        # Check backward pass results
        # assert torch.allclose(grad_ours, grad_baseline, rtol=1e-3) or torch.allclose(grad_ours, grad_baseline2, rtol=1e-3), "Gradients don't match baselines"
        rel_grad_1 = rel_err(grad_ours, grad_baseline, 1e-3) 
        rel_grad_2 = rel_err(grad_ours, grad_baseline2, 1e-3) 
        print(rel_grad_1, rel_grad_2)
        assert rel_grad_1 < rtol or rel_grad_2 < rtol, f"Gradients don't match baseline for dtype {dtype}"
        


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("variant", ["3d", "2d", "3d_gaussian", "2d_gaussian"])
def test_fused_cc_gradient_is_the_gradient_of_the_loss(variant, reduction):
    """The exact backward against finite differences of the forward it belongs to, in double precision."""
    torch.manual_seed(seed)
    shape = (2, 2, 9, 10, 11) if variant.startswith("3d") else (2, 2, 15, 17)  # two of each: a mean is over all of them
    img1 = (torch.rand(shape, dtype=torch.float64) * 4 + 1).cuda()
    img2 = (torch.rand(shape, dtype=torch.float64) * 4 + 1).cuda().requires_grad_(True)
    kernel_size, nr, dr = 3, 1e-5, 1e-5
    function, extra = {
        "3d": (FusedNCC3d, ()),
        "2d": (FusedNCC2d, ()),
        "3d_gaussian": (FusedNCC3dGaussian, (1.0,)),
        "2d_gaussian": (FusedNCC2dGaussian, (1.0,)),
    }[variant]
    loss = lambda x: function.apply(x, img1, kernel_size, *extra, nr, dr, reduction, False, True)
    # The backward rewrites its saved intermediates in place, so it runs once: its gradient is checked along random
    # directions against the derivative of the forward there, by central differences.
    (grad,) = torch.autograd.grad(loss(img2), img2)
    for _ in range(3):
        direction = torch.randn_like(img2)
        h = 1e-6
        numeric = (loss(img2.detach() + h * direction) - loss(img2.detach() - h * direction)) / (2 * h)
        assert torch.allclose((grad * direction).sum(), numeric, rtol=1e-5, atol=0), f"{(grad * direction).sum()} vs {numeric}"


def test_fused_cc_memory_forward():
    """Test memory usage during forward pass"""
    for i in range(6, 10):
        N = 2 ** i
        img1 = torch.rand(1, 1, N, N, N).cuda()
        img2 = torch.rand(1, 1, N, N, N).cuda().requires_grad_(True)
        
        # Calculate input tensor memory
        input_memory = (img1.element_size() * img1.nelement() + 
                       img2.element_size() * img2.nelement()) / 1024**2  # MB
        
        loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
        
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
        out_baseline2 = fast_lncc(img1, img2, 3)
        torch.cuda.synchronize()
        out_baseline2_time = time() - start
        baseline2_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        # Verify results match
        assert torch.allclose(out, out_baseline, rtol=1e-4), f"Results don't match baseline for N={N}"
        assert torch.allclose(out, out_baseline2, rtol=1e-4), f"Results don't match baseline2 for N={N}"
        
        # Print performance metrics
        print(f"\nN: {N}")
        print(f"Forward time: {out_time:.4f}s (fused) vs {out_baseline_time:.4f}s (baseline) vs {out_baseline2_time:.4f}s (baseline2)")
        print(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline) vs {baseline2_memory:.2f}MB (baseline2)")
        print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}% vs baseline, {(baseline2_memory - fused_memory)/baseline2_memory*100:.2f}% vs baseline2")

def test_fused_cc_memory_backward():
    """Test memory usage during backward pass"""
    for i in range(6, 10):
        N = 2 ** i
        eps = 1e-1
        img1 = (torch.rand(1, 1, N, N, N) + eps).cuda().requires_grad_(True)
        img2 = (torch.rand(1, 1, N, N, N) + eps).cuda().requires_grad_(True)
        
        # Calculate input tensor memory
        input_memory = (img1.element_size() * img1.nelement() + 
                       img2.element_size() * img2.nelement()) / 1024**2  # MB
        
        loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean', use_ants_gradient=False).cuda()
        loss_baseline = LocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction='mean').cuda()
        
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
        grad_ours = img2.grad
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
        out_baseline2 = fast_lncc(img1, img2, 3)
        torch.cuda.synchronize()
        t1 = time()
        out_baseline2.backward()
        torch.cuda.synchronize()
        t2 = time()
        fwd_time_baseline2 = t1 - t0
        bwd_time_baseline2 = t2 - t1
        grad_baseline2 = img2.grad
        baseline2_memory = (torch.cuda.max_memory_allocated() / 1024**2) - input_memory
        
        # Verify results match
        assert torch.allclose(grad_ours, grad_baseline, rtol=1e-3) or torch.allclose(grad_ours, grad_baseline2, rtol=1e-3), f"Gradients don't match baselines for N={N}"
        
        # Print performance metrics
        print(f"\nN: {N}")
        print(f"Forward time: {fwd_time_ours:.4f}s (fused) vs {fwd_time_baseline:.4f}s (baseline) vs {fwd_time_baseline2:.4f}s (baseline2)")
        print(f"Backward time: {bwd_time_ours:.4f}s (fused) vs {bwd_time_baseline:.4f}s (baseline) vs {bwd_time_baseline2:.4f}s (baseline2)")
        print(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline) vs {baseline2_memory:.2f}MB (baseline2)")
        print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}% vs baseline, {(baseline2_memory - fused_memory)/baseline2_memory*100:.2f}% vs baseline2")

if __name__ == '__main__':
    pytest.main([__file__])
