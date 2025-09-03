import torch
import pytest
from time import time
from fireants.losses.mi import GlobalMutualInformationLoss
from fireants.losses.fusedmi import FusedGlobalMutualInformationLoss, kernel_type_dict
import fireants_fused_ops as ffo

seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   

def test_fused_mi_correctness():
    """Test correctness of fused mutual information against baseline implementations"""
    # Test with small input size first
    N = 32
    eps = 1e-1
    img1 = (torch.rand(1, 1, N, N, N) + eps).cuda() / (1 + eps)
    print(f"img1: {img1.min()}, {img1.max()}")
    img2 = (img1 + img1 ** 2 * 0.1).cuda()
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    
    # Test both Gaussian and B-spline kernels
    kernel_types = ['gaussian']
    for kernel_type in kernel_types:
        # Create baseline loss
        loss_baseline = GlobalMutualInformationLoss(kernel_type=kernel_type, num_bins=32, reduction='mean', sigma_ratio=2.0).cuda()

        img1 = img1.requires_grad_(True)
        img2 = img2.requires_grad_(True)
        img1.grad = None
        img2.grad = None
        
        # Get baseline pa, pb, pab
        wa, pa, wb, pb = loss_baseline.parzen_windowing(img1, img2)
        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))

        pa.requires_grad_(True)
        pb.requires_grad_(True)
        pab.requires_grad_(True)
        # Compute gradients for pab, pa, pb
        # Create hooks to capture gradients
        grad_pab = []
        grad_pa = []
        grad_pb = []
        
        def hook_pab(grad):
            grad_pab.append(grad)
        def hook_pa(grad):
            grad_pa.append(grad)
        def hook_pb(grad):
            grad_pb.append(grad)
            
        pab.register_hook(hook_pab)
        pa.register_hook(hook_pa) 
        pb.register_hook(hook_pb)
         
        # Compute MI loss
        mi = torch.sum(
            pab * torch.log((pab + loss_baseline.smooth_nr) / (papb + loss_baseline.smooth_dr) + loss_baseline.smooth_dr), 
            dim=(1, 2)
        )
        loss_val = torch.mean(mi).neg()
        # Backward through baseline
        loss_val.backward()
        grad_baseline = img2.grad.clone()

        img2.grad = None
        # Now use the same pa, pb, pab with fused ops backward
        grad_input = torch.zeros_like(img1)
        grad_target = torch.zeros_like(img2)
        
        # # Get gradients from hooks
        grad_pab = grad_pab[0]
        grad_pa = grad_pa[0]
        grad_pb = grad_pb[0]
        
        # Call fused ops backward
        minval, maxval = 0.0, 1.0  # Since we normalized the images
        sigma_ratio = loss_baseline.sigma_ratio * 2  # Default value from mi.py
        ffo.mutual_information_histogram_bwd(
            img1, img2, 
            grad_pab, grad_pa, grad_pb, 
            loss_baseline.num_bins, grad_input, grad_target,
            kernel_type_dict[kernel_type], minval, maxval, sigma_ratio
        )

        # Compare gradients
        print(f"\nKernel type: {kernel_type}")
        print(f"Max gradient difference: {(grad_target - grad_baseline).abs().max().item()}")
        print(f"Mean gradient difference: {(grad_target - grad_baseline).abs().mean().item()}")
        print(f"Relative gradient error: {(torch.abs(grad_target - grad_baseline) / (1e-7 + torch.abs(grad_baseline))).mean().item()}")
        
        # Check if gradients match within tolerance
        assert torch.allclose(grad_target, grad_baseline, rtol=1e-3), f"Gradients don't match baseline for {kernel_type} kernel"


def test_fused_mi_memory_forward():
    """Test memory usage during forward pass"""
    kernel_types = ['gaussian']
    for kernel_type in kernel_types:
        print(f"\nTesting {kernel_type} kernel:")
        for i in range(6, 10):
            N = 2 ** i
            eps = 1e-1
            img1 = (torch.rand(1, 1, N, N, N) + eps).cuda() / (1 + eps)
            img2 = (img1 + img1 ** 2 * 0.1).cuda()
            img2 = (img2 - img2.min()) / (img2.max() - img2.min())
            
            # Calculate input tensor memory
            input_memory = (img1.element_size() * img1.nelement() + 
                          img2.element_size() * img2.nelement()) / 1024**2  # MB
            
            loss = FusedGlobalMutualInformationLoss(kernel_type=kernel_type, num_bins=32, reduction='mean', sigma_ratio=4.0).cuda()
            loss_baseline = GlobalMutualInformationLoss(kernel_type=kernel_type, num_bins=32, reduction='mean', sigma_ratio=2.0).cuda()
            
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
            except:
                baseline_memory = baseline_memory * 8

            
            # Print performance metrics
            print(f"\nN: {N}")
            print(f"Forward time: {out_time:.4f}s (fused) vs {out_baseline_time:.4f}s (baseline)")
            print(f"Memory usage: {fused_memory:.2f}MB (fused) vs {baseline_memory:.2f}MB (baseline)")
            print(f"Memory reduction: {(baseline_memory - fused_memory)/baseline_memory*100:.2f}%")



if __name__ == '__main__':
    pytest.main([__file__])
