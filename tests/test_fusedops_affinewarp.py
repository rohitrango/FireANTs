import torch
import pytest
from time import time
from torch import nn
from torch.nn import functional as F
from fireants.interpolator.grid_sample import torch_affine_warp_3d
from fireants.interpolator.fused_grid_sample import fused_affine_warp_3d
import itertools

# set seeds
seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   

def test_fused_affine_warp_3d():
    """Test correctness of fused affine warp against baseline implementation"""
    # Test with different input sizes and shapes
    H, W, D = 196, 160, 224
    disp = 0.01 * torch.randn(1, D, H, W, 3).cuda()
    affine = torch.linalg.matrix_exp(torch.randn(3, 3))
    affine = torch.cat([affine, 0.01 * torch.randn(3, 1)], dim=1)[None].cuda()
    affine = affine.detach().contiguous()

    def rel_error(a, b, eps=1e-5):
        return (torch.abs(a - b) / (eps + torch.abs(b))).mean()

    # Test different combinations of affine and warp configurations
    i = 0
    for affine_grad, disp_grad in itertools.product([True, False], [True, False]):
        if not affine_grad and not disp_grad:
            continue
        i += 1
        print(f"\n\n\n{i})Requires grad: affine={affine_grad}, disp={disp_grad}\n------------------------------------------------")
        
        # Set requires_grad for inputs
        affine = affine.detach().requires_grad_(affine_grad)
        disp = disp.detach().requires_grad_(disp_grad)

        # Test with and without affine
        for use_affine, align_corners in itertools.product([True, False], [True, False]):
            if not use_affine and affine_grad:
                continue  # Skip if affine is None but we're testing affine gradients

            # Test fused implementation
            start_ours = time()
            warped_ours = fused_affine_warp_3d(affine if use_affine else None, disp, align_corners=align_corners)
            if not warped_ours.requires_grad:
                continue
            (warped_ours.mean()).backward()
            torch.cuda.synchronize()
            end_ours = time()

            # Store gradients
            disp_grad_ours = disp.grad if disp.grad is not None else torch.ones(3).cuda()
            affine_grad_ours = affine.grad if (use_affine and affine.grad is not None) else torch.ones(1, 3, 4).cuda()

            # Clear gradients
            if disp.requires_grad:
                disp.grad = None
            if affine.requires_grad:
                affine.grad = None

            # Test baseline implementation
            start_baseline = time()
            warped_baseline = torch_affine_warp_3d(affine if use_affine else None, disp, align_corners=align_corners)
            (warped_baseline.mean()).backward()
            torch.cuda.synchronize()
            end_baseline = time()

            # Store baseline gradients
            disp_grad_baseline = disp.grad if disp.grad is not None else torch.ones(3).cuda()
            affine_grad_baseline = affine.grad if (use_affine and affine.grad is not None) else torch.ones(1, 3, 4).cuda()

            # Clear gradients again
            if disp.requires_grad:
                disp.grad = None
            if affine.requires_grad:
                affine.grad = None

            # Check outputs and gradients
            print(f"Config: use_affine={use_affine}, align_corners={align_corners}")
            print("rel_error output: ", rel_error(warped_ours, warped_baseline))
            assert rel_error(warped_ours, warped_baseline) < 1e-4

            if disp.requires_grad:
                print("rel_error disp_grad: ", rel_error(disp_grad_ours, disp_grad_baseline))
                assert rel_error(disp_grad_ours, disp_grad_baseline) < 1e-4

            if affine.requires_grad and use_affine:
                print("rel_error affine_grad: ", rel_error(affine_grad_ours, affine_grad_baseline))
                assert rel_error(affine_grad_ours, affine_grad_baseline) < 1e-2

            print(f"Runtime Ours: {end_ours - start_ours}s, Baseline: {end_baseline - start_baseline}s")
            print(f"Speedup: {(end_baseline - start_baseline) / (end_ours - start_ours)}")

if __name__ == '__main__':
    pytest.main([__file__])

