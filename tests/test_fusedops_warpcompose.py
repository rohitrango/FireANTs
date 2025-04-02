import torch
import pytest
from time import time
from torch import nn
from torch.nn import functional as F
from fireants.interpolator.fused_grid_sample import fused_warp_composer_3d
from fireants.interpolator.grid_sample import torch_warp_composer_3d
# set seeds
seed = 4531
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)   
import itertools

def rel_error(a, b, eps=1e-5):
    return (torch.abs(a - b) / (eps + torch.abs(b))).mean()

def test_fused_warp_composer_3d_correctness():
    """Test correctness of fused warp composer against baseline implementation"""
    Hi, Wi, Di = 196, 224, 260
    H, W, D = 256, 194, 160
    u = torch.randn(1, Hi, Wi, Di, 3).cuda()
    v = torch.randn(1, H, W, D, 3).cuda()
    affine = torch.linalg.matrix_exp(torch.randn(3, 3))
    affine = torch.cat([affine, 0.1*torch.rand(3, 1)-0.05], dim=1)[None].cuda()
    affine = affine.detach().contiguous()
    # check for all combinations of requires_grad
    i = 0
    for ug, vg, ag in itertools.product([True, False], [True, False], [True, False]):
        # detach and set requires_grad
        if ug == False and vg == False and ag == False:
            continue
        i += 1
        print(f"\n\n\n{i})Requires grad: u={ug}, v={vg}, affine={ag}\n------------------------------------------------")
        u = u.detach().requires_grad_(ug)
        v = v.detach().requires_grad_(vg)
        affine = affine.detach().requires_grad_(ag)
        # check for affine 
        for use_affine, align_corners in itertools.product([True, False], [True, False]):
            start_ours = time()
            composed_ours = fused_warp_composer_3d(u, affine if use_affine else None, v, align_corners=align_corners)
            if not composed_ours.requires_grad:
                continue
            print(composed_ours.is_contiguous())
            (composed_ours.mean()).backward()
            torch.cuda.synchronize()
            end_ours = time()
            u_grad_ours = u.grad if u.grad is not None else torch.ones(3).cuda()
            v_grad_ours = v.grad if v.grad is not None else torch.ones(3).cuda()
            affine_grad_ours = affine.grad if (use_affine and affine.grad is not None) else torch.ones(1, 3, 4).cuda()
            # check for all combinations of requires_grad
            if u.requires_grad:
                u.grad = None
            if v.requires_grad:
                v.grad = None
            if affine.requires_grad:
                affine.grad = None
            # compute baseline
            start_baseline = time()
            composed_baseline = torch_warp_composer_3d(u, affine if use_affine else None, v, align_corners=align_corners)
            (composed_baseline.mean()).backward()
            torch.cuda.synchronize()
            end_baseline = time()

            u_grad_baseline = u.grad if u.grad is not None else torch.ones(3).cuda()
            v_grad_baseline = v.grad if v.grad is not None else torch.ones(3).cuda()
            affine_grad_baseline = affine.grad if (use_affine and affine.grad is not None) else torch.ones(1, 3, 4).cuda()
            # check for all combinations of requires_grad
            if u.requires_grad:
                u.grad = None
            if v.requires_grad:
                v.grad = None
            if affine.requires_grad:
                affine.grad = None
            # Check if all quantities match
            print(f"Config: use_affine={use_affine}, align_corners={align_corners}")
            print("rel_error output: ", rel_error(composed_ours, composed_baseline))
            # assert torch.allclose(composed_ours, composed_baseline, atol=1e-3)
            if u.requires_grad:
                print("rel_error u_grad: ", rel_error(u_grad_ours, u_grad_baseline))
                assert rel_error(u_grad_ours, u_grad_baseline) < 1e-4
            if v.requires_grad:
                print("rel_error v_grad: ", rel_error(v_grad_ours, v_grad_baseline))
                assert rel_error(v_grad_ours, v_grad_baseline) < 1e-4
            if affine.requires_grad and use_affine:
                print("rel_error affine_grad: ", rel_error(affine_grad_ours, affine_grad_baseline))
                print(affine_grad_ours, affine_grad_baseline)
                assert rel_error(affine_grad_ours, affine_grad_baseline) < 1e-2
            print(f"Runtime Ours: {end_ours - start_ours}s, Baseline: {end_baseline - start_baseline}s")
            print(f"Speedup: {(end_baseline - start_baseline) / (end_ours - start_ours)}")
            # assert torch.allclose(u_grad_ours, u_grad_baseline, rtol=1e-4)
            # assert torch.allclose(v_grad_ours, v_grad_baseline, rtol=1e-4)
            # assert torch.allclose(affine_grad_ours, affine_grad_baseline, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__])
