# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.

"""Tests for the optional fused_lncc LNCC backend (loss_type='fused_lncc').

Skipped unless a CUDA GPU and the `fused_lncc` package are both available. The backend is
verified to match FusedLocalNormalizedCrossCorrelationLoss (with use_ants_gradient=False, the
exact gradient) in forward value and gradient, and to refuse the configurations it does not
support.
"""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="fused_lncc requires a CUDA GPU"),
    pytest.mark.skipif(importlib.util.find_spec("fused_lncc") is None, reason="fused_lncc not installed"),
]

dev = "cuda"


def _vol(seed, shape=(1, 1, 48, 48, 48)):
    g = torch.Generator(device=dev).manual_seed(seed)
    v = torch.randn(shape, device=dev, generator=g)
    k = torch.ones(1, 1, 7, 7, 7, device=dev) / 343
    for _ in range(2):
        v = F.conv3d(v.reshape(-1, 1, *shape[2:]), k, padding=3).reshape(shape)
    return (v - v.mean()) / (v.std() + 1e-6)


def _fusedcc(k):
    # exact gradient (use_ants_gradient=False) so it is the same algorithm as fused_lncc
    return FusedLocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=k, use_ants_gradient=False).cuda()


@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_forward_and_gradient_match_fusedcc(k):
    from fireants.losses.fused_lncc_backend import FusedLNCCLoss
    fixed, moving = _vol(1), _vol(2)
    mo = moving.clone().requires_grad_(True)
    mf = moving.clone().requires_grad_(True)
    lo = FusedLNCCLoss(spatial_dims=3, kernel_size=k)(mo, fixed)
    lf = _fusedcc(k)(mf, fixed)
    lo.backward(); lf.backward()
    assert abs(lo.item() - lf.item()) < 5e-4                                    # forward value
    assert F.cosine_similarity(mo.grad.flatten(), mf.grad.flatten(), dim=0) > 0.999   # direction
    assert (mo.grad - mf.grad).norm() / mf.grad.norm() < 1e-2                   # magnitude


def test_sign_and_range():
    from fireants.losses.fused_lncc_backend import FusedLNCCLoss
    fixed, moving = _vol(1), _vol(2)
    out = FusedLNCCLoss(kernel_size=7)(moving, fixed).item()
    assert abs(out - _fusedcc(7)(moving, fixed).item()) < 5e-4                  # matches fusedcc sign+value
    assert -1.0001 <= out <= 0.0001                                            # -mean(ncc) in [-1, 0]


def test_batched_gradient_scale():
    from fireants.losses.fused_lncc_backend import FusedLNCCLoss
    fixed, moving = _vol(1, (2, 1, 32, 32, 32)), _vol(2, (2, 1, 32, 32, 32))
    mo = moving.clone().requires_grad_(True)
    mf = moving.clone().requires_grad_(True)
    FusedLNCCLoss(kernel_size=5)(mo, fixed).backward()
    _fusedcc(5)(mf, fixed).backward()
    assert (mo.grad - mf.grad).norm() / mf.grad.norm() < 1e-2


def test_gradient_routes_to_grad_requiring_input():
    from fireants.losses.fused_lncc_backend import FusedLNCCLoss
    fixed = _vol(1).requires_grad_(True)
    moving = _vol(2)  # no grad
    FusedLNCCLoss(kernel_size=7)(moving, fixed).backward()  # pred has no grad -> swap
    assert fixed.grad is not None and fixed.grad.abs().sum() > 0


class TestScopeGuards:
    """Unsupported configurations must fail clearly rather than silently misbehave."""

    def test_gaussian_raises(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        with pytest.raises(NotImplementedError):
            FusedLNCCLoss(kernel_type="gaussian")

    def test_non_mean_reduction_raises(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        with pytest.raises(NotImplementedError):
            FusedLNCCLoss(reduction="sum")

    def test_masked_raises(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        with pytest.raises(NotImplementedError):
            FusedLNCCLoss(masked=True)

    def test_2d_raises(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        with pytest.raises(NotImplementedError):
            FusedLNCCLoss(spatial_dims=2)

    @pytest.mark.parametrize("bad_k", [4, 11])
    def test_unsupported_kernel_size_raises(self, bad_k):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        with pytest.raises(ValueError):
            FusedLNCCLoss(kernel_size=bad_k)

    def test_symmetric_gradient_raises(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        a = _vol(1).requires_grad_(True)
        b = _vol(2).requires_grad_(True)
        with pytest.raises(NotImplementedError):
            FusedLNCCLoss(kernel_size=7)(a, b)


class TestMultiScale:
    """kernel_size may be a list, one entry per registration scale."""

    def test_kernel_size_switches_per_scale(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        loss = FusedLNCCLoss(kernel_size=[7, 5, 3])
        loss.set_scales([4, 2, 1])
        loss.set_current_scale_and_iterations(2, 10)
        assert loss.kernel_size == 5 and loss.get_image_padding() == 2
        loss.set_current_scale_and_iterations(1, 10)
        assert loss.kernel_size == 3 and loss.get_image_padding() == 1

    def test_list_length_must_match_scales(self):
        from fireants.losses.fused_lncc_backend import FusedLNCCLoss
        with pytest.raises(AssertionError):
            FusedLNCCLoss(kernel_size=[7, 5]).set_scales([4, 2, 1])


def test_dispatcher_selects_backend_and_registers():
    """loss_type='fused_lncc' builds the backend and registers as well as fusedcc."""
    import numpy as np
    import SimpleITK as sitk
    from fireants.io.image import Image, BatchedImages
    from fireants.registration.greedy import GreedyRegistration

    S = 48
    fixed = _vol(1, (1, 1, S, S, S))
    disp = _vol(7, (1, 3, S, S, S)) * 0.06
    grid = F.affine_grid(torch.eye(3, 4, device=dev)[None], (1, 1, S, S, S), align_corners=True)
    moving = F.grid_sample(fixed, grid + disp.permute(0, 2, 3, 4, 1), padding_mode="border", align_corners=True)
    fnp, mnp = fixed.cpu().numpy()[0, 0], moving.cpu().numpy()[0, 0]

    def reg(loss_type, loss_params):
        fb = BatchedImages([Image(sitk.GetImageFromArray(fnp.astype(np.float32)), device=dev)])
        mb = BatchedImages([Image(sitk.GetImageFromArray(mnp.astype(np.float32)), device=dev)])
        r = GreedyRegistration(scales=[2, 1], iterations=[25, 25], fixed_images=fb, moving_images=mb,
                               loss_type=loss_type, optimizer="Adam", optimizer_lr=0.2, loss_params=loss_params,
                               cc_kernel_size=5, smooth_warp_sigma=0.25, smooth_grad_sigma=0.5, progress_bar=False)
        r.optimize()
        moved = r.evaluate(fb, mb).detach()
        return r, F.mse_loss(moved, fixed).item()

    r_ours, mse_ours = reg("fused_lncc", {})
    _, mse_fa = reg("fusedcc", {"use_ants_gradient": False})
    assert type(r_ours.loss_fn).__name__ == "FusedLNCCLoss"          # dispatcher built our backend
    assert mse_ours < 0.5                                            # registration actually converged
    assert abs(mse_ours - mse_fa) < 0.1 * mse_fa                     # matches fusedcc quality
