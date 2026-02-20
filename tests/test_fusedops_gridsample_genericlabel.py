"""
Tests for fused generic label grid sampler (2D and 3D).

Baseline: convert label map to one-hot, bilinear interpolate via fused_grid_sampler,
then argmax over channels for labels and use the interpolated value at the winning
channel as the weight. Forward checks labels and weights; backward uses sum(weights)
as loss and compares gradients of affine and deformation fields.
"""
import pytest
pytest.importorskip("fireants_fused_ops")

import os
import torch
from torch.nn import functional as F
import SimpleITK as sitk
import logging

from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d, fused_grid_sampler_2d
from fireants.utils.imageutils import integer_to_onehot

# Skip entire module if fused_ops was not built with generic label support
from fireants.interpolator.fused_grid_sample_genericlabel import (
    fused_grid_sampler_3d_generic_label,
    fused_grid_sampler_2d_generic_label,
)

logger = logging.getLogger(__name__)

SEED = 1221
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Directory for test data (relative to repo root or test file)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FUSED_GENERICLABEL_DATA = os.path.join(TESTS_DIR, "test_data", "fused_genericlabel")


def _load_label_sitk(path: str) -> torch.Tensor:
    """Load a segmentation/label image with SimpleITK; return tensor [1, 1, *spatial] long on CUDA."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X) for 3D or (Y, X) for 2D
    x = torch.from_numpy(arr.astype(int)).long().cuda()
    if x.ndim == 2:
        x = x[None, None, ...]  # (1, 1, H, W)
    else:
        x = x[None, None, ...]  # (1, 1, Z, Y, X)
    return x


def _label_to_onehot_3d(label: torch.Tensor, background_label: int = -1, max_label: int = None) -> torch.Tensor:
    """Label [1, 1, Z, Y, X] long -> one-hot [1, C, Z, Y, X] float."""
    assert label.ndim == 5 and label.shape[0] == 1 and label.shape[1] == 1
    flat = label.squeeze(0).squeeze(0)  # (Z, Y, X)
    if max_label is None:
        max_label = flat.max().item()
    onehot = integer_to_onehot(flat, background_label=background_label, dtype=torch.float32, max_label=max_label)
    return onehot[None]  # (1, C, Z, Y, X)


def _label_to_onehot_2d(label: torch.Tensor, background_label: int = -1, max_label: int = None) -> torch.Tensor:
    """Label [1, 1, H, W] long -> one-hot [1, C, H, W] float."""
    assert label.ndim == 4 and label.shape[0] == 1 and label.shape[1] == 1
    flat = label.squeeze(0).squeeze(0)  # (H, W)
    if max_label is None:
        max_label = flat.max().item()
    onehot = integer_to_onehot(flat, background_label=background_label, dtype=torch.float32, max_label=max_label)
    return onehot[None]  # (1, C, H, W)


def _baseline_generic_label_3d(
    onehot: torch.Tensor,
    affine: torch.Tensor = None,
    grid: torch.Tensor = None,
    grid_affine: torch.Tensor = None,
    out_shape: tuple = None,
    is_displacement: bool = True,
    align_corners: bool = True,
    padding_mode: str = "zeros",
):
    """Baseline: bilinear sample one-hot, then per-voxel we have (C,) weights; labels = weights (same as weights for one-hot)."""
    weights = fused_grid_sampler_3d(
        onehot,
        affine=affine,
        grid=grid,
        grid_affine=grid_affine,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
        out_shape=out_shape,
        is_displacement=is_displacement,
    )
    # Fused generic label outputs (B, C, D, H, W): for each channel, the interpolated value (weight) and the "label" (same value for one-hot).
    # return weights, weights
    maxdata = weights.max(dim=1, keepdim=True)
    labels = maxdata.indices
    weights = maxdata.values
    return labels, weights


def _baseline_generic_label_2d(
    onehot: torch.Tensor,
    affine: torch.Tensor = None,
    grid: torch.Tensor = None,
    grid_affine: torch.Tensor = None,
    out_shape: tuple = None,
    is_displacement: bool = True,
    align_corners: bool = True,
    padding_mode: str = "zeros",
):
    """Baseline: bilinear sample one-hot, then argmax over channels -> (labels [B,1,H,W], weights [B,1,H,W])."""
    weights = fused_grid_sampler_2d(
        onehot,
        affine=affine,
        grid=grid,
        grid_affine=grid_affine,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
        out_shape=out_shape,
        is_displacement=is_displacement,
    )
    maxdata = weights.max(dim=1, keepdim=True)
    labels = maxdata.indices
    weights = maxdata.values
    return labels, weights


# --------------- 3D tests ---------------


@pytest.mark.parametrize("use_affine", [True, False])
@pytest.mark.parametrize("use_grid", [True, False])
@pytest.mark.parametrize("is_displacement", [True, False])
def test_fused_generic_label_3d_forward(use_affine, use_grid, is_displacement):
    """3D forward: compare fused generic label vs baseline (one-hot + bilinear + same semantics)."""
    if not use_affine and not use_grid:
        pytest.skip("need at least affine or grid")
    rng = torch.Generator()
    rng.manual_seed(SEED)

    # Synthetic 3D label map
    B, Z, Y, X = 1, 12, 14, 16
    num_classes = 4
    label_int = torch.randint(0, num_classes, (1, 1, Z, Y, X), generator=rng).long().cuda()
    onehot = _label_to_onehot_3d(label_int, background_label=-1, max_label=num_classes - 1)
    onehot = onehot.cuda().float()

    out_shape = (8, 10, 12)
    align_corners = True
    padding_mode = "zeros"

    affine = None
    grid = None
    grid_affine = None
    if use_affine:
        aff = torch.linalg.matrix_exp(0.02 * torch.randn(3, 3, generator=rng)).cuda()
        aff = torch.cat([aff, 0.01 * torch.randn(3, 1, generator=rng).cuda()], dim=1)[None]
        affine = aff.detach().requires_grad_(True)
    if use_grid:
        if is_displacement:
            grid = (0.5 * torch.randn(1, *out_shape, 3, generator=rng).cuda()).detach().requires_grad_(True)
        else:
            base = F.affine_grid(torch.eye(3, 4)[None].cuda(), (1, 1, *out_shape), align_corners=align_corners)
            grid = (base + 0.02 * torch.randn(1, *out_shape, 3, generator=rng).cuda()).detach().requires_grad_(True)
        grid_affine = torch.eye(3, 3)[None].cuda()

    # baseline weights and labels
    baseline_labels, baseline_weights = _baseline_generic_label_3d(
        onehot, affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=is_displacement,
        align_corners=align_corners, padding_mode=padding_mode,
    )

    # Fused: input is one-hot (same as baseline)
    out_labels, out_weights = fused_grid_sampler_3d_generic_label(
        label_int.float(),
        affine=affine,
        grid=grid,
        grid_affine=grid_affine,
        out_shape=out_shape,
        is_displacement=is_displacement,
        align_corners=align_corners,
        padding_mode=padding_mode,
        return_probs=True,
        background_label=None,
    )
    assert out_labels.shape == baseline_labels.shape
    assert out_weights.shape == baseline_weights.shape

    label_agree_ratio = (out_labels.long() == baseline_labels.long()).float().mean().item()
    rel_weights = (torch.abs(out_weights - baseline_weights) / (1e-6 + torch.abs(baseline_weights))).mean().item()
    logger.info(f"3D forward use_affine={use_affine} use_grid={use_grid} is_disp={is_displacement} label_agree_ratio={label_agree_ratio} rel_weights={rel_weights}")
    assert rel_weights < 5e-3, f"Weights relative error too high: {rel_weights}"
    assert label_agree_ratio > 0.99, f"Label agreement too low: {label_agree_ratio}"


def test_fused_generic_label_3d_backward():
    """3D backward: loss = sum(weights), compare grad affine and grad grid. Same input as 3D forward."""
    rng = torch.Generator()
    rng.manual_seed(SEED)

    # Same synthetic 3D setup as test_fused_generic_label_3d_forward
    B, Z, Y, X = 1, 12, 14, 16
    num_classes = 4
    label_int = torch.randint(0, num_classes, (1, 1, Z, Y, X), generator=rng).long().cuda()
    onehot = _label_to_onehot_3d(label_int, background_label=-1, max_label=num_classes - 1)
    onehot = onehot.cuda().float()

    out_shape = (8, 10, 12)
    align_corners = True
    padding_mode = "zeros"

    aff = torch.linalg.matrix_exp(0.02 * torch.randn(3, 3, generator=rng)).cuda()
    aff = torch.cat([aff, 0.01 * torch.randn(3, 1, generator=rng).cuda()], dim=1)[None]
    affine = aff.detach().requires_grad_(True)
    grid = (0.5 * torch.randn(1, *out_shape, 3, generator=rng).cuda()).detach().requires_grad_(True)
    grid_affine = torch.eye(3, 3)[None].cuda()

    # Baseline: loss = sum(weights), backward
    baseline_labels, baseline_weights = _baseline_generic_label_3d(
        onehot, affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners, padding_mode=padding_mode,
    )
    loss_baseline = baseline_weights.sum()
    loss_baseline.backward()
    grad_affine_baseline = affine.grad.clone() if affine.grad is not None else None
    grad_grid_baseline = grid.grad.clone() if grid.grad is not None else None

    # Fused: use clones of affine/grid so we have a separate graph (avoid "backward through graph a second time")
    affine_fused = affine.detach().clone().requires_grad_(True)
    grid_fused = grid.detach().clone().requires_grad_(True)
    out_labels, out_weights = fused_grid_sampler_3d_generic_label(
        label_int.float(), affine=affine_fused, grid=grid_fused, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners,
        padding_mode=padding_mode, return_probs=True,
    )
    loss_fused = out_weights.sum()
    loss_fused.backward()
    grad_affine_fused = affine_fused.grad
    grad_grid_fused = grid_fused.grad

    logger.info(f"3D backward loss_baseline={loss_baseline.item()}, loss_fused={loss_fused.item()}")

    if grad_affine_baseline is not None and grad_affine_fused is not None:
        rel_affine = (torch.abs(grad_affine_fused - grad_affine_baseline) / (1e-6 + torch.abs(grad_affine_baseline))).mean().item()
        logger.info(f"3D backward rel_affine={rel_affine}")
        assert rel_affine < 1e-4, f"Affine gradient relative error too high: {rel_affine}"
    if grad_grid_baseline is not None and grad_grid_fused is not None:
        rel_grid = (torch.abs(grad_grid_fused - grad_grid_baseline) / (1e-6 + torch.abs(grad_grid_baseline))).mean().item()
        logger.info(f"3D backward rel_grid={rel_grid}")
        assert rel_grid < 1e-4, f"Grid gradient relative error too high: {rel_grid}"


def test_fused_generic_label_3d_from_data_dir():
    """3D forward/backward with same synthetic input as test_fused_generic_label_3d_forward."""
    rng = torch.Generator()
    rng.manual_seed(SEED)

    # Same synthetic 3D setup as test_fused_generic_label_3d_forward
    B, Z, Y, X = 1, 12, 14, 16
    num_classes = 4
    label_int = torch.randint(0, num_classes, (1, 1, Z, Y, X), generator=rng).long().cuda()
    onehot = _label_to_onehot_3d(label_int, background_label=-1, max_label=num_classes - 1)
    onehot = onehot.cuda().float()

    out_shape = (8, 10, 12)
    align_corners = True
    padding_mode = "zeros"

    aff = torch.linalg.matrix_exp(0.02 * torch.randn(3, 3, generator=rng)).cuda()
    aff = torch.cat([aff, 0.01 * torch.randn(3, 1, generator=rng).cuda()], dim=1)[None]
    affine = aff.detach().requires_grad_(True)
    grid = (0.5 * torch.randn(1, *out_shape, 3, generator=rng).cuda()).detach().requires_grad_(True)
    grid_affine = torch.eye(3, 3)[None].cuda()

    baseline_labels, baseline_weights = _baseline_generic_label_3d(
        onehot, affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners, padding_mode=padding_mode,
    )
    out_labels, out_weights = fused_grid_sampler_3d_generic_label(
        label_int.float(), affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners,
        padding_mode=padding_mode, return_probs=True,
    )
    label_agree_ratio = (out_labels.long() == baseline_labels.long()).float().mean().item()
    rel_weights = (torch.abs(out_weights - baseline_weights) / (1e-6 + torch.abs(baseline_weights))).mean().item()
    assert label_agree_ratio > 0.99, f"Label agreement too low: {label_agree_ratio}"
    assert rel_weights < 5e-3

    loss_baseline = baseline_weights.sum()
    loss_baseline.backward()
    ga_b, gg_b = affine.grad.clone(), grid.grad.clone()
    affine.grad = None
    grid.grad = None
    out_weights.sum().backward()
    rel_affine = (torch.abs(affine.grad - ga_b) / (1e-6 + torch.abs(ga_b))).mean().item()
    rel_grid = (torch.abs(grid.grad - gg_b) / (1e-6 + torch.abs(gg_b))).mean().item()
    assert rel_affine < 1e-4
    assert rel_grid < 1e-4


# --------------- 2D tests ---------------


@pytest.mark.parametrize("use_affine", [True, False])
@pytest.mark.parametrize("use_grid", [True, False])
@pytest.mark.parametrize("is_displacement", [True, False])
def test_fused_generic_label_2d_forward(use_affine, use_grid, is_displacement):
    if not use_affine and not use_grid:
        pytest.skip("need at least affine or grid")
    rng = torch.Generator()
    rng.manual_seed(SEED)
    B, H, W = 1, 24, 32
    num_classes = 3
    label_int = torch.randint(0, num_classes, (1, 1, H, W), generator=rng).long().cuda()
    onehot = _label_to_onehot_2d(label_int, background_label=-1, max_label=num_classes - 1)
    onehot = onehot.cuda().float()

    out_shape = (16, 20)
    align_corners = True
    padding_mode = "zeros"

    affine = None
    grid = None
    grid_affine = None
    if use_affine:
        aff = torch.linalg.matrix_exp(0.02 * torch.randn(2, 2, generator=rng)).cuda()
        aff = torch.cat([aff, 0.01 * torch.randn(2, 1, generator=rng).cuda()], dim=1)[None]
        affine = aff.detach().requires_grad_(True)
    if use_grid:
        if is_displacement:
            grid = (0.5 * torch.randn(1, *out_shape, 2, generator=rng).cuda()).detach().requires_grad_(True)
        else:
            base = F.affine_grid(torch.eye(2, 3)[None].cuda(), (1, 1, *out_shape), align_corners=align_corners)
            grid = (base + 0.02 * torch.randn(1, *out_shape, 2, generator=rng).cuda()).detach().requires_grad_(True)
        grid_affine = torch.eye(2, 2)[None].cuda()

    baseline_labels, baseline_weights = _baseline_generic_label_2d(
        onehot, affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=is_displacement,
        align_corners=align_corners, padding_mode=padding_mode,
    )
    out_labels, out_weights = fused_grid_sampler_2d_generic_label(
        label_int.float(), affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=is_displacement,
        align_corners=align_corners, padding_mode=padding_mode,
        return_probs=True, background_label=None,
    )
    assert out_labels.shape == baseline_labels.shape
    assert out_weights.shape == baseline_weights.shape
    label_agree_ratio = (out_labels.long() == baseline_labels.long()).float().mean().item()
    rel_weights = (torch.abs(out_weights - baseline_weights) / (1e-6 + torch.abs(baseline_weights))).mean().item()
    logger.info(f"2D forward use_affine={use_affine} use_grid={use_grid} is_disp={is_displacement} label_agree_ratio={label_agree_ratio} rel_weights={rel_weights}")
    assert label_agree_ratio > 0.99, f"Label agreement too low: {label_agree_ratio}"
    assert rel_weights < 5e-3


def test_fused_generic_label_2d_backward():
    """2D backward: same input as test_fused_generic_label_2d_forward."""
    rng = torch.Generator()
    rng.manual_seed(SEED)

    # Same synthetic 2D setup as test_fused_generic_label_2d_forward
    B, H, W = 1, 24, 32
    num_classes = 3
    label_int = torch.randint(0, num_classes, (1, 1, H, W), generator=rng).long().cuda()
    onehot = _label_to_onehot_2d(label_int, background_label=-1, max_label=num_classes - 1)
    onehot = onehot.cuda().float()

    out_shape = (16, 20)
    align_corners = True
    padding_mode = "zeros"

    aff = torch.linalg.matrix_exp(0.02 * torch.randn(2, 2, generator=rng)).cuda()
    aff = torch.cat([aff, 0.01 * torch.randn(2, 1, generator=rng).cuda()], dim=1)[None]
    affine = aff.detach().requires_grad_(True)
    grid = (0.5 * torch.randn(1, *out_shape, 2, generator=rng).cuda()).detach().requires_grad_(True)
    grid_affine = torch.eye(2, 2)[None].cuda()

    baseline_labels, baseline_weights = _baseline_generic_label_2d(
        onehot, affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners, padding_mode=padding_mode,
    )
    loss_baseline = baseline_weights.sum()
    loss_baseline.backward()
    grad_affine_baseline = affine.grad.clone() if affine.grad is not None else None
    grad_grid_baseline = grid.grad.clone() if grid.grad is not None else None
    affine.grad = None
    grid.grad = None

    out_labels, out_weights = fused_grid_sampler_2d_generic_label(
        label_int.float(), affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners,
        padding_mode=padding_mode, return_probs=True,
    )
    loss_fused = out_weights.sum()
    loss_fused.backward()

    if grad_affine_baseline is not None and affine.grad is not None:
        rel_affine = (torch.abs(affine.grad - grad_affine_baseline) / (1e-6 + torch.abs(grad_affine_baseline))).mean().item()
        assert rel_affine < 1e-4
    if grad_grid_baseline is not None and grid.grad is not None:
        rel_grid = (torch.abs(grid.grad - grad_grid_baseline) / (1e-6 + torch.abs(grad_grid_baseline))).mean().item()
        assert rel_grid < 1e-4


def test_fused_generic_label_2d_from_data_dir():
    """2D forward/backward with same synthetic input as test_fused_generic_label_2d_forward."""
    rng = torch.Generator()
    rng.manual_seed(SEED)

    # Same synthetic 2D setup as test_fused_generic_label_2d_forward
    B, H, W = 1, 24, 32
    num_classes = 3
    label_int = torch.randint(0, num_classes, (1, 1, H, W), generator=rng).long().cuda()
    onehot = _label_to_onehot_2d(label_int, background_label=-1, max_label=num_classes - 1)
    onehot = onehot.cuda().float()

    out_shape = (16, 20)
    align_corners = True
    padding_mode = "zeros"

    aff = torch.linalg.matrix_exp(0.02 * torch.randn(2, 2, generator=rng)).cuda()
    aff = torch.cat([aff, 0.01 * torch.randn(2, 1, generator=rng).cuda()], dim=1)[None]
    affine = aff.detach().requires_grad_(True)
    grid = (0.5 * torch.randn(1, *out_shape, 2, generator=rng).cuda()).detach().requires_grad_(True)
    grid_affine = torch.eye(2, 2)[None].cuda()

    baseline_labels, baseline_weights = _baseline_generic_label_2d(
        onehot, affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners, padding_mode=padding_mode,
    )
    out_labels, out_weights = fused_grid_sampler_2d_generic_label(
        label_int.float(), affine=affine, grid=grid, grid_affine=grid_affine,
        out_shape=out_shape, is_displacement=True, align_corners=align_corners,
        padding_mode=padding_mode, return_probs=True,
    )
    label_agree_ratio = (out_labels.long() == baseline_labels.long()).float().mean().item()
    rel_weights = (torch.abs(out_weights - baseline_weights) / (1e-6 + torch.abs(baseline_weights))).mean().item()
    assert label_agree_ratio > 0.99, f"Label agreement too low: {label_agree_ratio}"
    assert rel_weights < 5e-3

    loss_baseline = baseline_weights.sum()
    loss_baseline.backward()
    ga_b, gg_b = affine.grad.clone(), grid.grad.clone()
    affine.grad = None
    grid.grad = None
    out_weights.sum().backward()
    rel_affine = (torch.abs(affine.grad - ga_b) / (1e-6 + torch.abs(ga_b))).mean().item()
    rel_grid = (torch.abs(grid.grad - gg_b) / (1e-6 + torch.abs(gg_b))).mean().item()
    assert rel_affine < 1e-4
    assert rel_grid < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
