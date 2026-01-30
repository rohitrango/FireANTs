import pytest
import numpy as np
import SimpleITK as sitk
import torch
import logging

from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones
from fireants.registration.greedy import GreedyRegistration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

def _np_to_sitk(np_array: np.ndarray) -> sitk.Image:
    # For 3D volumes SimpleITK expects array in (z, y, x) order.
    return sitk.GetImageFromArray(np_array)

def _make_rotated_square_prism(
    size: int, half_width: float, angle_rad: float, center: float
) -> np.ndarray:
    """Binary 3D volume: a square in XY rotated by `angle_rad`, extruded across Z."""
    vol = np.zeros((size, size, size), dtype=np.float32)  # (z, y, x)
    c, s = np.cos(-angle_rad), np.sin(-angle_rad)
    for z in range(size):
        for y in range(size):
            for x in range(size):
                xr = (x - center) * c - (y - center) * s
                yr = (x - center) * s + (y - center) * c
                if abs(xr) <= half_width and abs(yr) <= half_width:
                    vol[z, y, x] = 1.0
    return vol


def _make_axis_aligned_cube_prism(size: int, half_width: int, center: int) -> np.ndarray:
    vol = np.zeros((size, size, size), dtype=np.float32)  # (z, y, x)
    vol[
        center - half_width : center + half_width,
        center - half_width : center + half_width,
        center - half_width : center + half_width,
    ] = 1.0
    return vol


def _make_synthetic_masked_pair_3d():
    """3D analogue of `tests/notebooks/test_2d_masked_cc.py`."""
    size = 32
    center = size // 2
    r = 6

    moving_vol = _make_axis_aligned_cube_prism(size=size, half_width=r, center=center)
    fixed_vol = _make_rotated_square_prism(
        size=size, half_width=float(r), angle_rad=float(np.pi / 4), center=float(center)
    )

    # Tile in XY to avoid tiny-image issues (keep Z modest)
    moving_vol = np.tile(moving_vol, (2, 2, 2))  # (z, 2y, 2x)
    fixed_vol = np.tile(fixed_vol, (2, 2, 2))

    # 3D diagonal half-space mask (1 = ROI)
    Z, Y, X = moving_vol.shape
    mask = np.zeros_like(moving_vol, dtype=np.float32)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if x + y + z < (X + Y + Z) / 3:
                    mask[z, y, x] = 1.0

    fixed_img = Image(_np_to_sitk(fixed_vol))
    moving_img = Image(_np_to_sitk(moving_vol))
    mask_img = Image(_np_to_sitk(mask))

    fixed_mask_all_ones = generate_image_mask_allones(fixed_img)
    fixed_img_masked = apply_mask_to_image(fixed_img, fixed_mask_all_ones)
    moving_img_masked = apply_mask_to_image(moving_img, mask_img)

    fixed_batch = BatchedImages([fixed_img_masked])
    moving_batch = BatchedImages([moving_img_masked])
    return fixed_batch, moving_batch


def _dice_score_over_mask(a_bin: torch.Tensor, b_bin: torch.Tensor, mask_bin: torch.Tensor) -> float:
    # a_m = a_bin * mask_bin
    # b_m = b_bin * mask_bin
    a_m = a_bin[mask_bin > 0.5].reshape(1, -1)
    b_m = b_bin[mask_bin > 0.5].reshape(1, -1)
    return float((1 - dice_loss(a_m, b_m)).item())


@pytest.mark.parametrize("loss_type", ["masked_cc", "masked_mse"])
def test_greedy_masked_fidelity_3d(loss_type: str):
    fixed_batch, moving_batch = _make_synthetic_masked_pair_3d()

    reg = GreedyRegistration(
        scales=[4, 2, 1],
        iterations=[200, 100, 50],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type=loss_type,
        optimizer="Adam",
        optimizer_lr=0.5,
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        progress_bar=False,
    )
    reg.optimize()

    moved = reg.evaluate(fixed_batch, moving_batch).detach()

    fixed_img = fixed_batch()[:, 0:1]
    moving_img = moving_batch()[:, 0:1]
    moved_img = moved[:, 0:1]

    moved_mask = (moved[:, 1:2] > 0.5).float()
    orig_mask = (moving_batch()[:, 1:2] > 0.5).float()
    neg_orig_mask = 1.0 - orig_mask

    fixed_bin = (fixed_img > 0.5).float()
    moving_bin = (moving_img > 0.5).float()
    moved_bin = (moved_img > 0.5).float()

    print(moving_bin.shape, fixed_bin.shape, moved_bin.shape)

    dice_in_mask_before = _dice_score_over_mask(moving_bin, fixed_bin, orig_mask)
    dice_in_mask_after = _dice_score_over_mask(moved_bin, fixed_bin, moved_mask)

    dice_neg_moving_fixed = _dice_score_over_mask(moving_bin, fixed_bin, neg_orig_mask)
    dice_neg_moved_moving = _dice_score_over_mask(moved_bin, moving_bin, neg_orig_mask)

    logger.info(f"[3D] dice in mask before: {dice_in_mask_before}")
    logger.info(f"[3D] dice in mask after: {dice_in_mask_after}")
    logger.info(f"[3D] dice neg moving fixed: {dice_neg_moving_fixed}")
    logger.info(f"[3D] dice neg moved moving: {dice_neg_moved_moving}")

    # Inside the (warped) mask region, moved should better match fixed than moving did.
    assert dice_in_mask_after > 0.95

    # Outside the mask, moved should stay closer to moving than fixed is to moving.
    assert dice_neg_moved_moving > dice_neg_moving_fixed
    assert dice_neg_moved_moving > 0.95

