import pytest
import numpy as np
import SimpleITK as sitk
import torch

from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones
from fireants.registration.greedy import GreedyRegistration

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

def _np_to_sitk(np_array: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(np_array)

def _make_synthetic_masked_pair():
    """Mirror `tests/notebooks/test_2d_masked_cc.py` (but without plotting)."""
    r = 16
    moving_image = np.zeros((64, 64), dtype=np.float32)
    moving_image[32 - r : 32 + r, 32 - r : 32 + r] = 1.0

    fixed_image = np.zeros((64, 64), dtype=np.float32)
    center = 32
    for i in range(64):
        for j in range(64):
            x_rot = (i - center) * np.cos(-np.pi / 4) - (j - center) * np.sin(-np.pi / 4)
            y_rot = (i - center) * np.sin(-np.pi / 4) + (j - center) * np.cos(-np.pi / 4)
            if abs(x_rot) <= r and abs(y_rot) <= r:
                fixed_image[i, j] = 1.0

    # make a 2x2 tiled image to avoid tiny image issues
    moving_image = np.vstack([np.hstack([moving_image] * 2)] * 2)
    fixed_image = np.vstack([np.hstack([fixed_image] * 2)] * 2)

    # diagonal half-plane mask (1 = ROI)
    mask = np.zeros_like(moving_image, dtype=np.float32)
    for i in range(moving_image.shape[0]):
        for j in range(moving_image.shape[1]):
            if i + j < moving_image.shape[0]:
                mask[i, j] = 1.0

    fixed_img = Image(_np_to_sitk(fixed_image))
    moving_img = Image(_np_to_sitk(moving_image))
    mask_img = Image(_np_to_sitk(mask))

    fixed_mask_all_ones = generate_image_mask_allones(fixed_img)
    fixed_img_masked = apply_mask_to_image(fixed_img, fixed_mask_all_ones)
    moving_img_masked = apply_mask_to_image(moving_img, mask_img)

    fixed_batch = BatchedImages([fixed_img_masked])
    moving_batch = BatchedImages([moving_img_masked])
    return fixed_batch, moving_batch


def _dice_score_over_mask(a_bin: torch.Tensor, b_bin: torch.Tensor, mask_bin: torch.Tensor) -> float:
    """Dice score between a and b, restricted to mask==1 voxels."""
    a_m = a_bin[mask_bin > 0.5].reshape(1, -1)
    b_m = b_bin[mask_bin > 0.5].reshape(1, -1)
    return float((1 - dice_loss(a_m, b_m)).item())


@pytest.mark.parametrize("loss_type", ["masked_cc", "masked_mse", "masked_fusedcc"])
def test_greedy_masked_fidelity(loss_type: str):
    fixed_batch, moving_batch = _make_synthetic_masked_pair()

    reg = GreedyRegistration(
        scales=[2, 1],
        iterations=[80, 40],
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

    # channels: [image, mask]
    fixed_img = fixed_batch()[:, 0:1]
    moving_img = moving_batch()[:, 0:1]
    moved_img = moved[:, 0:1]

    # use the *warped* mask to evaluate in fixed space
    moved_mask = (moved[:, 1:2] > 0.5).float()
    # use the original mask to define the complement region (negative mask)
    orig_mask = (moving_batch()[:, 1:2] > 0.5).float()
    neg_orig_mask = 1.0 - orig_mask

    # binarize images for dice (these are synthetic 0/1-ish)
    fixed_bin = (fixed_img > 0.5).float()
    moving_bin = (moving_img > 0.5).float()
    moved_bin = (moved_img > 0.5).float()

    dice_in_mask_before = _dice_score_over_mask(moving_bin, fixed_bin, orig_mask)
    dice_in_mask_after = _dice_score_over_mask(moved_bin, fixed_bin, moved_mask)

    dice_neg_moving_fixed = _dice_score_over_mask(moving_bin, fixed_bin, neg_orig_mask)
    dice_neg_moved_moving = _dice_score_over_mask(moved_bin, moving_bin, neg_orig_mask)

    logger.info(f"dice in mask before: {dice_in_mask_before}")
    logger.info(f"dice in mask after: {dice_in_mask_after}")
    logger.info(f"dice neg moving fixed: {dice_neg_moving_fixed}")
    logger.info(f"dice neg moved moving: {dice_neg_moved_moving}")

    # Inside the (warped) mask region, moved should better match fixed than moving did.
    gap = min(0.10, 0.98 - dice_in_mask_before)
    assert dice_in_mask_after > dice_in_mask_before + gap

    # Outside the mask, moved should stay closer to moving than fixed is to moving.
    assert dice_neg_moved_moving > dice_neg_moving_fixed

