"""
Generate images for the "Masked losses" How To tutorial.

Run from the project root with the fireants conda env, e.g.:
    conda activate fireants
    python docs/scripts/generate_masked_loss_images.py

Writes to docs/docs/assets/howto/:
  - masked_loss_input.png
  - masked_loss_result_masked.png
  - masked_loss_result_unmasked.png
  - masked_loss_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

# Allow importing fireants when run from project root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones
from fireants.registration.greedy import GreedyRegistration


OUT_DIR = REPO_ROOT / "docs" / "docs" / "assets" / "howto"


def np_to_sitk(np_array: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(np_array)


def make_synthetic_data():
    """Same synthetic setup as tests/notebooks/test_2d_masked_cc.py."""
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

    moving_image = np.vstack([np.hstack([moving_image] * 2)] * 2)
    fixed_image = np.vstack([np.hstack([fixed_image] * 2)] * 2)

    mask = np.zeros_like(moving_image, dtype=np.float32)
    for i in range(moving_image.shape[0]):
        for j in range(moving_image.shape[1]):
            if i + j < moving_image.shape[0]:
                mask[i, j] = 1.0

    return fixed_image, moving_image, mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed_image, moving_image, mask = make_synthetic_data()

    # 1. Input overview: fixed, moving, mask
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(moving_image, cmap="gray")
    ax[0].set_title("Moving image")
    ax[0].axis("off")
    ax[1].imshow(fixed_image, cmap="gray")
    ax[1].set_title("Fixed image")
    ax[1].axis("off")
    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Mask (ROI = 1)")
    ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "masked_loss_input.png", dpi=120, bbox_inches="tight")
    plt.close()

    fixed_sitk = Image(np_to_sitk(fixed_image))
    moving_sitk = Image(np_to_sitk(moving_image))
    mask_sitk = Image(np_to_sitk(mask))

    # 2. Masked registration: both images have mask concatenated; fixed uses all-ones mask
    fixed_mask_all_ones = generate_image_mask_allones(fixed_sitk)
    fixed_masked = apply_mask_to_image(fixed_sitk, fixed_mask_all_ones)
    moving_masked = apply_mask_to_image(moving_sitk, mask_sitk)
    fixed_batch_masked = BatchedImages([fixed_masked])
    moving_batch_masked = BatchedImages([moving_masked])

    reg_masked = GreedyRegistration(
        scales=[2, 1],
        iterations=[100, 50],
        fixed_images=fixed_batch_masked,
        moving_images=moving_batch_masked,
        loss_type="masked_cc",
        optimizer="Adam",
        optimizer_lr=0.5,
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        progress_bar=False,
    )
    reg_masked.optimize()
    moved_masked = reg_masked.evaluate(fixed_batch_masked, moving_batch_masked).detach()

    plt.figure(figsize=(6, 6))
    plt.imshow(moved_masked[0, 0].cpu().numpy(), cmap="gray")
    plt.title("Moved (masked_cc)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "masked_loss_result_masked.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 3. Unmasked registration: no masks, plain cc
    fixed_batch_plain = BatchedImages([fixed_sitk])
    moving_batch_plain = BatchedImages([moving_sitk])

    reg_plain = GreedyRegistration(
        scales=[2, 1],
        iterations=[100, 50],
        fixed_images=fixed_batch_plain,
        moving_images=moving_batch_plain,
        loss_type="cc",
        optimizer="Adam",
        optimizer_lr=0.5,
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        progress_bar=False,
    )
    reg_plain.optimize()
    moved_plain = reg_plain.evaluate(fixed_batch_plain, moving_batch_plain).detach()

    plt.figure(figsize=(6, 6))
    plt.imshow(moved_plain[0, 0].cpu().numpy(), cmap="gray")
    plt.title("Moved (cc, no mask)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "masked_loss_result_unmasked.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 4. Side-by-side comparison: masked vs unmasked result
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(moved_masked[0, 0].cpu().numpy(), cmap="gray")
    ax[0].set_title("With masked_cc (focus on ROI)")
    ax[0].axis("off")
    ax[1].imshow(moved_plain[0, 0].cpu().numpy(), cmap="gray")
    ax[1].set_title("With cc (no mask)")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "masked_loss_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
