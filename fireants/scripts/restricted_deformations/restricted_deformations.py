#!/usr/bin/env python3
"""
Run FireANTs deformable registration on reverse phase-encoding pairs from notepad/data.
Uses restrict_deformations based on folder name (AP/PA -> y only, RL/LR -> x only).
Saves fixed, moved, and moving slices with matplotlib.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure fireants is importable when run from notepad or repo root
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
import matplotlib.pyplot as plt
from time import time

from fireants.io import Image, BatchedImages, FakeBatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration


# ---------------------------------------------------------------------------
# Data and restriction mapping
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Subdirectory names that are example/output (skip)
SKIP_SUBDIRS = {"padded_ants_example"}

# Phase-encoding direction -> which dimensions to allow deformation (1 = allow, 0 = restrict).
# ANTs --restrict-deformation 0x1x0 means allow only y (AP/PA distortion). We use the same idea.
# 3D: [x, y, z]. AP/PA = y; RL/LR = x.
def get_restrict_deformations(subdir_name: str) -> list[float]:
    name_upper = subdir_name.upper()
    if "AP" in name_upper or "PA" in name_upper:
        return [0.0, 1.0, 0.0]  # allow deformation only in y
    if "RL" in name_upper or "LR" in name_upper:
        return [1.0, 0.0, 0.0]  # allow deformation only in x
    # default: allow all (no restriction)
    return None


def discover_pairs(data_dir: Path):
    """Yield (subdir_name, fixed_path, moving_path) for each pair directory."""
    if not data_dir.is_dir():
        return
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in SKIP_SUBDIRS:
            continue
        nii = sorted(subdir.glob("*.nii.gz"))
        if len(nii) < 2:
            continue
        # Use first as fixed, second as moving (arbitrary)
        yield subdir.name, str(nii[0]), str(nii[1])


def load_pair_ras(fixed_path: str, moving_path: str, device: str = "cuda"):
    """Load fixed and moving images in RAS orientation; optional winsorize like ANTs."""
    fixed = Image.load_file(
        fixed_path,
        device=device,
        orientation="RAS",
        winsorize=True,
        winsorize_percentile=(0.5, 99.5),
    )
    moving = Image.load_file(
        moving_path,
        device=device,
        orientation="RAS",
        winsorize=True,
        winsorize_percentile=(0.5, 99.5),
    )
    batch_fixed = BatchedImages([fixed])
    batch_moving = BatchedImages([moving])
    return batch_fixed, batch_moving


# ---------------------------------------------------------------------------
# Registration (affine + greedy with restrict_deformations)
# ---------------------------------------------------------------------------

def run_registration(
    batch_fixed: BatchedImages,
    batch_moving: BatchedImages,
    restrict_deformations: list[float] | None,
    scales: list[int] = (8, 4, 2, 1),
    iterations: list[int] = (200, 200, 100, 50),
    cc_kernel_size: int = 7,
    smooth_grad_sigma: float = 1.0,
    smooth_warp_sigma: float = 0.5,
    optimizer: str = "adam",
    optimizer_lr: float = 0.5,
):
    """Run affine then greedy deformable registration. Optionally restrict deformation directions."""
    # Affine
    # affine = AffineRegistration(
    #     list(scales),
    #     list(iterations),
    #     batch_fixed,
    #     batch_moving,
    #     optimizer=optimizer,
    #     optimizer_lr=3e-3,
    #     cc_kernel_size=cc_kernel_size,
    # )
    # t0 = time()
    # affine.optimize()
    # if batch_fixed().is_cuda:
    #     torch.cuda.synchronize()
    # print(f"  Affine done in {time() - t0:.1f}s")

    # Greedy deformable with optional direction restriction
    optimizer_params = {}
    if restrict_deformations is not None:
        optimizer_params["restrict_deformations"] = restrict_deformations

    reg = GreedyRegistration(
        scales=list(scales),
        iterations=list(iterations),
        fixed_images=batch_fixed,
        moving_images=batch_moving,
        cc_kernel_size=cc_kernel_size,
        deformation_type="compositive",
        smooth_grad_sigma=smooth_grad_sigma,
        smooth_warp_sigma=smooth_warp_sigma,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        optimizer_params=optimizer_params,
    )
    t0 = time()
    reg.optimize()
    if batch_fixed().is_cuda:
        torch.cuda.synchronize()
    print(f"  Greedy done in {time() - t0:.1f}s")

    return reg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_slice_figure(
    batch_fixed: BatchedImages,
    batch_moving: BatchedImages,
    moved: torch.Tensor,
    subdir_name: str,
    output_dir: Path,
    slice_idx: int | None = None,
):
    """Save a matplotlib figure with fixed, moved, moving (and optional slice index for 3D)."""
    fixed_np = batch_fixed()[0, 0].detach().cpu().numpy()
    moving_np = batch_moving()[0, 0].detach().cpu().numpy()
    moved_np = moved[0, 0].detach().cpu().numpy()

    ndim = fixed_np.ndim
    if ndim == 3:
        d, h, w = fixed_np.shape
        if slice_idx is None:
            slice_idx = d // 2
        fixed_sl = fixed_np[slice_idx, :, :]
        moved_sl = moved_np[slice_idx, :, :]
        moving_sl = moving_np[slice_idx, :, :]
    else:
        fixed_sl = fixed_np
        moved_sl = moved_np
        moving_sl = moving_np

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(fixed_sl, cmap="gray")
    axes[0].set_title("Fixed")
    axes[0].axis("off")
    axes[0].invert_yaxis()

    axes[1].imshow(moved_sl, cmap="gray")
    axes[1].set_title("Moved (registered)")
    axes[1].axis("off")
    axes[1].invert_yaxis()

    axes[2].imshow(moving_sl, cmap="gray")
    axes[2].set_title("Moving")
    axes[2].axis("off")
    axes[2].invert_yaxis()

    fig.suptitle(f"FireANTs — {subdir_name}", fontsize=14)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subdir_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = DATA_DIR.resolve()
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for subdir_name, fixed_path, moving_path in discover_pairs(data_dir):

        print(f"\n--- {subdir_name} ---")
        print(f"  Fixed:  {Path(fixed_path).name}")
        print(f"  Moving: {Path(moving_path).name}")

        restrict = get_restrict_deformations(subdir_name)
        print(f"  restrict_deformations: {restrict}")

        batch_fixed, batch_moving = load_pair_ras(fixed_path, moving_path, device=device)
        print(f"  batch_fixed: {batch_fixed.shape}")
        print(f"  batch_moving: {batch_moving.shape}")

        reg = run_registration(
            batch_fixed,
            batch_moving,
            restrict_deformations=restrict,
        )

        moved = reg.evaluate(batch_fixed, batch_moving)
        save_slice_figure(batch_fixed, batch_moving, moved, subdir_name, OUTPUT_DIR)

        # Construct output naming
        moved_img_path = OUTPUT_DIR / f"{subdir_name}_moved.nii.gz"
        moved_img_path_partial = OUTPUT_DIR / f"{subdir_name}_moved_partial.nii.gz"
        warp_path = OUTPUT_DIR / f"{subdir_name}_warp.nii.gz"

        # Construct partial warp 
        partial_grid_params = reg.get_partial_warped_parameters(batch_fixed, batch_moving, 0.5)
        moved_partial = reg.evaluate(batch_fixed, batch_moving, moved_coords=partial_grid_params)
        batch_moved_partial = FakeBatchedImages(moved_partial, batch_fixed)
        batch_moved_partial.write_image(str(moved_img_path_partial))
        print(f"  Saved moved image at {moved_img_path_partial}")

        # Save the moved image as NIfTI (use .save_nifti from BatchedImages)
        batch_moved = FakeBatchedImages(moved, batch_fixed)
        batch_moved.write_image(str(moved_img_path))
        print(f"  Saved moved image at {moved_img_path}")

        # Save the deformation field in ANTs-compatible format (using DeformableMixin)
        reg.save_as_ants_transforms([str(warp_path)])
        print(f"  Saved deformation field at {warp_path}")

        print("\nDone.")


if __name__ == "__main__":
    main()
