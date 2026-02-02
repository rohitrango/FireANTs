"""
Generate images for the "Moments Matching with Scaling" How To tutorial.

Run from the project root with the fireants conda env, e.g.:
    conda activate fireants
    python docs/scripts/generate_moments_scaling_images.py

Uses the same test data as tests/test_moments_scaling.py.

Writes to docs/docs/assets/howto/:
  - moments_scaling_input.png
  - moments_scaling_without.png
  - moments_scaling_with.png
  - moments_scaling_comparison.png
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
from torch.nn import functional as F

# Allow importing fireants when run from project root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireants.io.image import BatchedImages, Image
from fireants.registration.moments import MomentsRegistration


OUT_DIR = REPO_ROOT / "docs" / "docs" / "assets" / "howto"
TEST_DATA_DIR = REPO_ROOT / "tests" / "test_data" / "moments_scaling"


def load_test_data():
    """
    Load the actual test data used in tests/test_moments_scaling.py.
    
    These are real images demonstrating scale differences:
    - Fixed: fixed_rotated.nii.gz
    - Moving: moving_left_hemi.nii.gz (left hemisphere with different scale)
    
    Returns:
        tuple: (fixed_image, moving_image) as FireANTs Image objects
    """
    fixed_path = TEST_DATA_DIR / "fixed_rotated.nii.gz"
    moving_path = TEST_DATA_DIR / "moving_left_hemi.nii.gz"
    
    if not fixed_path.exists() or not moving_path.exists():
        raise FileNotFoundError(
            f"Test data not found in {TEST_DATA_DIR}. "
            f"Expected files: fixed_rotated.nii.gz, moving_left_hemi.nii.gz"
        )
    
    # Load images
    fixed_img = Image.load_file(str(fixed_path), dtype=torch.float32)
    moving_img = Image.load_file(str(moving_path), dtype=torch.float32)
    
    return fixed_img, moving_img


def plot_middle_slices(fixed, moving, moved, title_suffix=""):
    """
    Plot middle slices for all three views (axial, coronal, sagittal).
    Returns the figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Get middle indices
    mid_x = fixed.shape[0] // 2
    mid_y = fixed.shape[1] // 2
    mid_z = fixed.shape[2] // 2
    
    # Normalize images for display
    def normalize(img):
        if np.any(img > 0):
            vmin, vmax = np.percentile(img[img > 0], [1, 99])
        else:
            vmin, vmax = 0, 1
        return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    titles = ['Fixed', 'Moving', f'Moved{title_suffix}']
    views = ['Sagittal', 'Coronal', 'Axial']
    
    images = [fixed, moving, moved]
    
    for col, (title, img) in enumerate(zip(titles, images)):
        img_norm = normalize(img)
        
        # Sagittal view (YZ plane, slice along X)
        axes[0, col].imshow(img_norm[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'{title} - {views[0]}', fontsize=11)
        axes[0, col].axis('off')
        
        # Coronal view (XZ plane, slice along Y)
        axes[1, col].imshow(img_norm[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title(f'{title} - {views[1]}', fontsize=11)
        axes[1, col].axis('off')
        
        # Axial view (XY plane, slice along Z)
        axes[2, col].imshow(img_norm[:, :, mid_z].T, cmap='gray', origin='lower')
        axes[2, col].set_title(f'{title} - {views[2]}', fontsize=11)
        axes[2, col].axis('off')
    
    plt.tight_layout()
    return fig


def run_moments_registration(fixed_images, moving_images, perform_scaling):
    """Run moments registration and return moved image array."""
    moments = MomentsRegistration(
        scale=4,  # downscale for faster computation
        fixed_images=fixed_images,
        moving_images=moving_images,
        moments=2,  # use 2nd order moments for rotation
        orientation='rot',  # try rotations
        loss_type='cc',
        perform_scaling=perform_scaling,
    )
    moments.optimize()
    
    # Get warp parameters and apply transformation
    warp_params = moments.get_warp_parameters(fixed_images, moving_images)
    affine_mat = warp_params['affine']
    out_shape = warp_params['out_shape']
    
    # Generate grid and warp moving image
    grid = F.affine_grid(affine_mat, out_shape, align_corners=True)
    moved_array = F.grid_sample(
        moving_images(), 
        grid.to(moving_images().dtype), 
        mode='bilinear', 
        align_corners=True
    )
    
    return moved_array, moments


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading test data...")
    fixed_img, moving_img = load_test_data()
    
    fixed_images = BatchedImages([fixed_img])
    moving_images = BatchedImages([moving_img])
    
    # Get numpy arrays for visualization
    fixed_np = fixed_images()[0, 0].cpu().numpy()
    moving_np = moving_images()[0, 0].cpu().numpy()
    
    print(f"Fixed image shape: {fixed_images.shape}")
    print(f"Moving image shape: {moving_images.shape}")
    
    # 1. Input overview: fixed and moving (middle slices)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    mid = fixed_np.shape[0] // 2
    
    # Fixed image
    axes[0, 0].imshow(fixed_np[mid, :, :].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Fixed - Sagittal')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(fixed_np[:, mid, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Fixed - Coronal')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(fixed_np[:, :, mid].T, cmap='gray', origin='lower')
    axes[0, 2].set_title('Fixed - Axial')
    axes[0, 2].axis('off')
    
    # Moving image
    axes[1, 0].imshow(moving_np[mid, :, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title('Moving - Sagittal')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(moving_np[:, mid, :].T, cmap='gray', origin='lower')
    axes[1, 1].set_title('Moving - Coronal')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(moving_np[:, :, mid].T, cmap='gray', origin='lower')
    axes[1, 2].set_title('Moving - Axial')
    axes[1, 2].axis('off')
    
    plt.suptitle('Input Images: Fixed (top) has different scale than Moving (bottom)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "moments_scaling_input.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'moments_scaling_input.png'}")
    
    # 2. Registration WITHOUT scaling
    print("\nRunning moments registration WITHOUT scaling...")
    moved_no_scale, reg_no_scale = run_moments_registration(fixed_images, moving_images, perform_scaling=False)
    moved_no_scale_np = moved_no_scale[0, 0].cpu().numpy()
    
    fig = plot_middle_slices(fixed_np, moving_np, moved_no_scale_np, " (no scaling)")
    fig.suptitle('Moments Registration WITHOUT Scaling', fontsize=14, y=1.02)
    plt.savefig(OUT_DIR / "moments_scaling_without.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'moments_scaling_without.png'}")
    
    # 3. Registration WITH scaling
    print("\nRunning moments registration WITH scaling...")
    moved_with_scale, reg_with_scale = run_moments_registration(fixed_images, moving_images, perform_scaling=True)
    moved_with_scale_np = moved_with_scale[0, 0].cpu().numpy()
    
    fig = plot_middle_slices(fixed_np, moving_np, moved_with_scale_np, " (with scaling)")
    fig.suptitle('Moments Registration WITH Scaling', fontsize=14, y=1.02)
    plt.savefig(OUT_DIR / "moments_scaling_with.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'moments_scaling_with.png'}")
    
    # 4. Side-by-side comparison of moved images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    mid = fixed_np.shape[0] // 2
    
    # Without scaling
    axes[0, 0].imshow(moved_no_scale_np[mid, :, :].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Without Scaling - Sagittal')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(moved_no_scale_np[:, mid, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Without Scaling - Coronal')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(moved_no_scale_np[:, :, mid].T, cmap='gray', origin='lower')
    axes[0, 2].set_title('Without Scaling - Axial')
    axes[0, 2].axis('off')
    
    # With scaling
    axes[1, 0].imshow(moved_with_scale_np[mid, :, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title('With Scaling - Sagittal')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(moved_with_scale_np[:, mid, :].T, cmap='gray', origin='lower')
    axes[1, 1].set_title('With Scaling - Coronal')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(moved_with_scale_np[:, :, mid].T, cmap='gray', origin='lower')
    axes[1, 2].set_title('With Scaling - Axial')
    axes[1, 2].axis('off')
    
    # Compute losses for comparison
    loss_no_scale = reg_no_scale.loss_fn(moved_no_scale, fixed_images()).mean().item()
    loss_with_scale = reg_with_scale.loss_fn(moved_with_scale, fixed_images()).mean().item()
    
    plt.suptitle(
        f'Comparison: Without Scaling (top, loss={loss_no_scale:.4f}) vs With Scaling (bottom, loss={loss_with_scale:.4f})',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "moments_scaling_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'moments_scaling_comparison.png'}")
    
    print(f"\nLoss without scaling: {loss_no_scale:.6f}")
    print(f"Loss with scaling: {loss_with_scale:.6f}")
    print(f"Improvement: {((loss_no_scale - loss_with_scale) / abs(loss_no_scale)) * 100:.1f}%")
    
    print(f"\nAll images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
