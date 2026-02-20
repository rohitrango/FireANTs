"""
Generate images for the "2D Affine Registration via Subspace (Contour) Matching" How To tutorial.

Run from the project root with the fireants conda env, e.g.:
    conda activate fireants
    python docs/scripts/generate_subspaceaffine_images.py

Uses real histology images and masks from the assets folder.

Writes to docs/docs/assets/subspaceaffine/:
  - subspaceaffine_stain_0.png
  - subspaceaffine_stain_1.png
  - subspaceaffine_stain_2.png
  - subspaceaffine_stain_3.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage

# Allow importing fireants when run from project root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireants.io.image import BatchedImages, Image
from fireants.registration.subspace2daffine import Subspace2DAffineRegistration


ASSETS_DIR = REPO_ROOT / "docs" / "docs" / "assets" / "subspaceaffine"


def load_histo_data():
    """Load histo masks and images from assets folder."""
    histo_masks = sorted(ASSETS_DIR.glob("*_mask.png"))[::-1]
    histo_images = sorted(ASSETS_DIR.glob("*_masked.png"))[::-1]
    
    if len(histo_masks) == 0 or len(histo_images) == 0:
        raise FileNotFoundError(
            f"No histo files found in {ASSETS_DIR}. "
            f"Expected *_mask.png and *_masked.png files."
        )
    
    return histo_masks, histo_images


def load_moving_data():
    """Load moving mask and image."""
    moving_mask_path = ASSETS_DIR / "dwi_image_mask.nii"
    moving_img_path = ASSETS_DIR / "t2_image.nii"
    
    if not moving_mask_path.exists() or not moving_img_path.exists():
        raise FileNotFoundError(
            f"Moving files not found in {ASSETS_DIR}. "
            f"Expected: dwi_image_mask.nii, t2_image.nii"
        )
    
    moving_mask = Image.load_file(str(moving_mask_path), device='cpu')
    moving_img = Image.load_file(str(moving_img_path), device='cpu')
    
    return moving_mask, moving_img


def convert_to_grayscale(image):
    """Convert RGB image to grayscale."""
    arr = image.array[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    gray = np.array(PILImage.fromarray(arr).convert("L")).astype(np.float32)
    # Create a new Image object with grayscale data
    gray_image = Image.load_file(str(ASSETS_DIR / "dwi_image_mask.nii"))  # Use existing file to get structure
    gray_image.array = torch.from_numpy(gray)[None, None].to(image.array.device).float()
    return gray_image


def plot_registration_results(fixed_img, moving_img, moved_img, fixed_mask, moving_mask, moved_mask, stain_name, output_path):
    """Plot fixed, moving, moved images and masks in 2x3 subplot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get numpy arrays
    if isinstance(fixed_img.array, torch.Tensor):
        fixed_img_np = fixed_img.array[0].permute(1, 2, 0).cpu().numpy() / 255.0
    else:
        fixed_img_np = fixed_img.array[0].permute(1, 2, 0).cpu().numpy() / 255.0
    
    moving_img_np = moving_img.array[0, 0].cpu().numpy()
    moved_img_np = moved_img.detach().cpu().numpy()[0, 0]
    
    fixed_mask_np = fixed_mask.array[0, 0].cpu().numpy()
    moving_mask_np = moving_mask.array[0, 0].cpu().numpy()
    moved_mask_np = moved_mask.detach().cpu().numpy()[0, 0]
    
    # Normalize grayscale images for display
    def normalize_grayscale(img):
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        return img
    
    moving_img_np = normalize_grayscale(moving_img_np)
    moved_img_np = normalize_grayscale(moved_img_np)
    
    # Top row: Images
    axes[0, 0].imshow(fixed_img_np)
    axes[0, 0].set_title("Fixed Image", fontsize=18, fontweight='bold')
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(moving_img_np, cmap='gray')
    axes[0, 1].set_title("Moving Image", fontsize=18, fontweight='bold')
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(moved_img_np, cmap='gray')
    axes[0, 2].set_title("Moved Image", fontsize=18, fontweight='bold')
    axes[0, 2].axis("off")
    
    # Bottom row: Masks
    axes[1, 0].imshow(fixed_mask_np, cmap='gray')
    axes[1, 0].set_title("Fixed Mask", fontsize=18, fontweight='bold')
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(moving_mask_np, cmap='gray')
    axes[1, 1].set_title("Moving Mask", fontsize=18, fontweight='bold')
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(moved_mask_np, cmap='gray')
    axes[1, 2].set_title("Moved Mask", fontsize=18, fontweight='bold')
    axes[1, 2].axis("off")
    
    plt.suptitle(f"Subspace2D Affine Registration: {stain_name}", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading histo data...")
    histo_masks, histo_images = load_histo_data()
    print(f"Found {len(histo_masks)} histo masks")
    
    print("Loading moving data...")
    moving_mask, moving_img = load_moving_data()
    moving_mask_batch = BatchedImages([moving_mask])
    moving_img_batch = BatchedImages([moving_img])
    
    print(f"Moving mask shape: {moving_mask_batch.shape}")
    print(f"Moving image shape: {moving_img_batch.shape}")
    
    # Process all stains
    num_stains = len(histo_masks)
    print(f"\nProcessing all {num_stains} stains...")
    
    for i in range(num_stains):
        histo_mask_path = histo_masks[i]
        histo_image_path = histo_images[i]
        
        stain_name = histo_mask_path.stem.replace("_mask", "")
        print(f"\nProcessing {i+1}/{num_stains}: {stain_name}")
        
        # Load fixed mask and image
        histo_mask = Image.load_file(str(histo_mask_path), device='cpu')
        histo_mask.array = (histo_mask.array > 0).float()
        histo_image = Image.load_file(str(histo_image_path), device='cpu')
        
        # Convert to grayscale for registration (use grayscale version for better registration)
        histo_image_grayscale = Image.load_file(str(histo_image_path), device='cpu')
        arr = histo_image_grayscale.array[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gray = np.array(PILImage.fromarray(arr).convert("L")).astype(np.float32)
        histo_image_grayscale.array = torch.from_numpy(gray)[None, None].to(histo_image_grayscale.array.device).float()
        
        histo_mask_batch = BatchedImages([histo_mask])
        histo_image_batch = BatchedImages([histo_image])
        histo_image_grayscale_batch = BatchedImages([histo_image_grayscale])
        
        # Run subspace affine registration
        print(f"  Running Subspace2DAffineRegistration...")
        reg = Subspace2DAffineRegistration(
            fixed_images=histo_mask_batch,
            moving_images=moving_mask_batch,
            orientation="both"
        )
        reg.optimize()
        
        # Get moved image and mask
        # Evaluate moving image into fixed image space
        moved_image = reg.evaluate(histo_mask_batch, moving_img_batch)
        moved_mask = reg.evaluate(histo_mask_batch, moving_mask_batch)
        
        # Plot results
        output_path = ASSETS_DIR / f"subspaceaffine_stain_{i}.png"
        print(f"  Saving plot to {output_path}")
        plot_registration_results(
            histo_image,
            moving_img,
            moved_image,
            histo_mask,
            moving_mask,
            moved_mask,
            stain_name,
            output_path
        )
    
    print(f"\nAll images written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
