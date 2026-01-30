"""
Generate figures for the Keypoint Fidelity How-To documentation.

Uses the same geometry as tests/test_keypoint_fidelity.py: centered squares of half-size r,
with corners derived from np.where(arr > 0). Blue = fixed keypoints, red = moving keypoints.

Produces:
  1. Side-by-side view of fixed and moving squares (keypoint_fidelity_squares.png)
  2. Overlaid view showing both squares and keypoints together (keypoint_fidelity_overlay.png)

Run from the project root with the fireants conda env, e.g.:
    conda activate fireants
    python docs/scripts/generate_keypoint_fidelity_plot.py

Writes to docs/docs/assets/howto/
See: docs/docs/howto/keypoint-fidelity.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "docs" / "assets" / "howto"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _square_image_and_corners_2d(s: int = 128, r: int = 32):
    """Build s×s binary image with centered square of half-size r; return arr and corners (x, y) in pixel."""
    assert s > 2 * r
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, s),
        torch.linspace(-1.0, 1.0, s),
        indexing="ij",
    )
    half_norm = 2.0 * r / (s - 1)
    mask = (
        (torch.abs(grid_x) <= half_norm) & (torch.abs(grid_y) <= half_norm)
    ).float()
    arr = mask.numpy().astype(np.float32)
    y, x = np.where(arr > 0)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    corners = np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=np.float64,
    )
    return arr, corners


def generate_side_by_side(s, r_fixed, r_moving, arr_fixed, corners_fixed, arr_moving, corners_moving):
    """Generate side-by-side view of fixed and moving squares."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, arr, corners, title, color in [
        (axes[0], arr_fixed, corners_fixed, f"Fixed image (r={r_fixed})", "blue"),
        (axes[1], arr_moving, corners_moving, f"Moving image (r={r_moving})", "red"),
    ]:
        ax.imshow(arr, cmap="gray", origin="upper")
        ax.scatter(
            corners[:, 0],
            corners[:, 1],
            c=color,
            s=80,
            edgecolors="white",
            linewidths=2,
            label="keypoints",
            zorder=5,
        )
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.suptitle("Fixed and moving squares with corner keypoints", fontsize=14)
    plt.tight_layout()
    out_path = OUT_DIR / "keypoint_fidelity_squares.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def generate_overlay(s, r_fixed, r_moving, arr_fixed, corners_fixed, arr_moving, corners_moving):
    """Generate overlaid view showing both squares and keypoints together."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    # Create a combined image with fixed in blue channel, moving in red channel
    combined = np.zeros((s, s, 3), dtype=np.float32)
    combined[:, :, 2] = arr_fixed * 0.6   # Blue for fixed
    combined[:, :, 0] = arr_moving * 0.6  # Red for moving
    # Where both overlap, show purple
    overlap = (arr_fixed > 0) & (arr_moving > 0)
    combined[overlap, 0] = 0.6
    combined[overlap, 2] = 0.6
    
    ax.imshow(combined, origin="upper")
    
    # Plot fixed keypoints (blue)
    ax.scatter(
        corners_fixed[:, 0],
        corners_fixed[:, 1],
        c="blue",
        s=100,
        edgecolors="white",
        linewidths=2,
        marker="o",
        zorder=5,
    )
    # Plot moving keypoints (red)
    ax.scatter(
        corners_moving[:, 0],
        corners_moving[:, 1],
        c="red",
        s=100,
        edgecolors="white",
        linewidths=2,
        marker="s",
        zorder=5,
    )
    
    # Draw lines between corresponding keypoints to visualize initial distance
    for i in range(len(corners_fixed)):
        ax.plot(
            [corners_fixed[i, 0], corners_moving[i, 0]],
            [corners_fixed[i, 1], corners_moving[i, 1]],
            "g--",
            linewidth=1.5,
            alpha=0.7,
        )
    
    # Create legend
    fixed_patch = mpatches.Patch(color="blue", label=f"Fixed (r={r_fixed})")
    moving_patch = mpatches.Patch(color="red", label=f"Moving (r={r_moving})")
    ax.legend(handles=[fixed_patch, moving_patch], loc="upper right", fontsize=10)
    
    ax.set_title("Overlaid squares: blue=fixed, red=moving, dashed=initial keypoint distance", fontsize=11)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_aspect("equal")
    
    # Add annotation about expected initial distance
    initial_dist = np.sqrt(2) * (r_fixed - r_moving)
    ax.text(
        0.02, 0.02,
        f"Initial mean distance: √2·(r₁−r₂) = {initial_dist:.2f} px",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    plt.tight_layout()
    out_path = OUT_DIR / "keypoint_fidelity_overlay.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    s = 128
    r_fixed, r_moving = 32, 24
    arr_fixed, corners_fixed = _square_image_and_corners_2d(s, r_fixed)
    arr_moving, corners_moving = _square_image_and_corners_2d(s, r_moving)

    # Generate both figures
    generate_side_by_side(s, r_fixed, r_moving, arr_fixed, corners_fixed, arr_moving, corners_moving)
    generate_overlay(s, r_fixed, r_moving, arr_fixed, corners_fixed, arr_moving, corners_moving)
    
    print("\nAll figures generated successfully!")
    print("See: docs/docs/howto/keypoint-fidelity.md")


if __name__ == "__main__":
    main()
