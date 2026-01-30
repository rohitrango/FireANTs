"""Tests for keypoint fidelity: physical consistency and registration landmark error.

- Helpers create 2D square / 3D cube images (grid + threshold) and return keypoints in pixel and torch space.
- Physical-consistency tests assert as_physical_coordinates() match for pixel vs torch keypoints.
- Registration tests run Affine/Greedy with fusedcc (3D) or mse (2D; fusedcc is 3D-only), then assert
  landmark distance after registration is lower than before, and log the scores.
- Sanity: compute_keypoint_distance(same, same) == 0.
"""

import logging
import pytest
import numpy as np
import SimpleITK as sitk
import torch

from fireants.io.image import Image, BatchedImages
from fireants.io.keypoints import (
    Keypoints,
    BatchedKeypoints,
    compute_keypoint_distance,
)
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _square_image_2d(s=128, r=32, device="cpu"):
    """Create a 2D image of size s x s with a centered square of half-size r (pixels).

    Constraint: s > 2r. Uses a normalized grid and threshold to draw the square.
    Returns image (Image), keypoints_pixel (Keypoints in pixel space), keypoints_torch (Keypoints in torch space).
    """
    assert s > 2 * r, "s must be > 2r"
    # Normalized grid in [-1, 1] (same convention as grid_sample / Image torch space)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, s),
        torch.linspace(-1.0, 1.0, s),
        indexing="ij",
    )
    # Half-size in normalized units: extent r in pixels -> 2*r/(s-1) in [-1,1]
    half_norm = 2.0 * r / (s - 1)
    mask = (
        (torch.abs(grid_x) <= half_norm) & (torch.abs(grid_y) <= half_norm)
    ).float()
    arr = mask.numpy().astype(np.float32)
    y, x = np.where(arr > 0)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    # Corners in pixel (x, y): (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)
    corners_pixel = torch.tensor(
        [
            [float(xmin), float(ymin)],
            [float(xmax), float(ymin)],
            [float(xmax), float(ymax)],
            [float(xmin), float(ymax)],
        ],
        dtype=torch.float32,
        device=device,
    )
    corners_torch = (2.0 * corners_pixel / (s - 1)) - 1.0

    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing([1.0, 1.0])
    itk.SetOrigin([0.0, 0.0])
    img = Image(itk, device=device)

    kp_pixel = Keypoints(corners_pixel, img, device=device, space="pixel")
    kp_torch = Keypoints(corners_torch, img, device=device, space="torch")
    return img, kp_pixel, kp_torch


def _cube_image_3d(s=128, r=32, device="cpu"):
    """Create a 3D image of size s x s x s with a centered cube of half-size r (pixels).

    Constraint: s > 2r. Returns image, keypoints_pixel, keypoints_torch.
    """
    assert s > 2 * r, "s must be > 2r"
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, s),
        torch.linspace(-1.0, 1.0, s),
        torch.linspace(-1.0, 1.0, s),
        indexing="ij",
    )
    half_norm = 2.0 * r / (s - 1)
    mask = (
        (torch.abs(grid_x) <= half_norm)
        & (torch.abs(grid_y) <= half_norm)
        & (torch.abs(grid_z) <= half_norm)
    ).float()
    arr = mask.numpy().astype(np.float32)
    z, y, x = np.where(arr > 0)
    zmin, zmax = z.min(), z.max()
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    # Corners in pixel (x, y, z): 4 on zmin face, 4 on zmax face
    corners_pixel = torch.tensor(
        [
            [float(xmin), float(ymin), float(zmin)],
            [float(xmax), float(ymin), float(zmin)],
            [float(xmax), float(ymax), float(zmin)],
            [float(xmin), float(ymax), float(zmin)],
            [float(xmin), float(ymin), float(zmax)],
            [float(xmax), float(ymin), float(zmax)],
            [float(xmax), float(ymax), float(zmax)],
            [float(xmin), float(ymax), float(zmax)],
        ],
        dtype=torch.float32,
        device=device,
    )
    corners_torch = (2.0 * corners_pixel / (s - 1)) - 1.0

    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing([1.0, 1.0, 1.0])
    itk.SetOrigin([0.0, 0.0, 0.0])
    img = Image(itk, device=device)

    kp_pixel = Keypoints(corners_pixel, img, device=device, space="pixel")
    kp_torch = Keypoints(corners_torch, img, device=device, space="torch")
    return img, kp_pixel, kp_torch


# -----------------------------------------------------------------------------
# Test: physical coordinates match for pixel vs torch keypoints
# -----------------------------------------------------------------------------

def test_keypoint_physical_consistency_2d():
    """as_physical_coordinates() of pixel and torch keypoints (same geometry) should match."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img, kp_pixel, kp_torch = _square_image_2d(s=128, r=32, device=device)
    phy_pixel = kp_pixel.as_physical_coordinates()
    phy_torch = kp_torch.as_physical_coordinates()
    torch.testing.assert_close(phy_pixel, phy_torch, rtol=1e-5, atol=1e-5)


def test_keypoint_physical_consistency_3d():
    """as_physical_coordinates() of pixel and torch keypoints (3D cube) should match."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img, kp_pixel, kp_torch = _cube_image_3d(s=128, r=32, device=device)
    phy_pixel = kp_pixel.as_physical_coordinates()
    phy_torch = kp_torch.as_physical_coordinates()
    torch.testing.assert_close(phy_pixel, phy_torch, rtol=1e-5, atol=1e-5)

# -----------------------------------------------------------------------------
# Sanity: compute_keypoint_distance(same, same) == 0
# -----------------------------------------------------------------------------

def test_compute_keypoint_distance_sanity_same_keypoints():
    """Distance between the same set of keypoints should be zero."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img, kp_pixel, _ = _square_image_2d(s=128, r=32, device=device)
    kp_dup = Keypoints(kp_pixel.keypoints.clone(), img, device=device, space="pixel")
    dist = compute_keypoint_distance(kp_pixel, kp_dup, space="physical", reduction="mean")
    assert dist.numel() == 1
    torch.testing.assert_close(dist.squeeze(), torch.tensor(0.0, device=device), rtol=0, atol=1e-6)


# -----------------------------------------------------------------------------
# Affine registration + keypoint distance (2D)
# -----------------------------------------------------------------------------


def test_keypoint_fidelity_affine_2d():
    """After AffineRegistration (fusedcc), transformed fixed keypoints should be closer to moving keypoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 128
    r_fixed, r_moving = 32, 24
    img_fixed, _, kp_fixed_torch = _square_image_2d(s=s, r=r_fixed, device=device)
    img_moving, _, kp_moving_torch = _square_image_2d(s=s, r=r_moving, device=device)

    fixed_batch = BatchedImages([img_fixed])
    moving_batch = BatchedImages([img_moving])
    fixed_kp_batch = BatchedKeypoints([kp_fixed_torch])
    moving_kp_batch = BatchedKeypoints([kp_moving_torch])

    initial_dist = compute_keypoint_distance(
        fixed_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()
    # 2D: corresponding corners are (r1-r2) apart along each axis -> distance sqrt(2)*(r1-r2)
    expected_initial_2d = np.sqrt(2) * (r_fixed - r_moving)
    np.testing.assert_allclose(initial_dist, expected_initial_2d, rtol=0.05, atol=0.5)

    # fusedcc is 3D-only; use mse for 2D synthetic squares (simpler, more stable)
    reg = AffineRegistration(
        scales=[4, 2, 1],
        iterations=[200, 100, 50],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="mse",
        optimizer_lr=1e-2,
        progress_bar=False,
    )
    reg.optimize()

    moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
    final_dist = compute_keypoint_distance(
        moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()

    logger.info(
        "keypoint_fidelity_affine_2d: initial_dist=%.4f, final_dist=%.4f",
        initial_dist,
        final_dist,
    )
    # After registration, predicted moving positions should be closer to actual moving keypoints
    assert final_dist < initial_dist, (
        f"Expected final keypoint distance ({final_dist}) < initial ({initial_dist})"
    )


# -----------------------------------------------------------------------------
# Greedy registration + keypoint distance (2D)
# -----------------------------------------------------------------------------


def test_keypoint_fidelity_greedy_2d():
    """After GreedyRegistration (fusedcc), transformed fixed keypoints should be closer to moving keypoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 128
    r_fixed, r_moving = 32, 24
    img_fixed, _, kp_fixed_torch = _square_image_2d(s=s, r=r_fixed, device=device)
    img_moving, _, kp_moving_torch = _square_image_2d(s=s, r=r_moving, device=device)

    fixed_batch = BatchedImages([img_fixed])
    moving_batch = BatchedImages([img_moving])
    fixed_kp_batch = BatchedKeypoints([kp_fixed_torch])
    moving_kp_batch = BatchedKeypoints([kp_moving_torch])

    initial_dist = compute_keypoint_distance(
        fixed_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()
    expected_initial_2d = np.sqrt(2) * (r_fixed - r_moving)
    np.testing.assert_allclose(initial_dist, expected_initial_2d, rtol=0.05, atol=0.5)

    # fusedcc is 3D-only; use mse for 2D synthetic squares
    reg = GreedyRegistration(
        scales=[2, 1],
        iterations=[120, 60],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="mse",
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        optimizer_lr=0.3,
        progress_bar=False,
    )
    reg.optimize()

    moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
    final_dist = compute_keypoint_distance(
        moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()

    logger.info(
        "keypoint_fidelity_greedy_2d: initial_dist=%.4f, final_dist=%.4f",
        initial_dist,
        final_dist,
    )
    assert final_dist < initial_dist, (
        f"Expected final keypoint distance ({final_dist}) < initial ({initial_dist})"
    )


# -----------------------------------------------------------------------------
# Affine registration + keypoint distance (3D)
# -----------------------------------------------------------------------------


def test_keypoint_fidelity_affine_3d():
    """After AffineRegistration (fusedcc) on 3D cubes, keypoint distance should decrease."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 128
    r_fixed, r_moving = 32, 24
    img_fixed, _, kp_fixed_torch = _cube_image_3d(s=s, r=r_fixed, device=device)
    img_moving, _, kp_moving_torch = _cube_image_3d(s=s, r=r_moving, device=device)

    fixed_batch = BatchedImages([img_fixed])
    moving_batch = BatchedImages([img_moving])
    fixed_kp_batch = BatchedKeypoints([kp_fixed_torch])
    moving_kp_batch = BatchedKeypoints([kp_moving_torch])

    initial_dist = compute_keypoint_distance(
        fixed_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()
    # 3D: corresponding corners (r1-r2) along each axis -> distance sqrt(3)*(r1-r2)
    expected_initial_3d = np.sqrt(3) * (r_fixed - r_moving)
    np.testing.assert_allclose(initial_dist, expected_initial_3d, rtol=0.05, atol=0.5)

    reg = AffineRegistration(
        scales=[4, 2, 1],
        iterations=[150, 80, 40],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="fusedcc",
        cc_kernel_size=5,
        progress_bar=False,
    )
    reg.optimize()

    moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
    final_dist = compute_keypoint_distance(
        moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()

    logger.info(
        "keypoint_fidelity_affine_3d: initial_dist=%.4f, final_dist=%.4f",
        initial_dist,
        final_dist,
    )
    assert final_dist < initial_dist, (
        f"Expected final keypoint distance ({final_dist}) < initial ({initial_dist})"
    )


# -----------------------------------------------------------------------------
# Greedy registration + keypoint distance (3D)
# -----------------------------------------------------------------------------


def test_keypoint_fidelity_greedy_3d():
    """After GreedyRegistration (fusedcc) on 3D cubes, keypoint distance should decrease."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 128
    r_fixed, r_moving = 32, 24
    img_fixed, _, kp_fixed_torch = _cube_image_3d(s=s, r=r_fixed, device=device)
    img_moving, _, kp_moving_torch = _cube_image_3d(s=s, r=r_moving, device=device)

    fixed_batch = BatchedImages([img_fixed])
    moving_batch = BatchedImages([img_moving])
    fixed_kp_batch = BatchedKeypoints([kp_fixed_torch])
    moving_kp_batch = BatchedKeypoints([kp_moving_torch])

    initial_dist = compute_keypoint_distance(
        fixed_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()
    expected_initial_3d = np.sqrt(3) * (r_fixed - r_moving)
    np.testing.assert_allclose(initial_dist, expected_initial_3d, rtol=0.05, atol=0.5)

    reg = GreedyRegistration(
        scales=[4, 2, 1],
        iterations=[100, 50, 25],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="fusedcc",
        cc_kernel_size=5,
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        progress_bar=False,
    )
    reg.optimize()

    moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
    final_dist = compute_keypoint_distance(
        moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
    ).item()

    logger.info(
        "keypoint_fidelity_greedy_3d: initial_dist=%.4f, final_dist=%.4f",
        initial_dist,
        final_dist,
    )
    assert final_dist < initial_dist, (
        f"Expected final keypoint distance ({final_dist}) < initial ({initial_dist})"
    )
