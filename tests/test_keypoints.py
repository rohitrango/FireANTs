"""Tests for correctness of Keypoints and BatchedKeypoints in fireants.io.keypoints."""

import pytest
import numpy as np
import SimpleITK as sitk
import torch

from fireants.io.image import Image
from fireants.io.keypoints import (
    Keypoints,
    BatchedKeypoints,
    compute_keypoint_distance,
    _compute_keypoint_distance_helper,
)


def _make_image_2d(shape=(32, 32), spacing=(1.0, 1.0), origin=(0.0, 0.0), device="cpu", rng=None):
    """Create a minimal 2D Image for keypoint tests.

    If rng is provided, spacing and origin are sampled from it for heterogeneous tests:
    spacing in [0.5, 2.0] per axis, origin in [-15.0, 15.0] per axis.
    """
    if rng is not None:
        spacing = tuple(float(rng.uniform(0.5, 2.0)) for _ in range(2))
        origin = tuple(float(rng.uniform(-15.0, 15.0)) for _ in range(2))
    arr = np.zeros(shape, dtype=np.float32)
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing(spacing)
    itk.SetOrigin(origin)
    return Image(itk, device=device)


def _make_image_3d(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu", rng=None):
    """Create a minimal 3D Image for keypoint tests.

    If rng is provided, spacing and origin are sampled from it for heterogeneous tests:
    spacing in [0.5, 2.0] per axis, origin in [-15.0, 15.0] per axis.
    """
    if rng is not None:
        spacing = tuple(float(rng.uniform(0.5, 2.0)) for _ in range(3))
        origin = tuple(float(rng.uniform(-15.0, 15.0)) for _ in range(3))
    arr = np.zeros(shape, dtype=np.float32)
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing(spacing)
    itk.SetOrigin(origin)
    return Image(itk, device=device)


# -----------------------------------------------------------------------------
# Keypoints: construction and coordinate round-trips
# -----------------------------------------------------------------------------

# Seeds for RNG-based spacing/origin to add heterogeneity to roundtrip tests.
_ROUNDTRIP_RNG_SEEDS = [0, 1, 42]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", _ROUNDTRIP_RNG_SEEDS)
def test_keypoints_roundtrip_pixel_physical_pixel_2d(device, seed):
    """Storing keypoints in pixel space, converting to physical and back to pixel should recover the same values."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.RandomState(seed)
    img = _make_image_2d(device=device, rng=rng)
    pixel_coords = torch.tensor([[0.0, 0.0], [10.0, 20.0], [31.0, 31.0]], device=device)
    kp_pixel = Keypoints(pixel_coords, img, device=device, space="pixel")
    physical = kp_pixel.as_physical_coordinates()
    kp_physical = Keypoints(physical, img, device=device, space="physical")
    pixel_again = kp_physical.as_pixel_coordinates()
    torch.testing.assert_close(pixel_coords, pixel_again, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", _ROUNDTRIP_RNG_SEEDS)
def test_keypoints_roundtrip_pixel_torch_pixel_2d(device, seed):
    """Round-trip pixel -> torch -> pixel should recover the same values."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.RandomState(seed)
    img = _make_image_2d(device=device, rng=rng)
    pixel_coords = torch.tensor([[0.0, 0.0], [15.0, 15.0]], device=device)
    kp_pixel = Keypoints(pixel_coords, img, device=device, space="pixel")
    torch_coords = kp_pixel.as_torch_coordinates()
    kp_torch = Keypoints(torch_coords, img, device=device, space="torch")
    pixel_again = kp_torch.as_pixel_coordinates()
    torch.testing.assert_close(pixel_coords, pixel_again, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", _ROUNDTRIP_RNG_SEEDS)
def test_keypoints_roundtrip_physical_torch_physical_2d(device, seed):
    """Round-trip physical -> torch -> physical should recover the same values."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.RandomState(seed)
    img = _make_image_2d(device=device, rng=rng)
    # Use physical-like coords (e.g. origin + some offset in mm)
    physical_coords = torch.tensor([[0.0, 0.0], [5.0, 10.0]], device=device)
    kp_phy = Keypoints(physical_coords, img, device=device, space="physical")
    torch_coords = kp_phy.as_torch_coordinates()
    kp_torch = Keypoints(torch_coords, img, device=device, space="torch")
    physical_again = kp_torch.as_physical_coordinates()
    torch.testing.assert_close(physical_coords, physical_again, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", _ROUNDTRIP_RNG_SEEDS)
def test_keypoints_roundtrip_3d(seed):
    """Round-trips in 3D should be consistent."""
    rng = np.random.RandomState(seed)
    img = _make_image_3d(device="cpu", rng=rng)
    pixel_coords = torch.tensor([[0.0, 0.0, 0.0], [8.0, 8.0, 8.0]], device="cpu")
    kp_pixel = Keypoints(pixel_coords, img, device="cpu", space="pixel")
    physical = kp_pixel.as_physical_coordinates()
    kp_physical = Keypoints(physical, img, device="cpu", space="physical")
    pixel_again = kp_physical.as_pixel_coordinates()
    torch.testing.assert_close(pixel_coords, pixel_again, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Keypoints: consistency of as_* views
# -----------------------------------------------------------------------------

def test_keypoints_three_views_consistent():
    """The same physical point expressed in pixel, physical, and torch should be consistent when converted to a common space."""
    img = _make_image_2d(device="cpu")
    pixel_coords = torch.tensor([[10.0, 20.0]], device="cpu")
    kp_pixel = Keypoints(pixel_coords, img, device="cpu", space="pixel")
    phy_from_pixel = kp_pixel.as_physical_coordinates()
    torch_from_pixel = kp_pixel.as_torch_coordinates()

    kp_physical = Keypoints(phy_from_pixel.clone(), img, device="cpu", space="physical")
    kp_torch = Keypoints(torch_from_pixel.clone(), img, device="cpu", space="torch")

    # All three, when converted to physical, should match
    torch.testing.assert_close(kp_pixel.as_physical_coordinates(), kp_physical.as_physical_coordinates(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kp_pixel.as_physical_coordinates(), kp_torch.as_physical_coordinates(), rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Keypoints: update_keypoints and update_space
# -----------------------------------------------------------------------------

def test_keypoints_update_keypoints():
    """update_keypoints should change stored keypoints and num_keypoints; coordinate views should update accordingly."""
    img = _make_image_2d(device="cpu")
    coords1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    kp = Keypoints(coords1, img, device="cpu", space="pixel")
    assert kp.num_keypoints == 2
    coords2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device="cpu")
    kp.update_keypoints(coords2)
    assert kp.num_keypoints == 3
    torch.testing.assert_close(kp.as_pixel_coordinates(), coords2, rtol=0, atol=0)


def test_keypoints_update_space():
    """update_space only changes the label; converting to the new space should return the same stored tensor."""
    img = _make_image_2d(device="cpu")
    pixel_coords = torch.tensor([[10.0, 20.0]], device="cpu")
    kp = Keypoints(pixel_coords, img, device="cpu", space="pixel")
    kp.update_space("physical")
    # Stored value is still the same numbers; they are now interpreted as physical
    torch.testing.assert_close(kp.keypoints, pixel_coords, rtol=0, atol=0)
    # Converting to physical returns the same (since space is physical)
    torch.testing.assert_close(kp.as_physical_coordinates(), pixel_coords, rtol=0, atol=0)

# -----------------------------------------------------------------------------
# Keypoints: invalid inputs
# -----------------------------------------------------------------------------

def test_keypoints_wrong_ndim_raises():
    """Keypoints must be 2D (N, dims)."""
    img = _make_image_2d(device="cpu")
    with pytest.raises(AssertionError, match="2D tensor"):
        Keypoints(torch.tensor([1.0, 2.0]), img, device="cpu", space="pixel")


def test_keypoints_wrong_dims_raises():
    """Keypoints second dimension must match image dims."""
    img = _make_image_2d(device="cpu")
    with pytest.raises(AssertionError, match="dimensionality"):
        Keypoints(torch.tensor([[1.0, 2.0, 3.0]]), img, device="cpu", space="pixel")


def test_keypoints_invalid_space_raises():
    """Invalid space string should raise."""
    img = _make_image_2d(device="cpu")
    coords = torch.tensor([[1.0, 2.0]], device="cpu")
    with pytest.raises(AssertionError, match="Invalid space"):
        Keypoints(coords, img, device="cpu", space="invalid")


# -----------------------------------------------------------------------------
# BatchedKeypoints
# -----------------------------------------------------------------------------


def test_batched_keypoints_collate():
    """BatchedKeypoints with same number of keypoints per item should return stacked tensors from as_*."""
    img = _make_image_2d(device="cpu")
    kp1 = Keypoints(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu"), img, device="cpu", space="pixel")
    kp2 = Keypoints(torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu"), img, device="cpu", space="pixel")
    batch = BatchedKeypoints([kp1, kp2])
    assert batch.can_collate is True
    px = batch.as_pixel_coordinates()
    assert px.shape == (2, 2, 2)
    torch.testing.assert_close(px[0], kp1.as_pixel_coordinates(), rtol=0, atol=0)
    torch.testing.assert_close(px[1], kp2.as_pixel_coordinates(), rtol=0, atol=0)


def test_batched_keypoints_no_collate():
    """BatchedKeypoints with different num_keypoints should return list from as_*."""
    img = _make_image_2d(device="cpu")
    kp1 = Keypoints(torch.tensor([[1.0, 2.0]], device="cpu"), img, device="cpu", space="pixel")
    kp2 = Keypoints(torch.tensor([[3.0, 4.0], [5.0, 6.0]], device="cpu"), img, device="cpu", space="pixel")
    batch = BatchedKeypoints([kp1, kp2])
    assert batch.can_collate is False
    px = batch.as_pixel_coordinates()
    assert isinstance(px, list)
    assert len(px) == 2
    torch.testing.assert_close(px[0], kp1.as_pixel_coordinates(), rtol=0, atol=0)
    torch.testing.assert_close(px[1], kp2.as_pixel_coordinates(), rtol=0, atol=0)


def test_batched_keypoints_from_tensor_and_metadata():
    """from_tensor_and_metadata then as_* in that space should return the same tensor."""
    img = _make_image_2d(device="cpu")
    kp1 = Keypoints(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu"), img, device="cpu", space="pixel")
    kp2 = Keypoints(torch.tensor([[0.0, 0.0], [1.0, 1.0]], device="cpu"), img, device="cpu", space="pixel")
    batch = BatchedKeypoints([kp1, kp2])
    torch_coords = batch.as_torch_coordinates()
    new_batch = BatchedKeypoints.from_tensor_and_metadata(torch_coords, batch, space="torch")
    torch_again = new_batch.as_torch_coordinates()
    torch.testing.assert_close(torch_coords, torch_again, rtol=1e-5, atol=1e-5)


def test_batched_keypoints_transform_keypoints_batch():
    """transform_keypoints_batch applies (R @ x + t); translation (B, dims) is broadcast over (B, N, dims)."""
    # Identity rotation + translation: (B, dims+1, dims+1) and (B, N, dims)
    B, N, D = 2, 2, 2
    matrix = torch.eye(D + 1).unsqueeze(0).expand(B, -1, -1).clone()
    matrix[:, :D, -1] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    keypoints = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]])  # (B, N, D)
    out = BatchedKeypoints.transform_keypoints_batch(matrix, keypoints)
    # Implementation broadcasts (B, dims) with (B, N, dims): last two dims align, so row n gets added along N
    expected = keypoints + matrix[:, :D, -1].unsqueeze(0).expand(B, N, D)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# -----------------------------------------------------------------------------
# compute_keypoint_distance
# -----------------------------------------------------------------------------


def test_keypoint_distance_same_point_zero():
    """Distance between identical keypoints in any space should be zero."""
    img = _make_image_2d(device="cpu")
    coords = torch.tensor([[10.0, 20.0], [5.0, 5.0]], device="cpu")
    kp1 = Keypoints(coords, img, device="cpu", space="pixel")
    kp2 = Keypoints(coords.clone(), img, device="cpu", space="pixel")
    for space in ("pixel", "physical", "torch"):
        d = _compute_keypoint_distance_helper(kp1, kp2, space=space, reduction="none")
        torch.testing.assert_close(d, torch.zeros_like(d), rtol=0, atol=1e-6)


def test_keypoint_distance_pixel_matches_manual():
    """In pixel space, distance should equal norm(kp1 - kp2)."""
    img = _make_image_2d(device="cpu")
    c1 = torch.tensor([[0.0, 0.0], [10.0, 0.0]], device="cpu")
    c2 = torch.tensor([[3.0, 4.0], [10.0, 10.0]], device="cpu")
    kp1 = Keypoints(c1, img, device="cpu", space="pixel")
    kp2 = Keypoints(c2, img, device="cpu", space="pixel")
    d = _compute_keypoint_distance_helper(kp1, kp2, space="pixel", reduction="none")
    expected = torch.norm(c1 - c2, dim=-1)
    torch.testing.assert_close(d, expected, rtol=1e-5, atol=1e-5)


def test_keypoint_distance_batched():
    """compute_keypoint_distance with BatchedKeypoints should return stacked distances."""
    img = _make_image_2d(device="cpu")
    kp1a = Keypoints(torch.tensor([[0.0, 0.0]], device="cpu"), img, device="cpu", space="pixel")
    kp1b = Keypoints(torch.tensor([[3.0, 4.0]], device="cpu"), img, device="cpu", space="pixel")
    kp2a = Keypoints(torch.tensor([[0.0, 0.0]], device="cpu"), img, device="cpu", space="pixel")
    kp2b = Keypoints(torch.tensor([[0.0, 0.0]], device="cpu"), img, device="cpu", space="pixel")
    batch1 = BatchedKeypoints([kp1a, kp1b])
    batch2 = BatchedKeypoints([kp2a, kp2b])
    dist = compute_keypoint_distance(batch1, batch2, space="pixel", reduction="none")
    assert dist.shape == (2, 1)
    torch.testing.assert_close(dist[0], torch.tensor([0.0]), rtol=0, atol=1e-6)
    torch.testing.assert_close(dist[1], torch.tensor([5.0]), rtol=1e-5, atol=1e-5)
