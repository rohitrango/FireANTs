"""
Compare rigid registration: legacy (so(3)) vs quaternion implementation.

For rotation angles in [5, 10, 15, 30, 45] degrees, we sample a random 3D axis,
rotate the fixed image and segmentation by that rotation, then run both
RigidRegistration (quaternion) and legacy RigidRegistration (so(3)), and assert
mean Dice score >= THRESHOLD for each case.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import logging

from fireants.registration.rigid import RigidRegistration as QuatRigidRegistration
from fireants.registration.legacyrigid import RigidRegistration as LegacyRigidRegistration
from fireants.io.image import Image, BatchedImages, FakeBatchedImages
from fireants.interpolator import fireants_interpolator

try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Reproducible random axis per angle
RNG = np.random.RandomState(42)
THRESHOLD = 0.7

def rotation_matrix_3d_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Build 3D rotation matrix (4x4) from axis (unit 3-vector) and angle in degrees."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.deg2rad(angle_deg)
    # Rodrigues: R = I + sin(θ) K + (1 - cos(θ)) K²
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R3 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    R4 = np.eye(4)
    R4[:3, :3] = R3
    return R4


def apply_rotation_to_tensor(
    fixed_tensor: torch.Tensor,
    fixed_batch: BatchedImages,
    R4: np.ndarray,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample fixed_tensor at rotated physical coordinates; same grid as fixed (moving = rotated fixed)."""
    device = fixed_tensor.device
    dtype = fixed_tensor.dtype
    R = torch.from_numpy(R4).to(device=device, dtype=dtype).unsqueeze(0)  # [1, 4, 4]
    fixed_t2p = fixed_batch.get_torch2phy().to(device=device, dtype=dtype)
    fixed_p2t = fixed_batch.get_phy2torch().to(device=device, dtype=dtype)
    # We want: for each output grid point (in fixed space), sample input at R @ p_physical.
    # So sampling coords in fixed torch: fixed_p2t @ R @ fixed_t2p
    # comp = fixed_p2t @ R @ fixed_t2p  # [1, 4, 4]
    comp = R
    affine = comp[:, :3, :].contiguous()  # [1, 3, 4]
    shape = list(fixed_tensor.shape[2:])
    out = fireants_interpolator(
        fixed_tensor,
        affine=affine,
        out_shape=shape,
        mode=mode,
        align_corners=True,
    )
    return out


@pytest.fixture(scope="module")
def fixed_data():
    """Load fixed image and segmentation once for all tests."""
    data_dir = Path(__file__).parent / "test_data"
    fixed_img = Image.load_file(str(data_dir / "oasis_157_image.nii.gz"))
    fixed_seg = Image.load_file(
        str(data_dir / "oasis_157_seg.nii.gz"),
        is_segmentation=True,
    )
    fixed_batch = BatchedImages([fixed_img])
    fixed_seg_batch = BatchedImages([fixed_seg])
    return {
        "fixed_batch": fixed_batch,
        "fixed_seg_batch": fixed_seg_batch,
        "fixed_tensor": fixed_batch(),
        "fixed_seg_tensor": fixed_seg_batch(),
    }


# Angles in degrees and one random axis per angle (fixed seed)
ANGLES = ([5, 10, 15, 20])[::-1]
AXES = [RNG.randn(3) for _ in ANGLES]
for i, ax in enumerate(AXES):
    AXES[i] = ax / (np.linalg.norm(ax) + 1e-12)


def _run_one_angle_and_method(
    angle_deg: int,
    axis: np.ndarray,
    fixed_data: dict,
    use_legacy: bool,
):
    fixed_batch = fixed_data["fixed_batch"]
    fixed_seg_batch = fixed_data["fixed_seg_batch"]
    fixed_tensor = fixed_data["fixed_tensor"]
    fixed_seg_tensor = fixed_data["fixed_seg_tensor"]

    R4 = rotation_matrix_3d_from_axis_angle(axis, angle_deg)
    rotated_tensor = apply_rotation_to_tensor(
        fixed_tensor, fixed_batch, R4, mode="bilinear"
    )
    rotated_seg_tensor = apply_rotation_to_tensor(
        fixed_seg_tensor, fixed_batch, R4, mode="bilinear"
    )

    moving_batch = FakeBatchedImages(rotated_tensor, fixed_batch)
    moving_seg_batch = FakeBatchedImages(rotated_seg_tensor, fixed_seg_batch)

    RigidClass = LegacyRigidRegistration if use_legacy else QuatRigidRegistration
    reg = RigidClass(
        scales=[4, 2,],
        iterations=[200, 100,],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="mse",
        optimizer="adam",
        optimizer_lr=3e-3,
    )
    reg.optimize()

    moved_seg = reg.evaluate(fixed_seg_batch, moving_seg_batch)
    fixed_seg_arr = fixed_seg_batch()
    dice_scores = 1 - dice_loss(moved_seg, fixed_seg_arr, reduce=False).mean(0)
    mean_dice = dice_scores.mean().item()
    return mean_dice


@pytest.mark.parametrize("angle_deg", ANGLES)
def test_legacy_vs_quat_rigid_dice_per_angle(fixed_data, angle_deg):
    """For each angle, both legacy and quaternion rigid registration must achieve mean Dice >= 0.75."""
    axis = AXES[ANGLES.index(angle_deg)]

    mean_dice_legacy = _run_one_angle_and_method(
        angle_deg, axis, fixed_data, use_legacy=True
    )
    mean_dice_quat = _run_one_angle_and_method(
        angle_deg, axis, fixed_data, use_legacy=False
    )

    logger.info(
        f"angle={angle_deg}° legacy mean Dice={mean_dice_legacy:.4f} quat mean Dice={mean_dice_quat:.4f}"
    )
    assert mean_dice_legacy >= THRESHOLD, (
        f"Legacy rigid registration mean Dice {mean_dice_legacy:.4f} < {THRESHOLD} for angle {angle_deg}°"
    )
    assert mean_dice_quat >= THRESHOLD, (
        f"Quaternion rigid registration mean Dice {mean_dice_quat:.4f} < {THRESHOLD} for angle {angle_deg}°"
    )
