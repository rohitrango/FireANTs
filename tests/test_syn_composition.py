"""SyN's field reads the inverse of the reverse warp in the fixed image's coordinates, which the affine takes to the
moving image's."""

import math

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from fireants.io.image import BatchedImages, Image
from fireants.registration.syn import SyNRegistration

SIZE = (40, 40, 40)
FIXED_SPACING = 2.0


def _image(spacing: float) -> BatchedImages:
    image = sitk.GetImageFromArray(np.random.default_rng(0).random(SIZE[::-1]).astype(np.float32))
    image.SetSpacing((spacing,) * 3)
    return BatchedImages([Image(image, device="cpu")])


def _millimetres(images: BatchedImages, coordinates: torch.Tensor) -> torch.Tensor:
    torch2phy = images.get_torch2phy()[0]
    return coordinates @ torch2phy[:3, :3].T + torch2phy[:3, 3]


def _field(registration: SyNRegistration, fixed: BatchedImages, moving: BatchedImages) -> torch.Tensor:
    """The registration's displacement field, in mm, on the fixed grid."""
    parameters = registration.get_warp_parameters(fixed, moving)
    shape = [1, 1, *SIZE[::-1]]
    moved = F.affine_grid(parameters["affine"], shape, align_corners=True) + parameters["grid"]
    identity = F.affine_grid(torch.eye(3, 4)[None], shape, align_corners=True)
    return (_millimetres(moving, moved) - _millimetres(fixed, identity)).detach()[0]


@pytest.mark.parametrize("moving_spacing", [2.0, 1.0, 3.0])
@pytest.mark.parametrize("scaled_rotation", [False, True])
def test_the_reverse_warp_moves_the_fixed_image_by_its_own_millimetres(moving_spacing, scaled_rotation):
    """No forward warp, a constant reverse warp ``c``: the midpoint ``x`` meets the fixed image at ``x + c``, so the
    reverse warp's share of the field is ``-M L c``, ``L c`` being ``c`` in the fixed image's mm and ``M`` the affine's
    linear part, whatever the moving image's field of view."""
    fixed, moving = _image(FIXED_SPACING), _image(moving_spacing)
    affine = torch.eye(4)[None]
    if scaled_rotation:
        angle = math.radians(10)
        affine[0, :2, :2] = 1.1 * torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    registration = SyNRegistration(
        [1],
        [1],
        fixed,
        moving,
        loss_type="mse",
        optimizer="Adam",
        smooth_warp_sigma=0,
        smooth_grad_sigma=0,
        init_affine=affine,
    )
    registration.fwd_warp.warp.data.zero_()
    registration.rev_warp.warp.data.zero_()
    affine_alone = _field(registration, fixed, moving)
    shift = torch.tensor([0.1, -0.05, 0.02])  # normalized to the fixed image, (x, y, z)
    registration.rev_warp.warp.data[:] = shift
    share = _field(registration, fixed, moving) - affine_alone
    expected = -affine[0, :3, :3] @ (shift * 0.5 * (torch.tensor(SIZE) - 1) * FIXED_SPACING)
    inner = share[8:-8, 8:-8, 8:-8]  # the inverse is exact away from the faces
    assert (inner - expected).abs().max() < 0.05
