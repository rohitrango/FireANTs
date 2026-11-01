"""Regression test for issue #111: geodesic (StationaryVelocity) deformations in Greedy and SyN.

GreedyRegistration and SyNRegistration call ``warp.step(loss)`` so that the Levenberg-Marquardt
optimizer can see the loss. ``StationaryVelocity.step`` must accept (and may ignore) that argument.
"""
import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from fireants.io.image import BatchedImages, Image
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration

device = "cuda" if torch.cuda.is_available() else "cpu"


def _blob_pair(size=48, shift=3):
    ''' a smooth blob and a shifted copy, as (fixed, moving) BatchedImages '''
    zz, yy, xx = np.meshgrid(*[np.arange(size, dtype=np.float32)] * 3, indexing="ij")
    c = size / 2
    fixed = np.exp(-((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) / (2 * (size / 6) ** 2))
    moving = np.roll(fixed, shift, axis=0)
    to_batch = lambda a: BatchedImages([Image(sitk.GetImageFromArray(a.astype(np.float32)), device=device)])
    return to_batch(fixed), to_batch(moving)


@pytest.mark.parametrize("reg_class", [GreedyRegistration, SyNRegistration])
def test_geodesic_deformation_optimizes(reg_class):
    fixed, moving = _blob_pair()
    reg = reg_class(scales=[2, 1], iterations=[10, 10], fixed_images=fixed, moving_images=moving,
                    loss_type="mse", deformation_type="geodesic", optimizer="Adam", optimizer_lr=0.03,
                    integrator_n=4, smooth_grad_sigma=0.5, smooth_warp_sigma=0.5, progress_bar=False)
    reg.optimize()   # raised TypeError: step() takes 1 positional argument but 2 were given
    moved = reg.evaluate(fixed, moving).detach()
    assert F.mse_loss(moved, fixed()) < 0.5 * F.mse_loss(moving(), fixed())
