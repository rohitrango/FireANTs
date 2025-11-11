import pytest
import torch
import numpy as np
from pathlib import Path
import SimpleITK as sitk

from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.io.image import Image, BatchedImages
from fireants.io import FakeBatchedImages
from .test_moments_registration import generate_3d_ellipse
try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

def create_synthetic_data_np(size):
    rng = np.random.RandomState(42)
    
    # Generate random axes lengths
    a = rng.uniform(size // 4, 3*size // 8)
    b = rng.uniform(3*size // 16, a-5)
    c = rng.uniform(2*size // 16, b-5)
    axes = (a, b, c)
    
    # Generate fixed image
    fixed_center = (2 * size // 32, -3*size // 32, size // 32)
    fixed_angles = (np.pi/6, np.pi/4, -np.pi/3)
    fixed_arr = generate_3d_ellipse(size=size, axes=axes, center=fixed_center, 
                                    angles=fixed_angles, rng=rng)
    
    # Generate moving image with different center and rotation
    moving_center = (-4*size // 32, size // 32, -2*size // 32)
    moving_angles = (-np.pi/3, -np.pi/6, np.pi/2)
    moving_arr = generate_3d_ellipse(size=size, axes=axes, center=moving_center, 
                                     angles=moving_angles, rng=rng)
    
    return fixed_arr, moving_arr


def test_lowdim():
    test_data_dir = Path(__file__).parent / "test_data"
    fixed_image_path = str(test_data_dir / "deformable_image_1.nii.gz")
    moving_image_path = str(test_data_dir / "deformable_image_2.nii.gz")

    downscales = [24, 12, 6]
    expected_zdims = [128 // (128 // zdim) for zdim in downscales]
    for downscale, expected_zdim in zip(downscales, expected_zdims):
        if any(not Path(f).exists() for f in [fixed_image_path, moving_image_path]):
            fixed_np, moving_np = create_synthetic_data_np(128)
        else:
            fixed_img = Image.load_file(fixed_image_path)
            moving_img = Image.load_file(moving_image_path)
            fixed_np = sitk.GetArrayFromImage(fixed_img.itk_image)
            moving_np = sitk.GetArrayFromImage(moving_img.itk_image)
            expected_zdim = fixed_np.shape[2] // (fixed_np.shape[2] // downscale)

        # Scale down
        fixed_np = fixed_np[:,:,::128//downscale]
        moving_np = moving_np[:,:,::128//downscale]

        fixed_dims = fixed_np.shape
        moving_dims = moving_np.shape

        fixed_itk = sitk.GetImageFromArray(fixed_np)
        moving_itk = sitk.GetImageFromArray(moving_np)

        fixed_itk.SetSpacing((1.0, 1.0, 128//downscale))
        moving_itk.SetSpacing((1.0, 1.0, 128//downscale))

        fixed_img = Image(fixed_itk, device='cuda')
        moving_img = Image(moving_itk, device='cuda')

        fixed_batch = BatchedImages([fixed_img])
        moving_batch = BatchedImages([moving_img])

        # Test AffineRegistration
        reg = AffineRegistration(
            scales=[4, 2, 1],
            iterations=[200, 100, 50],
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type='mse',
            optimizer='Adam',
            optimizer_lr=3e-2,
        )
        assert reg.min_dim <= min(fixed_dims), f"Min dimension is {reg.min_dim}, expected {min(fixed_dims)}"
        reg.optimize()

        # Test GreedyRegistration
        reg = GreedyRegistration(
            scales=[4, 2, 1],
            iterations=[200, 100, 50],
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type='cc',
            optimizer='Adam',
            optimizer_lr=0.2,
            smooth_warp_sigma=0.25,
            smooth_grad_sigma=0.5
        )
        assert reg.min_dim <= min(fixed_dims), f"Min dimension is {reg.min_dim}, expected {min(fixed_dims)}"
        reg.optimize()

        # Test SynRegistration
        reg = SyNRegistration(
            scales=[4, 2, 1],
            iterations=[200, 100, 50],
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type='cc',
            optimizer='Adam',
            optimizer_lr=0.2,
            smooth_warp_sigma=0.25,
            smooth_grad_sigma=0.5
        )
        assert reg.min_dim <= min(fixed_dims), f"Min dimension is {reg.min_dim}, expected {min(fixed_dims)}"
        reg.optimize()