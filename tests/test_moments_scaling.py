"""
Test case for moments registration with and without scaling.

This test verifies that when fixed and moving images have different scales,
using perform_scaling=True with moments=2 results in better alignment (lower loss)
than without scaling compensation.
"""

import pytest
import torch
from pathlib import Path
import logging
from torch.nn import functional as F

# Import FireANTs components
from fireants.registration.moments import MomentsRegistration
from fireants.io.image import Image, BatchedImages

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "moments_scaling"


@pytest.fixture(scope="function")
def scaling_test_data():
    """Fixture to load test data for moments scaling registration."""
    fixed_path = TEST_DATA_DIR / "fixed_rotated.nii.gz"
    moving_path = TEST_DATA_DIR / "moving_left_hemi.nii.gz"
    
    # Skip test if data doesn't exist
    if not fixed_path.exists() or not moving_path.exists():
        pytest.skip(f"Test data not found in {TEST_DATA_DIR}")
    
    # Load images
    fixed_img = Image.load_file(str(fixed_path), dtype=torch.float32)
    moving_img = Image.load_file(str(moving_path), dtype=torch.float32)
    
    # Create batched images
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])
    
    return {
        'fixed_batch': fixed_batch,
        'moving_batch': moving_batch,
        'fixed_path': fixed_path,
        'moving_path': moving_path,
    }


def compute_loss_after_registration(fixed_images, moving_images, perform_scaling, loss_type='cc'):
    """
    Helper function to run moments registration and compute the final loss.
    
    Args:
        fixed_images: BatchedImages for fixed image
        moving_images: BatchedImages for moving image
        perform_scaling: Whether to perform scaling correction
        loss_type: Loss type to use ('cc', 'mse', etc.)
    
    Returns:
        loss_value: The loss value after registration
        moved_image: The transformed moving image
    """
    # Run moments registration with 2nd order moments
    moments = MomentsRegistration(
        scale=4,  # downscale for faster computation
        fixed_images=fixed_images,
        moving_images=moving_images,
        moments=2,  # use 2nd order moments for rotation and scaling
        orientation='rot',  # try rotations
        loss_type=loss_type,
        perform_scaling=perform_scaling,
    )
    moments.optimize()
    
    # Get warp parameters and apply transformation
    moved_image = moments.evaluate(fixed_images, moving_images)
    # Compute loss between moved and fixed images
    fixed_array = fixed_images()
    loss = moments.loss_fn(moved_image, fixed_array).mean().item()
    
    return loss, moved_image


class TestMomentsScaling:
    """Test suite for moments-based registration with scaling."""
    
    def test_scaling_improves_alignment(self, scaling_test_data):
        """
        Test that moments registration with scaling produces lower loss 
        than without scaling when images have different scales.
        """
        fixed_batch = scaling_test_data['fixed_batch']
        moving_batch = scaling_test_data['moving_batch']
        
        # Run registration WITHOUT scaling
        logger.info("Running moments registration WITHOUT scaling...")
        loss_without_scaling, moved_without_scaling = compute_loss_after_registration(
            fixed_batch, moving_batch, perform_scaling=False
        )
        logger.info(f"Loss without scaling: {loss_without_scaling:.6f}")
        
        # Run registration WITH scaling
        logger.info("Running moments registration WITH scaling...")
        loss_with_scaling, moved_with_scaling = compute_loss_after_registration(
            fixed_batch, moving_batch, perform_scaling=True
        )
        logger.info(f"Loss with scaling: {loss_with_scaling:.6f}")
        
        # Assert that scaling improves the alignment (lower loss)
        assert loss_with_scaling < loss_without_scaling, (
            f"Expected scaling to improve alignment, but got:\n"
            f"  Loss with scaling: {loss_with_scaling:.6f}\n"
            f"  Loss without scaling: {loss_without_scaling:.6f}\n"
            f"The loss with scaling should be lower."
        )
        
        # Log the improvement
        improvement = (loss_without_scaling - loss_with_scaling) / loss_without_scaling * 100
        logger.info(f"Improvement with scaling: {improvement:.2f}%")

    def test_moments_registration_without_scaling(self, scaling_test_data):
        """Test that moments registration without scaling runs successfully."""
        fixed_batch = scaling_test_data['fixed_batch']
        moving_batch = scaling_test_data['moving_batch']
        
        # Run moments registration without scaling
        reg = MomentsRegistration(
            scale=4.0,
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            moments=2,
            orientation='rot',
            blur=True,
            loss_type='cc',
            perform_scaling=False
        )
        
        # Optimize the registration
        reg.optimize()
        
        # Get results
        moved_image = reg.evaluate(fixed_batch, moving_batch)
        
        # Basic sanity checks
        assert moved_image is not None
        assert moved_image.shape == fixed_batch().shape
        logger.info("Moments registration without scaling completed successfully.")

    def test_moments_registration_with_scaling(self, scaling_test_data):
        """Test that moments registration with scaling runs successfully."""
        fixed_batch = scaling_test_data['fixed_batch']
        moving_batch = scaling_test_data['moving_batch']
        
        # Run moments registration with scaling
        reg = MomentsRegistration(
            scale=4.0,
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            moments=2,
            orientation='rot',
            blur=True,
            loss_type='cc',
            perform_scaling=True
        )
        
        # Optimize the registration
        reg.optimize()
        
        # Get results
        moved_image = reg.evaluate(fixed_batch, moving_batch)
        
        # Basic sanity checks
        assert moved_image is not None
        assert moved_image.shape == fixed_batch().shape
        logger.info("Moments registration with scaling completed successfully.")

    def test_affine_matrix_difference(self, scaling_test_data):
        """
        Test that the affine matrices differ between scaled and unscaled registration,
        indicating that scaling correction is being applied.
        """
        fixed_batch = scaling_test_data['fixed_batch']
        moving_batch = scaling_test_data['moving_batch']
        
        # Registration without scaling
        reg_no_scale = MomentsRegistration(
            scale=4.0,
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            moments=2,
            orientation='rot',
            perform_scaling=False
        )
        reg_no_scale.optimize()
        affine_no_scale = reg_no_scale.get_affine_init()
        
        # Registration with scaling
        reg_with_scale = MomentsRegistration(
            scale=4.0,
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            moments=2,
            orientation='rot',
            perform_scaling=True
        )
        reg_with_scale.optimize()
        affine_with_scale = reg_with_scale.get_affine_init()
        
        # Check that the affine matrices are different
        diff = torch.abs(affine_with_scale - affine_no_scale).max().item()
        logger.info(f"Max difference between affine matrices: {diff:.6f}")
        
        assert diff > 1e-6, (
            "Expected different affine matrices with and without scaling, "
            f"but max difference is only {diff:.6f}"
        )
