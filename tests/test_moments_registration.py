import pytest
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path
import logging

# Import FireANTs components
from fireants.registration.moments import MomentsRegistration
from fireants.io.image import Image, BatchedImages

try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_2d_ellipse(size=128, axes=(40, 20), center=None, angle=None, rng=None):
    """Generate a 2D ellipse in a size x size image.
    
    Args:
        size (int): Size of the image (size x size)
        axes (tuple): Major and minor axes lengths (a, b)
        center (tuple): Center coordinates (x, y). If None, randomly chosen
        angle (float): Rotation angle in radians. If None, randomly chosen
        rng (np.random.RandomState): Random number generator
        
    Returns:
        np.ndarray: Binary image containing the ellipse
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    # Generate grid coordinates
    x = np.linspace(-size//2, size//2-1, size)
    y = np.linspace(-size//2, size//2-1, size)
    X, Y = np.meshgrid(x, y)
    
    # Random center if not provided
    if center is None:
        # Ensure ellipse stays within image bounds
        max_offset = min(size//6, min(axes))
        center = (rng.uniform(-max_offset, max_offset),
                 rng.uniform(-max_offset, max_offset))
    
    # Random angle if not provided
    if angle is None:
        angle = rng.uniform(0, 2*np.pi)
        
    # Translate coordinates
    X = X - center[0]
    Y = Y - center[1]
    
    # Rotate coordinates
    X_rot = X * np.cos(angle) + Y * np.sin(angle)
    Y_rot = -X * np.sin(angle) + Y * np.cos(angle)
    
    # Create ellipse
    ellipse = ((X_rot/axes[0])**2 + (Y_rot/axes[1])**2) <= 1
    return ellipse.astype(np.float32)

def generate_3d_ellipse(size=128, axes=(40, 30, 20), center=None, angles=None, rng=None):
    """Generate a 3D ellipsoid in a size x size x size image.
    
    Args:
        size (int): Size of the image (size x size x size)
        axes (tuple): Axes lengths (a, b, c)
        center (tuple): Center coordinates (x, y, z). If None, randomly chosen
        angles (tuple): Rotation angles (theta, phi, psi) in radians. If None, randomly chosen
        rng (np.random.RandomState): Random number generator
        
    Returns:
        np.ndarray: Binary image containing the ellipsoid
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    # Generate grid coordinates
    x = np.linspace(-size//2, size//2-1, size)
    y = np.linspace(-size//2, size//2-1, size)
    z = np.linspace(-size//2, size//2-1, size)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Random center if not provided
    if center is None:
        # Ensure ellipsoid stays within image bounds
        max_offset = min(size//6, min(axes))
        center = (rng.uniform(-max_offset, max_offset),
                 rng.uniform(-max_offset, max_offset),
                 rng.uniform(-max_offset, max_offset))
    
    # Random angles if not provided
    if angles is None:
        angles = (rng.uniform(0, 2*np.pi),
                 rng.uniform(0, 2*np.pi),
                 rng.uniform(0, 2*np.pi))
    
    # Translate coordinates
    X = X - center[0]
    Y = Y - center[1]
    Z = Z - center[2]
    
    # Rotate coordinates using Euler angles
    # First rotation about z
    X1 = X * np.cos(angles[0]) - Y * np.sin(angles[0])
    Y1 = X * np.sin(angles[0]) + Y * np.cos(angles[0])
    Z1 = Z
    
    # Second rotation about y
    X2 = X1 * np.cos(angles[1]) + Z1 * np.sin(angles[1])
    Y2 = Y1
    Z2 = -X1 * np.sin(angles[1]) + Z1 * np.cos(angles[1])
    
    # Third rotation about z
    X3 = X2 * np.cos(angles[2]) - Y2 * np.sin(angles[2])
    Y3 = X2 * np.sin(angles[2]) + Y2 * np.cos(angles[2])
    Z3 = Z2
    
    # Create ellipsoid
    ellipsoid = ((X3/axes[0])**2 + (Y3/axes[1])**2 + (Z3/axes[2])**2) <= 1
    return ellipsoid.astype(np.float32)

@pytest.fixture(scope="function")
def registration_2d_data():
    """Fixture to generate 2D test data for moments registration."""
    # Set random seed
    rng = np.random.RandomState(42)
    size = 128
    
    # Generate random axes lengths
    a = rng.uniform(30, 50)
    b = rng.uniform(20, a-5)
    axes = (a, b)
    
    # Generate fixed image
    fixed_center = (10, -15)
    fixed_angle = np.pi/6
    fixed_array = generate_2d_ellipse(size=size, axes=axes, center=fixed_center, 
                                    angle=fixed_angle, rng=rng)
    
    # Generate moving image with different center and rotation
    moving_center = (-20, 5)
    moving_angle = -np.pi/3
    moving_array = generate_2d_ellipse(size=size, axes=axes, center=moving_center, 
                                     angle=moving_angle, rng=rng)
    
    # Convert to ITK images
    fixed_itk = sitk.GetImageFromArray(fixed_array)
    moving_itk = sitk.GetImageFromArray(moving_array)
    
    # Create Image objects
    fixed_img = Image(fixed_itk, device='cuda')
    moving_img = Image(moving_itk, device='cuda')
    
    # Create BatchedImages objects
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])
    
    return {
        'fixed_batch': fixed_batch,
        'moving_batch': moving_batch,
        'size': size,
        'axes': axes,
        'fixed_center': fixed_center,
        'moving_center': moving_center,
        'fixed_angle': fixed_angle,
        'moving_angle': moving_angle,
    }

@pytest.fixture(scope="function")
def registration_3d_data():
    """Fixture to generate 3D test data for moments registration."""
    # Set random seed
    rng = np.random.RandomState(42)
    size = 128
    
    # Generate random axes lengths
    a = rng.uniform(30, 50)
    b = rng.uniform(25, a-5)
    c = rng.uniform(20, b-5)
    axes = (a, b, c)
    
    # Generate fixed image
    fixed_center = (10, -15, 5)
    fixed_angles = (np.pi/6, np.pi/4, -np.pi/3)
    fixed_array = generate_3d_ellipse(size=size, axes=axes, center=fixed_center, 
                                    angles=fixed_angles, rng=rng)
    
    # Generate moving image with different center and rotation
    moving_center = (-20, 5, -10)
    moving_angles = (-np.pi/3, -np.pi/6, np.pi/2)
    moving_array = generate_3d_ellipse(size=size, axes=axes, center=moving_center, 
                                     angles=moving_angles, rng=rng)
    
    # Convert to ITK images
    fixed_itk = sitk.GetImageFromArray(fixed_array)
    moving_itk = sitk.GetImageFromArray(moving_array)
    
    # Create Image objects
    fixed_img = Image(fixed_itk, device='cuda')
    moving_img = Image(moving_itk, device='cuda')
    
    # Create BatchedImages objects
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])
    
    return {
        'fixed_batch': fixed_batch,
        'moving_batch': moving_batch,
        'size': size,
        'axes': axes,
        'fixed_center': fixed_center,
        'moving_center': moving_center,
        'fixed_angles': fixed_angles,
        'moving_angles': moving_angles,
    }

class TestMomentsRegistration2D:
    """Test suite for 2D moments-based registration."""
    
    def test_initial_dice_score(self, registration_2d_data):
        """Test that initial Dice score is low."""
        dice_score_before = 1 - dice_loss(registration_2d_data['moving_batch'](), 
                                      registration_2d_data['fixed_batch'](), 
                                      reduce=False).mean()
        logger.info(f"Dice score before registration: {dice_score_before:.3f}")
        assert dice_score_before < 0.7, f"Initial Dice score ({dice_score_before:.3f}) should be low"
    
    def test_registration_and_final_dice(self, registration_2d_data):
        """Test that moments registration achieves high accuracy."""
        # Run moments registration
        reg = MomentsRegistration(
            scale=1.0,
            fixed_images=registration_2d_data['fixed_batch'],
            moving_images=registration_2d_data['moving_batch'],
            moments=2,
            orientation='rot',
            blur=False,
            loss_type='mse'
        )
        
        # Optimize the registration
        reg.optimize()
        
        # Get results
        moved_image = reg.evaluate(registration_2d_data['fixed_batch'], 
                                 registration_2d_data['moving_batch'])
        
        # Calculate final Dice score
        dice_score = 1 - dice_loss(moved_image, 
                                registration_2d_data['fixed_batch'](), 
                                reduce=False).mean()
        
        logger.info(f"Final Dice score: {dice_score:.3f}")
        assert dice_score > 0.98, f"Final Dice score ({dice_score:.3f}) is below threshold"

class TestMomentsRegistration3D:
    """Test suite for 3D moments-based registration."""
    
    def test_initial_dice_score(self, registration_3d_data):
        """Test that initial Dice score is low."""
        dice_score_before = 1 - dice_loss(registration_3d_data['moving_batch'](), 
                                      registration_3d_data['fixed_batch'](), 
                                      reduce=False).mean()
        logger.info(f"Dice score before registration: {dice_score_before:.3f}")
        assert dice_score_before < 0.7, f"Initial Dice score ({dice_score_before:.3f}) should be low"
    
    def test_registration_and_final_dice(self, registration_3d_data):
        """Test that moments registration achieves high accuracy."""
        # Run moments registration
        reg = MomentsRegistration(
            scale=1.0,
            fixed_images=registration_3d_data['fixed_batch'],
            moving_images=registration_3d_data['moving_batch'],
            moments=2,
            orientation='rot',
            blur=False,
            loss_type='mse'
        )
        
        # Optimize the registration
        reg.optimize()
        
        # Get results
        moved_image = reg.evaluate(registration_3d_data['fixed_batch'], 
                                 registration_3d_data['moving_batch'])
        
        # Calculate final Dice score
        dice_score = 1 - dice_loss(moved_image, 
                                registration_3d_data['fixed_batch'](), 
                                reduce=False).mean()
        
        logger.info(f"Final Dice score: {dice_score:.3f}")
        assert dice_score > 0.98, f"Final Dice score ({dice_score:.3f}) is below threshold"
