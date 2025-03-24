import pytest
import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from time import sleep
import logging
logging.basicConfig(level=logging.INFO)

# Import FireANTs components
from fireants.registration import MomentsRegistration, AffineRegistration, RigidRegistration
from fireants.io.image import Image, BatchedImages, FakeBatchedImages
try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TestMomentsRegistration:
    """Test suite for moments-based registration."""
    
    def test_moments_registration_with_labels(self):
        """Test moments-based registration using OASIS test data with label overlap validation."""
        # Load test data
        test_data_dir = Path(__file__).parent / "test_data"
        output_dir = Path(__file__).parent / "test_results"

        # load images
        fixed_img = Image.load_file(str(test_data_dir / "oasis_157_image.nii.gz"))
        moving_img = Image.load_file(str(test_data_dir / "oasis_157_image_rotation_transform.nii.gz"))
        fixed_seg = Image.load_file(str(test_data_dir / "oasis_157_seg.nii.gz"), is_segmentation=True)
        moving_seg = Image.load_file(str(test_data_dir / "oasis_157_seg_rotation_transform.nii.gz"), is_segmentation=True)
        
        # Create BatchedImages objects
        fixed_batch = BatchedImages([fixed_img])
        moving_batch = BatchedImages([moving_img])
        
        # Run moments-based registration
        reg = MomentsRegistration(
            scale=2.0,  # No downsampling
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            moments=2,  # Use second-order moments
            orientation='rot',  # Try both rotation and anti-rotation
            blur=False,  # Apply Gaussian blur during downsampling
            loss_type='cc',  # Use cross-correlation loss
            cc_kernel_type='rectangular',
            cc_kernel_size=5
        )
        
        # Optimize the registration
        reg.optimize()
        
        # Get the transformation matrix
        # transform_matrix = reg.get_affine_init()
        
        # Apply the transformation to the moving image
        # coords = reg.get_warped_coordinates(fixed_batch, moving_batch)
        # registered_img = F.grid_sample(moving_batch(), coords[..., :-1], mode='bilinear', align_corners=True)
        moving_seg_batch = BatchedImages([moving_seg])
        fixed_seg_batch = BatchedImages([fixed_seg])
        moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch)

        dice_score_before = 1 - dice_loss(moving_seg_batch(), fixed_seg_batch(), reduce=False).mean(0)
        logger.info(f"Dice score before registration: {dice_score_before.mean()}")
        logger.info(f"Number of labels with Dice > 0.7: {sum(1 for d in dice_score_before if d > 0.7)}")
        logger.info(f"Number of labels with Dice > 0.8: {sum(1 for d in dice_score_before if d > 0.8)}")
        logger.info(f"Number of labels with Dice > 0.9: {sum(1 for d in dice_score_before if d > 0.9)}")
        logger.info("--------------------------------")

        # just get the rotation matrix
        rotation_matrix = reg.get_affine_init()[:, :, :-1]

        dice_scores = 1 - dice_loss(moved_seg_batch, fixed_seg_batch(), reduce=False).mean(0)
        mean_dice = dice_scores.mean()
        logger.info(f"Average Dice score: {mean_dice:.3f}")
        # Apply the same transformation to the segmentation
        assert mean_dice > 0.8, f"Average Dice score ({mean_dice:.3f}) is below threshold"
        
        # Check if the transformation matrix is valid
        dims = fixed_batch.images[0].dims
        assert rotation_matrix.shape == (1, dims, dims), "Transform matrix has incorrect shape"
        assert torch.allclose(torch.linalg.det(rotation_matrix), torch.tensor(1.0), atol=1e-5), \
               "Transform matrix determinant should be 1.0 (or close to it)"
        
        # Log the results
        logger.info("\nMoments Registration Results:")
        logger.info(f"Average Dice Score: {mean_dice:.3f}")
        logger.info(f"Number of labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
        logger.info(f"Number of labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
        logger.info(f"Number of labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")

        # save the results
        moved_image = reg.evaluate(fixed_batch, moving_batch)
        reg.save_moved_images(moved_image, str(output_dir / "moved_image_moments.nii.gz"))
        reg.save_moved_images(moved_seg_batch, str(output_dir / "moved_seg_moments.nii.gz"))

        # save the transformation matrix!


# class TestAffineRegistration:
#     """Test suite for affine registration."""
    
#     def test_affine_registration_tensor(self, create_test_images, compute_similarity):
#         """Test affine registration with tensor inputs."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Compute initial similarity
#         initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
        
#         # Run registration
#         reg = affine.AffineRegistration(
#             n_iter=50,
#             learning_rate=0.01,
#             loss='mse'
#         )
        
#         result = reg.register(fixed_img, moving_img)
#         registered_img = result.registered_image
#         affine_matrix = result.transform
        
#         # Compute registration quality
#         final_mse = compute_similarity(fixed_img, registered_img, metric='mse')
        
#         # Assertions
#         assert final_mse < initial_mse, "Registration did not improve image similarity"
#         assert affine_matrix.shape == (4, 4), "Affine matrix has incorrect shape"
#         assert torch.allclose(affine_matrix.det(), torch.tensor(1.0), atol=1e-5) or \
#                torch.abs(torch.abs(affine_matrix.det()) - 1.0) < 1e-5, \
#                "Affine matrix determinant should be +/-1 (or close to it)"
    
#     def test_affine_registration_file(self, save_test_images, compute_similarity):
#         """Test affine registration with file inputs."""
#         # Save test images to files
#         fixed_path, moving_path = save_test_images()
        
#         # Load images to compute initial similarity
#         fixed_sitk = sitk.ReadImage(fixed_path)
#         moving_sitk = sitk.ReadImage(moving_path)
#         fixed_np = sitk.GetArrayFromImage(fixed_sitk)
#         moving_np = sitk.GetArrayFromImage(moving_sitk)
        
#         initial_mse = compute_similarity(fixed_np, moving_np, metric='mse')
        
#         # Run registration
#         reg = affine.AffineRegistration(
#             n_iter=50,
#             learning_rate=0.01,
#             loss='mse'
#         )
        
#         result = reg.register_files(fixed_path, moving_path)
#         registered_img = result.registered_image
#         affine_matrix = result.transform
        
#         # Convert to numpy for similarity calculation
#         if isinstance(registered_img, torch.Tensor):
#             registered_np = registered_img.squeeze().detach().cpu().numpy()
#         else:
#             registered_np = sitk.GetArrayFromImage(registered_img)
        
#         # Compute registration quality
#         final_mse = compute_similarity(fixed_np, registered_np, metric='mse')
        
#         # Assertions
#         assert final_mse < initial_mse, "File-based registration did not improve image similarity"
#         assert affine_matrix.shape == (4, 4), "Affine matrix has incorrect shape"


# class TestRigidRegistration:
    # """Test suite for rigid registration."""
    
    # def test_rigid_registration_tensor(self, create_test_images, compute_similarity):
    #     """Test rigid registration with tensor inputs."""
    #     # Create test images
    #     fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
    #     # Compute initial similarity
    #     initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
        
    #     # Run registration
    #     reg = rigid.RigidRegistration(
    #         n_iter=50,
    #         learning_rate=0.01,
    #         loss='mse'
    #     )
        
    #     result = reg.register(fixed_img, moving_img)
    #     registered_img = result.registered_image
    #     rigid_matrix = result.transform
        
    #     # Compute registration quality
    #     final_mse = compute_similarity(fixed_img, registered_img, metric='mse')
        
    #     # Assertions
    #     assert final_mse < initial_mse, "Registration did not improve image similarity"
    #     assert rigid_matrix.shape == (4, 4), "Rigid matrix has incorrect shape"
    #     assert torch.allclose(rigid_matrix.det(), torch.tensor(1.0), atol=1e-5), \
    #            "Rigid matrix determinant should be 1.0 (or close to it)"
        
    #     # Check orthogonality of rotation part (upper-left 3x3)
    #     rotation_part = rigid_matrix[:3, :3]
    #     identity = torch.matmul(rotation_part, rotation_part.transpose(-1, -2))
    #     assert torch.allclose(identity, torch.eye(3), atol=1e-5), \
    #            "Rotation part of rigid matrix should be orthogonal"
    
    # def test_rigid_registration_file(self, save_test_images, compute_similarity):
    #     """Test rigid registration with file inputs."""
    #     # Save test images to files
    #     fixed_path, moving_path = save_test_images()
        
    #     # Load images to compute initial similarity
    #     fixed_sitk = sitk.ReadImage(fixed_path)
    #     moving_sitk = sitk.ReadImage(moving_path)
    #     fixed_np = sitk.GetArrayFromImage(fixed_sitk)
    #     moving_np = sitk.GetArrayFromImage(moving_sitk)
        
    #     initial_mse = compute_similarity(fixed_np, moving_np, metric='mse')
        
    #     # Run registration
    #     reg = rigid.RigidRegistration(
    #         n_iter=50,
    #         learning_rate=0.01,
    #         loss='mse'
    #     )
        
    #     result = reg.register_files(fixed_path, moving_path)
    #     registered_img = result.registered_image
    #     rigid_matrix = result.transform
        
    #     # Convert to numpy for similarity calculation
    #     if isinstance(registered_img, torch.Tensor):
    #         registered_np = registered_img.squeeze().detach().cpu().numpy()
    #     else:
    #         registered_np = sitk.GetArrayFromImage(registered_img)
        
    #     # Compute registration quality
    #     final_mse = compute_similarity(fixed_np, registered_np, metric='mse')
        
    #     # Assertions
    #     assert final_mse < initial_mse, "File-based registration did not improve image similarity"
    #     assert rigid_matrix.shape == (4, 4), "Rigid matrix has incorrect shape"
        
    #     # Check orthogonality of rotation part (upper-left 3x3)
    #     rotation_part = rigid_matrix[:3, :3].cpu().numpy()
    #     identity = np.matmul(rotation_part, rotation_part.transpose())
    #     assert np.allclose(identity, np.eye(3), atol=1e-5), \
    #            "Rotation part of rigid matrix should be orthogonal" 