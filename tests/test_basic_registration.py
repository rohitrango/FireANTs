import pytest
import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from time import sleep
import logging
import subprocess
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

@pytest.fixture(scope="class")
def registration_results():
    """Fixture to compute and share registration results across subtests."""
    # Load test data
    test_data_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    ext = "txt"

    # load images
    fixed_img = Image.load_file(str(test_data_dir / "oasis_157_image.nii.gz"))
    moving_img = Image.load_file(str(test_data_dir / "oasis_157_image_rotated.nii.gz"))
    fixed_seg = Image.load_file(str(test_data_dir / "oasis_157_seg.nii.gz"), is_segmentation=True)
    moving_seg = Image.load_file(str(test_data_dir / "oasis_157_seg_rotated.nii.gz"), is_segmentation=True)
    
    # Create BatchedImages objects
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])
    moving_seg_batch = BatchedImages([moving_seg])
    fixed_seg_batch = BatchedImages([fixed_seg])
    
    # Run moments-based registration
    reg = MomentsRegistration(
        scale=1.0,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        moments=2,
        orientation='rot',
        blur=False,
        loss_type='cc',
        cc_kernel_type='rectangular',
        cc_kernel_size=5
    )
    
    # Optimize the registration
    reg.optimize()
    
    # Get results
    moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch)
    moved_image = reg.evaluate(fixed_batch, moving_batch)
   
    # Save results
    reg.save_moved_images(moved_image, str(output_dir / "moved_image_moments.nii.gz"))
    reg.save_moved_images(moved_seg_batch, str(output_dir / "moved_seg_moments.nii.gz"))
    
    ## TODO: implement save function in ANTs format
    # Save transformation matrix
    transform_matrix = reg.get_affine_init()
    print(transform_matrix)
    reg.save_as_ants_transforms(str(output_dir / f"transform_matrix_moments.{ext}"))

    return {
        'reg': reg,
        'fixed_batch': fixed_batch,
        'moving_batch': moving_batch,
        'fixed_seg_batch': fixed_seg_batch,
        'moving_seg_batch': moving_seg_batch,
        'moved_seg_batch': moved_seg_batch,
        'moved_image': moved_image,
        'transform_matrix': transform_matrix,
        'output_dir': output_dir,
        'fixed_path': str(test_data_dir / "oasis_157_image.nii.gz"),
        'moving_path': str(test_data_dir / "oasis_157_image_rotated.nii.gz"),
        'transform_path': str(output_dir / f"transform_matrix_moments.{ext}"),
    }

class TestMomentsRegistration:
    """Test suite for moments-based registration."""
    
    def test_initial_dice_score(self, registration_results):
        """Test that initial Dice score is close to 0."""
        dice_score_before = 1 - dice_loss(registration_results['moving_seg_batch'](), 
                                        registration_results['fixed_seg_batch'](), 
                                        reduce=False).mean(0)
        mean_dice = dice_score_before.mean()
        logger.info(f"Dice score before registration: {mean_dice:.3f}")
        assert mean_dice < 0.2, f"Initial Dice score ({mean_dice:.3f}) should be close to 0"
    
    def test_final_dice_score(self, registration_results):
        """Test that final Dice score is above threshold."""
        dice_scores = 1 - dice_loss(registration_results['moved_seg_batch'], 
                                  registration_results['fixed_seg_batch'](), 
                                  reduce=False).mean(0)
        mean_dice = dice_scores.mean()
        logger.info(f"Average Dice score: {mean_dice:.3f}")
        assert mean_dice > 0.8, f"Average Dice score ({mean_dice:.3f}) is below threshold"
        
        # Log detailed results
        logger.info("\nMoments Registration Results:")
        logger.info(f"Average Dice Score: {mean_dice:.3f}")
        logger.info(f"Number of labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
        logger.info(f"Number of labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
        logger.info(f"Number of labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    
    def test_transform_consistency(self, registration_results):
        """Test that saved transform and antsApplyTransforms give consistent results."""
        # Load the saved transform matrix
        transform_matrix = registration_results['transform_matrix']
        
        # Check if the transformation matrix is valid
        dims = registration_results['fixed_batch'].images[0].dims
        rotation_matrix = transform_matrix[:, :, :-1]
        assert rotation_matrix.shape == (1, dims, dims), "Transform matrix has incorrect shape"
        assert torch.allclose(torch.linalg.det(rotation_matrix), torch.tensor(1.0), atol=1e-5), \
               "Transform matrix determinant should be 1.0 (or close to it)"
        
        # Apply transform using antsApplyTransforms
        fixed_path = registration_results['fixed_path']
        moving_path = registration_results['moving_path']
        transform_path = registration_results['transform_path']
        ants_output = str(registration_results['output_dir'] / "moved_image_ants.nii.gz")

        # Save fixed and moving images for ANTs
        # registration_results['fixed_batch'].save_to_file(fixed_path)
        # registration_results['moving_batch'].save_to_file(moving_path)
        
        # Run antsApplyTransforms with the ANTs transform matrix
        cmd = f"antsApplyTransforms -d 3 -i {moving_path} -r {fixed_path} -t {transform_path} -o {ants_output}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Load and compare results
        ants_result = Image.load_file(ants_output)
        ants_array = ants_result.array.squeeze().cpu().numpy()
        moved_array = registration_results['moved_image'].squeeze().cpu().numpy()
        logger.info(f"ANTS array min: {ants_array.min()}, max: {ants_array.max()}")
        logger.info(f"Moved array min: {moved_array.min()}, max: {moved_array.max()}")
        
        # Compare using normalized cross-correlation
        # mse = np.mean((ants_array - moved_array)**2)
        rel_error = np.mean(np.abs(ants_array - moved_array) / (np.abs(moved_array) + 1e-6))
        logger.info(f"Relative error: {rel_error:.4f}")
        assert rel_error < 0.001, f"Transform consistency check failed. Relative error: {rel_error:.4f}"


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