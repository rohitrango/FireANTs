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
    moving_img.array = moving_img.array * 1.0 / moving_img.array.max() * fixed_img.array.max()

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

@pytest.fixture(scope="class")
def rigid_registration_results():
    """Fixture to compute and share rigid registration results across subtests."""
    # Load test data
    test_data_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    ext = "txt"

    # load images
    fixed_img = Image.load_file(str(test_data_dir / "oasis_157_image.nii.gz"))
    moving_img = Image.load_file(str(test_data_dir / "oasis_157_image_affine.nii.gz"))
    moving_img.array = moving_img.array * 1.0 / moving_img.array.max() * fixed_img.array.max()

    fixed_seg = Image.load_file(str(test_data_dir / "oasis_157_seg.nii.gz"), is_segmentation=True)
    moving_seg = Image.load_file(str(test_data_dir / "oasis_157_seg_affine.nii.gz"), is_segmentation=True)
    
    # Create BatchedImages objects
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])
    moving_seg_batch = BatchedImages([moving_seg])
    fixed_seg_batch = BatchedImages([fixed_seg])

    # reg1 = MomentsRegistration(
    #     scale=1.0,
    #     fixed_images=fixed_batch,
    #     moving_images=moving_batch,
    #     moments=2,
    #     orientation='rot',
    #     blur=False,
    #     loss_type='cc',
    #     cc_kernel_type='rectangular',
    #     cc_kernel_size=5
    # )
    # reg1.optimize()
    
    # Run rigid registration
    reg = RigidRegistration(
        scales=[8, 4, 2, 1],
        iterations=[200, 200, 100, 50],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        # init_moment=reg1.get_rigid_moment_init(),
        # init_translation=reg1.get_rigid_transl_init(),
        loss_type='mse',
        optimizer='Adam',
        optimizer_lr=3e-1,
        scaling=False,
        cc_kernel_type='rectangular',
        cc_kernel_size=5,
        blur=False
    )
    
    # Optimize the registration
    reg.optimize()
    
    # Get results
    moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch).detach()
    moved_image = reg.evaluate(fixed_batch, moving_batch).detach()
   
    # Save results
    reg.save_moved_images(moved_image, str(output_dir / "moved_image_rigid.nii.gz"))
    reg.save_moved_images(moved_seg_batch, str(output_dir / "moved_seg_rigid.nii.gz"))
    
    # Save transformation matrix
    transform_matrix = reg.get_rigid_matrix()
    reg.save_as_ants_transforms(str(output_dir / f"transform_matrix_rigid.{ext}"))

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
        'moving_path': str(test_data_dir / "oasis_157_image_affine.nii.gz"),
        'transform_path': str(output_dir / f"transform_matrix_rigid.{ext}"),
    }

class TestRigidRegistration:
    """Test suite for rigid registration."""
    
    def test_initial_dice_score(self, rigid_registration_results):
        """Test that initial Dice score is close to 0."""
        dice_score_before = 1 - dice_loss(rigid_registration_results['moving_seg_batch'](), 
                                        rigid_registration_results['fixed_seg_batch'](), 
                                        reduce=False).mean(0)
        mean_dice = dice_score_before.mean()
        logger.info(f"Dice score before registration: {mean_dice:.3f}")
        assert mean_dice < 0.7, f"Initial Dice score ({mean_dice:.3f}) should be small"
    
    def test_final_dice_score(self, rigid_registration_results):
        """Test that final Dice score is above threshold."""
        dice_scores = 1 - dice_loss(rigid_registration_results['moved_seg_batch'], 
                                  rigid_registration_results['fixed_seg_batch'](), 
                                  reduce=False).mean(0)
        mean_dice = dice_scores.mean()
        logger.info(f"Average Dice score: {mean_dice:.3f}")
        assert mean_dice > 0.8, f"Average Dice score ({mean_dice:.3f}) is below threshold"
        
        # Log detailed results
        logger.info("\nRigid Registration Results:")
        logger.info(f"Average Dice Score: {mean_dice:.3f}")
        logger.info(f"Number of labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
        logger.info(f"Number of labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
        logger.info(f"Number of labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    

    def test_transform_consistency(self, rigid_registration_results):
        """Test that saved transform and antsApplyTransforms give consistent results."""
        # Load the saved transform matrix
        transform_matrix = rigid_registration_results['transform_matrix']
        
        # Check if the transformation matrix is valid
        dims = rigid_registration_results['fixed_batch'].images[0].dims
        rotation_matrix = transform_matrix[:, :dims, :dims]
        assert rotation_matrix.shape == (1, dims, dims), "Transform matrix has incorrect shape"
        assert torch.allclose(torch.linalg.det(rotation_matrix), torch.tensor(1.0), atol=1e-5), \
               "Transform matrix determinant should be 1.0 (or close to it)"
        
        # Check orthogonality of rotation part
        identity = torch.matmul(rotation_matrix, rotation_matrix.transpose(-1, -2))
        assert torch.allclose(identity, torch.eye(dims, device=rotation_matrix.device), atol=1e-5), \
               "Rotation matrix should be orthogonal"
        
        # Apply transform using antsApplyTransforms
        fixed_path = rigid_registration_results['fixed_path']
        moving_path = rigid_registration_results['moving_path']
        transform_path = rigid_registration_results['transform_path']
        ants_output = str(rigid_registration_results['output_dir'] / "moved_image_ants_rigid.nii.gz")
        
        # Run antsApplyTransforms with the ANTs transform matrix
        cmd = f"antsApplyTransforms -d 3 -i {moving_path} -r {fixed_path} -t {transform_path} -o {ants_output}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Load and compare results
        ants_result = Image.load_file(ants_output)
        # normalize the ants array to the same range as the fixed image
        ants_array = ants_result.array.squeeze().cpu().numpy()
        ants_array = ants_array * rigid_registration_results['moving_batch'].images[0].array.max().item() / ants_array.max()
        moved_array = rigid_registration_results['moved_image'].squeeze().cpu().numpy()
        logger.info(f"ANTS array min: {ants_array.min()}, max: {ants_array.max()}")
        logger.info(f"Moved array min: {moved_array.min()}, max: {moved_array.max()}")
        
        # Compare using relative error
        rel_error = np.mean(np.abs(ants_array - moved_array) / (np.abs(moved_array) + 1e-6))
        logger.info(f"Relative error: {rel_error:.4f}")
        assert rel_error < 0.01, f"Transform consistency check failed. Relative error: {rel_error:.4f}"

@pytest.fixture(scope="class")
def affine_registration_results():
    """Fixture to compute and share affine registration results across subtests."""
    # Load test data
    test_data_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    ext = "txt"

    # load images
    fixed_img = Image.load_file(str(test_data_dir / "oasis_157_image.nii.gz"))
    moving_img = Image.load_file(str(test_data_dir / "oasis_157_image_affine.nii.gz"))
    moving_img.array = moving_img.array * 1.0 / moving_img.array.max() * fixed_img.array.max()

    fixed_seg = Image.load_file(str(test_data_dir / "oasis_157_seg.nii.gz"), is_segmentation=True)
    moving_seg = Image.load_file(str(test_data_dir / "oasis_157_seg_affine.nii.gz"), is_segmentation=True)
    
    # Create BatchedImages objects
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])
    moving_seg_batch = BatchedImages([moving_seg])
    fixed_seg_batch = BatchedImages([fixed_seg])
    
    # Run affine registration
    reg = AffineRegistration(
        scales=[4, 2, 1],
        iterations=[200, 100, 50],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        # init_rigid=init_rigid,
        loss_type='mse',
        optimizer='Adam',
        optimizer_lr=3e-2,
    )
    
    # Optimize the registration
    reg.optimize()
    
    # Get results
    moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch).detach()
    moved_image = reg.evaluate(fixed_batch, moving_batch).detach()
   
    # Save results
    reg.save_moved_images(moved_image, str(output_dir / "moved_image_affine.nii.gz"))
    reg.save_moved_images(moved_seg_batch, str(output_dir / "moved_seg_affine.nii.gz"))
    
    # Save transformation matrix
    transform_matrix = reg.get_affine_matrix()
    reg.save_as_ants_transforms(str(output_dir / f"transform_matrix_affine.{ext}"))

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
        'moving_path': str(test_data_dir / "oasis_157_image_affine.nii.gz"),
        'transform_path': str(output_dir / f"transform_matrix_affine.{ext}"),
    }

class TestAffineRegistration:
    """Test suite for affine registration."""
    
    def test_initial_dice_score(self, affine_registration_results):
        """Test that initial Dice score is close to 0."""
        dice_score_before = 1 - dice_loss(affine_registration_results['moving_seg_batch'](), 
                                        affine_registration_results['fixed_seg_batch'](), 
                                        reduce=False).mean(0)
        mean_dice = dice_score_before.mean()
        logger.info(f"Dice score before registration: {mean_dice:.3f}")
        assert mean_dice < 0.7, f"Initial Dice score ({mean_dice:.3f}) should be small"
    
    def test_final_dice_score(self, affine_registration_results):
        """Test that final Dice score is above threshold."""
        dice_scores = 1 - dice_loss(affine_registration_results['moved_seg_batch'], 
                                  affine_registration_results['fixed_seg_batch'](), 
                                  reduce=False).mean(0)
        mean_dice = dice_scores.mean()
        logger.info(f"Average Dice score: {mean_dice:.3f}")
        assert mean_dice > 0.8, f"Average Dice score ({mean_dice:.3f}) is below threshold"
        
        # Log detailed results
        logger.info("\nAffine Registration Results:")
        logger.info(f"Average Dice Score: {mean_dice:.3f}")
        logger.info(f"Number of labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
        logger.info(f"Number of labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
        logger.info(f"Number of labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
    
    def test_transform_consistency(self, affine_registration_results):
        """Test that saved transform and antsApplyTransforms give consistent results."""
        # Load the saved transform matrix
        transform_matrix = affine_registration_results['transform_matrix']
        
        # Check if the transformation matrix is valid
        dims = affine_registration_results['fixed_batch'].images[0].dims
        affine_matrix = transform_matrix[:, :dims, :dims]
        assert affine_matrix.shape == (1, dims, dims), "Transform matrix has incorrect shape"
        assert torch.abs(torch.linalg.det(affine_matrix)) > 0.1, \
               "Affine matrix determinant should be non-zero"
        
        # Apply transform using antsApplyTransforms
        fixed_path = affine_registration_results['fixed_path']
        moving_path = affine_registration_results['moving_path']
        transform_path = affine_registration_results['transform_path']
        ants_output = str(affine_registration_results['output_dir'] / "moved_image_ants_affine.nii.gz")
        
        # Run antsApplyTransforms with the ANTs transform matrix
        cmd = f"antsApplyTransforms -d 3 -i {moving_path} -r {fixed_path} -t {transform_path} -o {ants_output}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Load and compare results
        ants_result = Image.load_file(ants_output)
        # normalize the ants array to the same range as the fixed image
        ants_array = ants_result.array.squeeze().cpu().numpy()
        ants_array = ants_array * affine_registration_results['moving_batch'].images[0].array.max().item() / ants_array.max()
        moved_array = affine_registration_results['moved_image'].squeeze().cpu().numpy()
        logger.info(f"ANTS array min: {ants_array.min()}, max: {ants_array.max()}")
        logger.info(f"Moved array min: {moved_array.min()}, max: {moved_array.max()}")
        
        # Compare using relative error
        rel_error = np.mean(np.abs(ants_array - moved_array) / (np.abs(moved_array) + 1e-6))
        logger.info(f"Relative error: {rel_error:.4f}")
        assert rel_error < 0.005, f"Transform consistency check failed. Relative error: {rel_error:.4f}"

