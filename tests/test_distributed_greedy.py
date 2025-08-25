import pytest
import torch
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import logging
from pathlib import Path
import torch.nn.functional as F

from fireants.registration.distributedgreedy import DistributedGreedyRegistration
from fireants.io.image import Image, BatchedImages
from fireants.utils.imageutils import jacobian
from fireants.registration.distributed import parallel_state
try:
    from .conftest import dice_loss
except ImportError:
    from conftest import dice_loss

# Set up logging
logger = logging.getLogger(__name__)

test_data_dir = Path(__file__).parent / "test_data"
fixed_image_path = str(test_data_dir / "deformable_image_1.nii.gz")
moving_image_path = str(test_data_dir / "deformable_image_2.nii.gz")
fixed_seg_path = str(test_data_dir / "deformable_seg_1.nii.gz")
moving_seg_path = str(test_data_dir / "deformable_seg_2.nii.gz")

@pytest.fixture(scope="class")
def distributed_greedy_registration_results():
    """Fixture to compute and share distributed greedy registration results across subtests.
    
    This fixture requires a distributed environment to run. Set the following environment variables:
    - WORLD_SIZE: Number of processes to use (e.g. 4)
    - RANK: Process rank (0 to WORLD_SIZE-1)
    
    Example:
        WORLD_SIZE=4 python -m torch.distributed.run --nproc_per_node=4 -m pytest tests/test_distributed_greedy.py
    """
    # Initialize distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    parallel_state.initialize_parallel_state(grid_parallel_size=world_size)
    rank = parallel_state.get_parallel_state().get_rank()

    # Create output directory
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    ext = "nii.gz"

    # Run registration with and without ring sampler
    results = {}
    for use_ring_sampler in [True, False]:
        suffix = "_ring" if use_ring_sampler else "_standard"

        fixed_image = Image.load_file(fixed_image_path, device='cpu')
        moving_image = Image.load_file(moving_image_path, device='cpu')
        fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True, device='cpu')
        moving_seg = Image.load_file(moving_seg_path, is_segmentation=True, device='cpu')

        # Create BatchedImages objects
        fixed_batch = BatchedImages([fixed_image])
        moving_batch = BatchedImages([moving_image])
        moving_seg_batch = BatchedImages([moving_seg])
        fixed_seg_batch = BatchedImages([fixed_seg])
    
        
        # Run distributed greedy registration
        reg = DistributedGreedyRegistration(
            scales=[4, 2, 1],
            iterations=[200, 100, 50],
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type='fusedcc',
            optimizer='Adam',
            optimizer_lr=0.2,
            smooth_warp_sigma=0.25,
            smooth_grad_sigma=0.5,
            use_ring_sampler=use_ring_sampler
        )
        
        # Optimize the registration
        reg.optimize()
        
        # Get results
        moved_seg_batch = reg.evaluate(fixed_seg_batch, moving_seg_batch).detach()
        moved_image = reg.evaluate(fixed_batch, moving_batch).detach()

        print(moved_seg_batch.shape)
        print(moved_image.shape)
       
        # Save results only from rank 0
        if rank == 0:
            reg.save_moved_images(moved_image, str(output_dir / f"moved_image_distributed_greedy{suffix}.nii.gz"))
            reg.save_moved_images(moved_seg_batch, str(output_dir / f"moved_seg_distributed_greedy{suffix}.nii.gz"))
        reg.save_as_ants_transforms(str(output_dir / f"warp_field_distributed_greedy{suffix}.{ext}"))

        # Ensure all processes are synced
        torch.distributed.barrier()

        # Store results
        results[f"reg{suffix}"] = reg
        results[f"moved_seg_batch{suffix}"] = moved_seg_batch
        results[f"moved_image{suffix}"] = moved_image
        results[f"transform_path{suffix}"] = str(output_dir / f"warp_field_distributed_greedy{suffix}.{ext}")

    # redefine the images again (unsharded this time)
    fixed_seg = Image.load_file(fixed_seg_path, is_segmentation=True, device='cpu')
    moving_seg = Image.load_file(moving_seg_path, is_segmentation=True, device='cpu')
    moving_seg_batch = BatchedImages([moving_seg]).to(parallel_state.get_device())
    fixed_seg_batch = BatchedImages([fixed_seg]).to(parallel_state.get_device())

    return {
        'fixed_batch': fixed_batch,
        'moving_batch': moving_batch,
        'fixed_seg_batch': fixed_seg_batch,
        'moving_seg_batch': moving_seg_batch,
        'output_dir': output_dir,
        'fixed_path': fixed_image_path,
        'moving_path': moving_image_path,
        'rank': rank,
        **results  # Include all registration results
    }

class TestDistributedGreedyRegistration:
    """Test suite for distributed greedy deformable registration."""
    
    def test_initial_dice_score(self, distributed_greedy_registration_results):
        """Test that initial Dice score is close to 0."""
        dice_score_before = 1 - dice_loss(distributed_greedy_registration_results['moving_seg_batch'](), 
                                      distributed_greedy_registration_results['fixed_seg_batch'](), 
                                      reduce=False).mean(0)
        mean_dice = dice_score_before.mean()
        if distributed_greedy_registration_results['rank'] == 0:
            logger.info(f"Dice score before registration: {mean_dice:.3f}")
        assert mean_dice < 0.6, f"Initial Dice score ({mean_dice:.3f}) should be small"
    
    def test_final_dice_score(self, distributed_greedy_registration_results):
        """Test that final Dice score is above threshold for both standard and ring sampler."""
        for suffix in ['_standard', '_ring']:
            moved_seg_batch = distributed_greedy_registration_results[f'moved_seg_batch{suffix}']
            dice_scores = 1 - dice_loss(moved_seg_batch, 
                                    distributed_greedy_registration_results['fixed_seg_batch'](), 
                                    reduce=False).mean(0)
            mean_dice = dice_scores.mean()
            
            if distributed_greedy_registration_results['rank'] == 0:
                logger.info(f"\nDistributed Greedy Registration Results ({suffix[1:].title()}):")
                logger.info(f"Average Dice Score: {mean_dice:.3f}")
                logger.info(f"Number of labels with Dice > 0.7: {sum(1 for d in dice_scores if d > 0.7)}")
                logger.info(f"Number of labels with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}")
                logger.info(f"Number of labels with Dice > 0.9: {sum(1 for d in dice_scores if d > 0.9)}")
            
            assert mean_dice > 0.75, f"Average Dice score ({mean_dice:.3f}) is below threshold for {suffix[1:]} method"
    
    def test_transform_consistency(self, distributed_greedy_registration_results):
        """Test that saved transform and antsApplyTransforms give consistent results for both methods."""
        # Only run this test on rank 0 since it involves file I/O
        if distributed_greedy_registration_results['rank'] == 0:
            for suffix in ['_standard', '_ring']:
                # Apply transform using antsApplyTransforms
                fixed_path = distributed_greedy_registration_results['fixed_path']
                moving_path = distributed_greedy_registration_results['moving_path']
                transform_path = distributed_greedy_registration_results[f'transform_path{suffix}']
                output_dir = distributed_greedy_registration_results['output_dir']
                ants_output = str(output_dir / f"moved_image_ants_distributed_greedy{suffix}.nii.gz")
                
                # Run antsApplyTransforms with the ANTs transform field
                cmd = f"antsApplyTransforms -d 3 -i {moving_path} -r {fixed_path} -t {transform_path} -o {ants_output}"
                subprocess.run(cmd, shell=True, check=True)
                
                # Load and compare results
                ants_result = Image.load_file(ants_output)
                ants_array = ants_result.array.squeeze().cpu().numpy()
                moved_array = distributed_greedy_registration_results[f'moved_image{suffix}'].squeeze().cpu().numpy()
                logger.info(f"\n{suffix[1:].title()} Method Results:")
                logger.info(f"ANTS array min: {ants_array.min()}, max: {ants_array.max()}")
                logger.info(f"Moved array min: {moved_array.min()}, max: {moved_array.max()}")
                
                # Compare using relative error
                rel_error = np.mean(np.abs(ants_array - moved_array) / (np.abs(moved_array) + 1e-6))
                logger.info(f"Relative error: {rel_error:.4f}")
                assert rel_error < 0.005, f"Transform consistency check failed for {suffix[1:]} method. Relative error: {rel_error:.4f}"

        # Ensure all processes are synced
        torch.distributed.barrier()

    def test_jacobian_determinant(self, distributed_greedy_registration_results):
        """Test that the transformation has positive Jacobian determinant everywhere for both methods."""
        rank = distributed_greedy_registration_results['rank']
        fixed_batch = distributed_greedy_registration_results['fixed_batch']
        moving_batch = distributed_greedy_registration_results['moving_batch']
        
        for suffix in ['_standard', '_ring']:
            reg = distributed_greedy_registration_results[f'reg{suffix}']
            
            # Get the warped coordinates
            coords = reg.get_warped_coordinates(fixed_batch, moving_batch)
            
            # Compute Jacobian determinant using imageutils.jacobian
            J = jacobian(coords, normalize=True).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1, :]
            
            # Compute determinant
            det = torch.linalg.det(J)
            
            # Check for negative determinants
            neg_det = (det < 0).float().mean()
            if rank == 0:
                logger.info(f"\n{suffix[1:].title()} Method:")
                logger.info(f"Percentage of negative Jacobian determinants: {neg_det:.4f}")
            
            # Assert that there are no negative determinants
            assert neg_det < 0.0001, f"Found {neg_det:.4f}% negative Jacobian determinants in {suffix[1:]} method"

            # Ensure all processes are synced
            torch.distributed.barrier()

if __name__ == '__main__':
    pytest.main([__file__])
