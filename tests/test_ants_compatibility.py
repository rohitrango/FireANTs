# import pytest
# import torch
# import SimpleITK as sitk
# import numpy as np
# import os
# import tempfile
# import shutil
# import subprocess
# from pathlib import Path

# # Import FireANTs components
# from fireants.registration import greedy, syn
# from fireants.io import transforms


# class TestANTsCompatibility:
#     """Test suite for ANTs format compatibility."""
    
#     @pytest.mark.skipif(shutil.which("antsApplyTransforms") is None,
#                         reason="ANTs not installed or not in PATH")
#     def test_ants_format_save_load(self, save_test_images, compute_similarity):
#         """Test saving and loading transformations in ANTs format."""
#         # Save test images to files
#         fixed_path, moving_path = save_test_images()
        
#         # Temporary directory for outputs
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Paths for outputs
#             fireants_warp = os.path.join(temp_dir, "fireants_warp.nii.gz")
#             fireants_applied = os.path.join(temp_dir, "fireants_applied.nii.gz")
#             ants_applied = os.path.join(temp_dir, "ants_applied.nii.gz")
            
#             # Run registration with FireANTs
#             reg = greedy.GreedyRegistration(
#                 n_iter=30,
#                 learning_rate=0.1,
#                 loss='mse',
#                 smoothing_sigma=1.0
#             )
            
#             result = reg.register_files(
#                 fixed_path, 
#                 moving_path,
#                 displacement_field_path=fireants_warp,
#                 registered_image_path=fireants_applied
#             )
            
#             # Check if files were created
#             assert os.path.exists(fireants_warp), "Displacement field was not saved"
#             assert os.path.exists(fireants_applied), "Registered image was not saved"
            
#             # Apply the same warp using ANTs
#             cmd = [
#                 "antsApplyTransforms",
#                 "-d", "3",
#                 "-i", moving_path,
#                 "-r", fixed_path,
#                 "-o", ants_applied,
#                 "-t", fireants_warp
#             ]
            
#             try:
#                 subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             except subprocess.CalledProcessError as e:
#                 pytest.fail(f"ANTs command failed: {e.stderr.decode()}")
            
#             # Check if ANTs applied transformation
#             assert os.path.exists(ants_applied), "ANTs failed to apply the transformation"
            
#             # Compare results from FireANTs and ANTs
#             fireants_img = sitk.ReadImage(fireants_applied)
#             ants_img = sitk.ReadImage(ants_applied)
            
#             fireants_np = sitk.GetArrayFromImage(fireants_img)
#             ants_np = sitk.GetArrayFromImage(ants_img)
            
#             # Compute similarity between the two registered images
#             similarity = compute_similarity(fireants_np, ants_np, metric='mse')
            
#             # The results should be very similar (small MSE)
#             assert similarity < 0.01, "FireANTs and ANTs results differ significantly"
    
#     def test_load_ants_transform(self, save_test_images, compute_similarity):
#         """Test loading ANTs transformations with FireANTs."""
#         # Save test images to files
#         fixed_path, moving_path = save_test_images()
        
#         # Temporary directory for outputs
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Paths for outputs
#             warp_path = os.path.join(temp_dir, "warp.nii.gz")
#             applied_path = os.path.join(temp_dir, "applied.nii.gz")
            
#             # Run registration to generate a transform
#             reg = greedy.GreedyRegistration(
#                 n_iter=30,
#                 learning_rate=0.1,
#                 loss='mse',
#                 smoothing_sigma=1.0
#             )
            
#             result = reg.register_files(
#                 fixed_path, 
#                 moving_path,
#                 displacement_field_path=warp_path,
#                 registered_image_path=applied_path
#             )
            
#             # Load the transform using FireANTs
#             loaded_transform = transforms.load_transform(warp_path)
            
#             # Check that the transform is correctly loaded
#             assert isinstance(loaded_transform, torch.Tensor), "Loaded transform should be a tensor"
#             assert loaded_transform.dim() >= 4, "Transform should have at least 4 dimensions"
#             assert loaded_transform.shape[1] == 3, "Transform should have 3 components (x, y, z)"
            
#             # Apply the loaded transform to the moving image
#             moving_img = sitk.ReadImage(moving_path)
#             moving_tensor = torch.from_numpy(sitk.GetArrayFromImage(moving_img)).float()
#             moving_tensor = moving_tensor.unsqueeze(0).unsqueeze(0)
            
#             # Apply transform
#             warped_img = transforms.apply_transform(moving_tensor, loaded_transform)
            
#             # Compare with the previously applied transform result
#             applied_img = sitk.ReadImage(applied_path)
#             applied_np = sitk.GetArrayFromImage(applied_img)
            
#             warped_np = warped_img.squeeze().detach().cpu().numpy()
            
#             # Compute similarity
#             similarity = compute_similarity(warped_np, applied_np, metric='mse')
            
#             # The results should be very similar
#             assert similarity < 0.01, "Applied transform doesn't match expected result"


# class TestInverseTransformations:
#     """Test suite for inverse transformations."""
    
#     def test_greedy_inverse_transform(self, create_test_images, compute_similarity):
#         """Test inverse transformation for greedy registration."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Run registration
#         reg = greedy.GreedyRegistration(
#             n_iter=30,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0
#         )
        
#         result = reg.register(fixed_img, moving_img)
#         displacement_field = result.displacement_field
        
#         # Get inverse displacement field
#         inverse_field = transforms.invert_displacement_field(displacement_field)
        
#         # Check shapes
#         assert inverse_field.shape == displacement_field.shape, "Inverse field has incorrect shape"
        
#         # Apply forward and then inverse transform to check round-trip consistency
#         warped_img = transforms.apply_transform(moving_img, displacement_field)
#         restored_img = transforms.apply_transform(warped_img, inverse_field)
        
#         # Compute similarity between original and restored images
#         similarity = compute_similarity(moving_img, restored_img, metric='mse')
        
#         # The round-trip transformation should approximately recover the original
#         assert similarity < 0.05, "Round-trip transformation differs significantly from original"
    
#     def test_syn_inverse_transform(self, create_test_images, compute_similarity):
#         """Test inverse transformation for SyN registration."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Run registration
#         reg = syn.SyNRegistration(
#             n_iter=30,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0
#         )
        
#         result = reg.register(fixed_img, moving_img)
#         forward_field = result.forward_field
#         inverse_field = result.inverse_field
        
#         # Apply forward and then inverse transform to check round-trip consistency
#         warped_img = transforms.apply_transform(moving_img, forward_field)
#         restored_img = transforms.apply_transform(warped_img, inverse_field)
        
#         # Compute similarity between original and restored images
#         similarity = compute_similarity(moving_img, restored_img, metric='mse')
        
#         # The round-trip transformation should approximately recover the original
#         assert similarity < 0.05, "Round-trip transformation differs significantly from original"
        
#         # Test in the other direction (fixed -> moving -> fixed)
#         warped_fixed = transforms.apply_transform(fixed_img, inverse_field)
#         restored_fixed = transforms.apply_transform(warped_fixed, forward_field)
        
#         # Compute similarity between original fixed and restored fixed
#         similarity_fixed = compute_similarity(fixed_img, restored_fixed, metric='mse')
        
#         # The round-trip transformation should approximately recover the original
#         assert similarity_fixed < 0.05, "Round-trip transformation of fixed image differs significantly from original" 