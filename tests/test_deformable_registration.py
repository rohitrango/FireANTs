# import pytest
# import torch
# import SimpleITK as sitk
# import numpy as np
# import os
# from pathlib import Path
# import tempfile
# import shutil

# # Import FireANTs components
# from fireants.registration import greedy, syn
# from fireants.io import transforms


# class TestGreedyRegistration:
#     """Test suite for greedy deformable registration."""
    
#     def test_greedy_registration_tensor(self, create_test_images, compute_similarity):
#         """Test greedy deformable registration with tensor inputs."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Compute initial similarity
#         initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
        
#         # Run registration
#         reg = greedy.GreedyRegistration(
#             n_iter=30,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0
#         )
        
#         result = reg.register(fixed_img, moving_img)
#         registered_img = result.registered_image
#         displacement_field = result.displacement_field
        
#         # Compute registration quality
#         final_mse = compute_similarity(fixed_img, registered_img, metric='mse')
        
#         # Assertions
#         assert final_mse < initial_mse, "Registration did not improve image similarity"
#         assert displacement_field.shape[2:] == fixed_img.shape[2:], "Displacement field has incorrect spatial dimensions"
#         assert displacement_field.shape[1] == 3, "Displacement field should have 3 components (x, y, z)"
    
#     def test_greedy_registration_file(self, save_test_images, compute_similarity):
#         """Test greedy deformable registration with file inputs."""
#         # Save test images to files
#         fixed_path, moving_path = save_test_images()
        
#         # Load images to compute initial similarity
#         fixed_sitk = sitk.ReadImage(fixed_path)
#         moving_sitk = sitk.ReadImage(moving_path)
#         fixed_np = sitk.GetArrayFromImage(fixed_sitk)
#         moving_np = sitk.GetArrayFromImage(moving_sitk)
        
#         initial_mse = compute_similarity(fixed_np, moving_np, metric='mse')
        
#         # Temporary directory for outputs
#         with tempfile.TemporaryDirectory() as temp_dir:
#             out_warp = os.path.join(temp_dir, "warp.nii.gz")
#             out_img = os.path.join(temp_dir, "registered.nii.gz")
            
#             # Run registration
#             reg = greedy.GreedyRegistration(
#                 n_iter=30,
#                 learning_rate=0.1,
#                 loss='mse',
#                 smoothing_sigma=1.0
#             )
            
#             result = reg.register_files(
#                 fixed_path, 
#                 moving_path,
#                 displacement_field_path=out_warp,
#                 registered_image_path=out_img
#             )
            
#             # Check if output files were created
#             assert os.path.exists(out_warp), "Displacement field file was not created"
#             assert os.path.exists(out_img), "Registered image file was not created"
            
#             # Load registered image
#             registered_sitk = sitk.ReadImage(out_img)
#             registered_np = sitk.GetArrayFromImage(registered_sitk)
            
#             # Compute registration quality
#             final_mse = compute_similarity(fixed_np, registered_np, metric='mse')
            
#             # Assertions
#             assert final_mse < initial_mse, "File-based registration did not improve image similarity"
            
#             # Load displacement field and check properties
#             warp_sitk = sitk.ReadImage(out_warp)
#             assert warp_sitk.GetDimension() == 3, "Displacement field should be 3D"
#             assert warp_sitk.GetNumberOfComponentsPerPixel() == 3, "Displacement field should have 3 components per voxel"


# class TestSyNRegistration:
#     """Test suite for SyN deformable registration."""
    
#     def test_syn_registration_tensor(self, create_test_images, compute_similarity):
#         """Test SyN deformable registration with tensor inputs."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Compute initial similarity
#         initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
        
#         # Run registration
#         reg = syn.SyNRegistration(
#             n_iter=30,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0
#         )
        
#         result = reg.register(fixed_img, moving_img)
#         registered_img = result.registered_image
#         forward_field = result.forward_field
#         inverse_field = result.inverse_field
        
#         # Compute registration quality
#         final_mse = compute_similarity(fixed_img, registered_img, metric='mse')
        
#         # Assertions
#         assert final_mse < initial_mse, "Registration did not improve image similarity"
#         assert forward_field.shape[2:] == fixed_img.shape[2:], "Forward field has incorrect spatial dimensions"
#         assert inverse_field.shape[2:] == fixed_img.shape[2:], "Inverse field has incorrect spatial dimensions"
#         assert forward_field.shape[1] == 3, "Forward field should have 3 components (x, y, z)"
#         assert inverse_field.shape[1] == 3, "Inverse field should have 3 components (x, y, z)"
    
#     def test_syn_registration_file(self, save_test_images, compute_similarity):
#         """Test SyN deformable registration with file inputs."""
#         # Save test images to files
#         fixed_path, moving_path = save_test_images()
        
#         # Load images to compute initial similarity
#         fixed_sitk = sitk.ReadImage(fixed_path)
#         moving_sitk = sitk.ReadImage(moving_path)
#         fixed_np = sitk.GetArrayFromImage(fixed_sitk)
#         moving_np = sitk.GetArrayFromImage(moving_sitk)
        
#         initial_mse = compute_similarity(fixed_np, moving_np, metric='mse')
        
#         # Temporary directory for outputs
#         with tempfile.TemporaryDirectory() as temp_dir:
#             out_forward = os.path.join(temp_dir, "forward.nii.gz")
#             out_inverse = os.path.join(temp_dir, "inverse.nii.gz")
#             out_img = os.path.join(temp_dir, "registered.nii.gz")
            
#             # Run registration
#             reg = syn.SyNRegistration(
#                 n_iter=30,
#                 learning_rate=0.1,
#                 loss='mse',
#                 smoothing_sigma=1.0
#             )
            
#             result = reg.register_files(
#                 fixed_path, 
#                 moving_path,
#                 forward_field_path=out_forward,
#                 inverse_field_path=out_inverse,
#                 registered_image_path=out_img
#             )
            
#             # Check if output files were created
#             assert os.path.exists(out_forward), "Forward field file was not created"
#             assert os.path.exists(out_inverse), "Inverse field file was not created"
#             assert os.path.exists(out_img), "Registered image file was not created"
            
#             # Load registered image
#             registered_sitk = sitk.ReadImage(out_img)
#             registered_np = sitk.GetArrayFromImage(registered_sitk)
            
#             # Compute registration quality
#             final_mse = compute_similarity(fixed_np, registered_np, metric='mse')
            
#             # Assertions
#             assert final_mse < initial_mse, "File-based registration did not improve image similarity"
            
#             # Load fields and check properties
#             forward_sitk = sitk.ReadImage(out_forward)
#             inverse_sitk = sitk.ReadImage(out_inverse)
            
#             assert forward_sitk.GetDimension() == 3, "Forward field should be 3D"
#             assert inverse_sitk.GetDimension() == 3, "Inverse field should be 3D"
#             assert forward_sitk.GetNumberOfComponentsPerPixel() == 3, "Forward field should have 3 components per voxel"
#             assert inverse_sitk.GetNumberOfComponentsPerPixel() == 3, "Inverse field should have 3 components per voxel" 