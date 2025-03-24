# import pytest
# import torch
# import SimpleITK as sitk
# import numpy as np
# from pathlib import Path

# # Import FireANTs components
# from fireants.registration import greedy, syn
# from fireants.utils import diff_utils


# class TestJacobianDeterminant:
#     """Test suite for Jacobian determinant calculations."""
    
#     def test_jacobian_determinant_greedy(self, create_test_images):
#         """Test Jacobian determinant for greedy registration."""
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
        
#         # Calculate Jacobian determinant
#         jac_det = diff_utils.compute_jacobian_determinant(displacement_field)
        
#         # Check shape
#         assert jac_det.shape == fixed_img.shape[2:], "Jacobian determinant has incorrect shape"
        
#         # With reasonable smoothing_sigma, Jacobian determinant should be positive everywhere
#         # (or at least almost everywhere, allowing for small numerical errors)
#         min_jac_det = torch.min(jac_det)
#         percentage_negative = torch.sum(jac_det <= 0).item() / jac_det.numel() * 100
        
#         assert percentage_negative < 1.0, f"Too many negative Jacobian determinants: {percentage_negative:.2f}% (min: {min_jac_det:.5f})"
        
#         # The mean Jacobian determinant should be close to 1 for small deformations
#         mean_jac_det = torch.mean(jac_det)
#         assert 0.9 <= mean_jac_det <= 1.1, f"Mean Jacobian determinant too far from 1.0: {mean_jac_det:.5f}"
    
#     def test_jacobian_determinant_syn(self, create_test_images):
#         """Test Jacobian determinant for SyN registration."""
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
        
#         # Calculate Jacobian determinant
#         jac_det = diff_utils.compute_jacobian_determinant(forward_field)
        
#         # Check shape
#         assert jac_det.shape == fixed_img.shape[2:], "Jacobian determinant has incorrect shape"
        
#         # SyN should guarantee positive Jacobian determinants with enough smoothing
#         min_jac_det = torch.min(jac_det)
#         percentage_negative = torch.sum(jac_det <= 0).item() / jac_det.numel() * 100
        
#         assert percentage_negative < 0.1, f"Too many negative Jacobian determinants: {percentage_negative:.2f}% (min: {min_jac_det:.5f})"
        
#         # The mean Jacobian determinant should be close to 1 for diffeomorphic transforms
#         mean_jac_det = torch.mean(jac_det)
#         assert 0.95 <= mean_jac_det <= 1.05, f"Mean Jacobian determinant too far from 1.0: {mean_jac_det:.5f}"


# class TestDiffeomorphicProperties:
#     """Test suite for diffeomorphic properties."""
    
#     def test_syn_composition(self, create_test_images, compute_similarity):
#         """Test composition properties of SyN transformations."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Run SyN registration
#         reg = syn.SyNRegistration(
#             n_iter=30,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0
#         )
        
#         result = reg.register(fixed_img, moving_img)
#         forward_field = result.forward_field
#         inverse_field = result.inverse_field
        
#         # Compose forward and inverse fields
#         # This should result in an identity transform (zero displacement field)
#         composed_field = diff_utils.compose_displacement_fields(forward_field, inverse_field)
        
#         # The composed field should be approximately zero everywhere
#         composed_magnitude = torch.sqrt(torch.sum(composed_field**2, dim=1))
#         mean_magnitude = torch.mean(composed_magnitude)
#         max_magnitude = torch.max(composed_magnitude)
        
#         # Assert that the mean magnitude is close to zero
#         assert mean_magnitude < 0.5, f"Mean displacement magnitude after composition too high: {mean_magnitude:.5f}"
#         assert max_magnitude < 2.0, f"Max displacement magnitude after composition too high: {max_magnitude:.5f}"
    
#     def test_identity_transformation(self, create_test_images, compute_similarity):
#         """Test that identity transformation preserves the image."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Create an identity displacement field (all zeros)
#         identity_field = torch.zeros_like(moving_img)
#         identity_field = identity_field.repeat(1, 3, 1, 1, 1)  # 3 channels for x, y, z
        
#         # Apply the identity transform
#         from fireants.io import transforms
#         transformed_img = transforms.apply_transform(moving_img, identity_field)
        
#         # The transformed image should be identical to the original
#         mse = compute_similarity(moving_img, transformed_img, metric='mse')
#         assert mse < 1e-5, f"Identity transformation changed the image: MSE = {mse:.8f}"


# class TestRegularization:
#     """Test suite for transformation regularization."""
    
#     def test_smoothing_effect(self, create_test_images):
#         """Test the effect of smoothing on registration quality."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Run registration with low smoothing
#         reg_low = greedy.GreedyRegistration(
#             n_iter=20,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=0.1  # Low smoothing
#         )
        
#         result_low = reg_low.register(fixed_img, moving_img)
        
#         # Run registration with high smoothing
#         reg_high = greedy.GreedyRegistration(
#             n_iter=20,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=2.0  # High smoothing
#         )
        
#         result_high = reg_high.register(fixed_img, moving_img)
        
#         # Calculate Jacobian determinants
#         jac_det_low = diff_utils.compute_jacobian_determinant(result_low.displacement_field)
#         jac_det_high = diff_utils.compute_jacobian_determinant(result_high.displacement_field)
        
#         # Higher smoothing should result in fewer negative Jacobian determinants
#         negative_low = torch.sum(jac_det_low <= 0).item() / jac_det_low.numel() * 100
#         negative_high = torch.sum(jac_det_high <= 0).item() / jac_det_high.numel() * 100
        
#         assert negative_high <= negative_low, f"Higher smoothing resulted in more negative Jacobians: {negative_high:.2f}% vs {negative_low:.2f}%"
        
#         # Higher smoothing should also result in less extreme Jacobian values
#         std_low = torch.std(jac_det_low)
#         std_high = torch.std(jac_det_high)
        
#         assert std_high <= std_low, f"Higher smoothing resulted in more variable Jacobians: std {std_high:.4f} vs {std_low:.4f}" 