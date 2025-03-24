# import pytest
# import torch
# import SimpleITK as sitk
# import numpy as np
# import time
# from pathlib import Path

# # Import FireANTs components
# from fireants.registration import affine, rigid, greedy, syn
# from fireants.losses import similarity


# class TestMultiResolution:
#     """Test suite for multi-resolution behavior."""
    
#     def test_multi_resolution_affine(self, create_test_images, compute_similarity):
#         """Test multi-resolution behavior in affine registration."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(64, 64, 64))
        
#         # Compute initial similarity
#         initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
        
#         # Run registration with a single level
#         reg_single = affine.AffineRegistration(
#             n_iter=30,
#             learning_rate=0.01,
#             loss='mse',
#             n_levels=1
#         )
        
#         result_single = reg_single.register(fixed_img, moving_img)
        
#         # Run registration with multiple levels
#         reg_multi = affine.AffineRegistration(
#             n_iter=30,
#             learning_rate=0.01,
#             loss='mse',
#             n_levels=3
#         )
        
#         result_multi = reg_multi.register(fixed_img, moving_img)
        
#         # Compute registration quality
#         final_mse_single = compute_similarity(fixed_img, result_single.registered_image, metric='mse')
#         final_mse_multi = compute_similarity(fixed_img, result_multi.registered_image, metric='mse')
        
#         # Assertions
#         assert final_mse_single < initial_mse, "Single-level registration did not improve image similarity"
#         assert final_mse_multi < initial_mse, "Multi-level registration did not improve image similarity"
        
#         # Multi-resolution should be at least as good as single resolution
#         assert final_mse_multi <= final_mse_single * 1.1, "Multi-resolution registration should perform at least as well as single-resolution"
    
#     def test_multi_resolution_deformable(self, create_test_images, compute_similarity):
#         """Test multi-resolution behavior in deformable registration."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(64, 64, 64))
        
#         # Compute initial similarity
#         initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
        
#         # Run registration with a single level
#         reg_single = greedy.GreedyRegistration(
#             n_iter=20,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0,
#             n_levels=1
#         )
        
#         result_single = reg_single.register(fixed_img, moving_img)
        
#         # Run registration with multiple levels
#         reg_multi = greedy.GreedyRegistration(
#             n_iter=20,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0,
#             n_levels=3
#         )
        
#         result_multi = reg_multi.register(fixed_img, moving_img)
        
#         # Compute registration quality
#         final_mse_single = compute_similarity(fixed_img, result_single.registered_image, metric='mse')
#         final_mse_multi = compute_similarity(fixed_img, result_multi.registered_image, metric='mse')
        
#         # Assertions
#         assert final_mse_single < initial_mse, "Single-level registration did not improve image similarity"
#         assert final_mse_multi < initial_mse, "Multi-level registration did not improve image similarity"
        
#         # Multi-resolution should be at least as good as single resolution
#         assert final_mse_multi <= final_mse_single * 1.1, "Multi-resolution registration should perform at least as well as single-resolution"


# class TestSimilarityMetrics:
#     """Test suite for different similarity metrics."""
    
#     def test_different_metrics_affine(self, create_test_images):
#         """Test affine registration with different similarity metrics."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Test with MSE
#         reg_mse = affine.AffineRegistration(
#             n_iter=30,
#             learning_rate=0.01,
#             loss='mse'
#         )
        
#         result_mse = reg_mse.register(fixed_img, moving_img)
        
#         # Test with NCC
#         reg_ncc = affine.AffineRegistration(
#             n_iter=30,
#             learning_rate=0.01,
#             loss='ncc'
#         )
        
#         result_ncc = reg_ncc.register(fixed_img, moving_img)
        
#         # Both methods should produce valid transformation matrices
#         assert result_mse.transform.shape == (4, 4), "MSE registration failed to produce valid transform"
#         assert result_ncc.transform.shape == (4, 4), "NCC registration failed to produce valid transform"
    
#     def test_different_metrics_deformable(self, create_test_images):
#         """Test deformable registration with different similarity metrics."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Test with MSE
#         reg_mse = greedy.GreedyRegistration(
#             n_iter=20,
#             learning_rate=0.1,
#             loss='mse',
#             smoothing_sigma=1.0
#         )
        
#         result_mse = reg_mse.register(fixed_img, moving_img)
        
#         # Test with NCC
#         reg_ncc = greedy.GreedyRegistration(
#             n_iter=20,
#             learning_rate=0.1,
#             loss='ncc',
#             smoothing_sigma=1.0
#         )
        
#         result_ncc = reg_ncc.register(fixed_img, moving_img)
        
#         # Both methods should produce valid displacement fields
#         assert result_mse.displacement_field.shape[1] == 3, "MSE registration failed to produce valid displacement field"
#         assert result_ncc.displacement_field.shape[1] == 3, "NCC registration failed to produce valid displacement field"


# class TestRobustness:
#     """Test suite for robustness against noise and initialization."""
    
#     def test_noise_robustness(self, create_test_images, compute_similarity):
#         """Test registration robustness against noise."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Add noise to moving image
#         noise = torch.randn_like(moving_img) * 0.1
#         noisy_moving = moving_img + noise
#         noisy_moving = torch.clamp(noisy_moving, 0, 1)
        
#         # Run registration
#         reg = affine.AffineRegistration(
#             n_iter=50,
#             learning_rate=0.01,
#             loss='mse'
#         )
        
#         result = reg.register(fixed_img, noisy_moving)
        
#         # Compute initial and final similarity
#         initial_mse = compute_similarity(fixed_img, noisy_moving, metric='mse')
#         final_mse = compute_similarity(fixed_img, result.registered_image, metric='mse')
        
#         # Registration should still improve similarity despite noise
#         assert final_mse < initial_mse, "Registration failed to improve similarity with noisy input"
    
#     def test_initialization_robustness(self, create_test_images, compute_similarity):
#         """Test robustness against different initializations."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Run registration with default initialization
#         reg1 = affine.AffineRegistration(
#             n_iter=50,
#             learning_rate=0.01,
#             loss='mse'
#         )
        
#         result1 = reg1.register(fixed_img, moving_img)
        
#         # Run registration with custom initialization (identity + small random perturbation)
#         identity = torch.eye(4)
#         perturbation = torch.randn(4, 4) * 0.01
#         perturbation[3, :] = 0  # Keep last row as [0,0,0,1]
#         init_matrix = identity + perturbation
#         init_matrix[3, 3] = 1.0
        
#         reg2 = affine.AffineRegistration(
#             n_iter=50,
#             learning_rate=0.01,
#             loss='mse',
#             initial_transform=init_matrix
#         )
        
#         result2 = reg2.register(fixed_img, moving_img)
        
#         # Both registrations should improve similarity
#         initial_mse = compute_similarity(fixed_img, moving_img, metric='mse')
#         final_mse1 = compute_similarity(fixed_img, result1.registered_image, metric='mse')
#         final_mse2 = compute_similarity(fixed_img, result2.registered_image, metric='mse')
        
#         assert final_mse1 < initial_mse, "Registration with default initialization failed"
#         assert final_mse2 < initial_mse, "Registration with custom initialization failed"
        
#         # The results should be reasonably close to each other
#         assert abs(final_mse1 - final_mse2) < 0.05, "Different initializations led to vastly different results"


# class TestPerformance:
#     """Test suite for performance metrics."""
    
#     def test_execution_time(self, create_test_images):
#         """Test execution time is reasonable."""
#         # Create test images
#         fixed_img, moving_img = create_test_images(size=(32, 32, 32))
        
#         # Time affine registration
#         reg_affine = affine.AffineRegistration(n_iter=10, learning_rate=0.01, loss='mse')
#         start = time.time()
#         reg_affine.register(fixed_img, moving_img)
#         affine_time = time.time() - start
        
#         # Time greedy registration
#         reg_greedy = greedy.GreedyRegistration(n_iter=10, learning_rate=0.1, loss='mse', smoothing_sigma=1.0)
#         start = time.time()
#         reg_greedy.register(fixed_img, moving_img)
#         greedy_time = time.time() - start
        
#         # Assertions - times are reasonable and in expected order
#         assert affine_time < 10, f"Affine registration took too long: {affine_time:.2f}s"
#         assert greedy_time < 30, f"Greedy registration took too long: {greedy_time:.2f}s"
        
#         # Deformable should take longer than affine
#         assert greedy_time > affine_time, "Deformable registration should take longer than affine" 