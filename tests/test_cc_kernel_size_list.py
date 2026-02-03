# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.

"""
Test suite for kernel_size_list functionality in LocalNormalizedCrossCorrelationLoss.

This tests the multi-scale kernel size support added to cc.py to match fusedcc.py.
"""

import pytest
import torch
import logging

from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestKernelSizeListBasic:
    """Basic tests for kernel_size_list functionality."""

    def test_single_kernel_size_backward_compatibility(self):
        """Test that single kernel size (int) still works as before."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=5,
            kernel_type='rectangular'
        )
        assert loss.kernel_size == 5
        assert loss.kernel_size_list is None
        assert loss.kernel.shape[0] == 5

    def test_kernel_size_list_initialization(self):
        """Test initialization with kernel_size as a list."""
        kernel_sizes = [3, 5, 7]
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=kernel_sizes,
            kernel_type='rectangular'
        )
        assert loss.kernel_size_list == kernel_sizes
        assert loss.kernel_size == 3  # Should start with first element
        assert loss.kernel.shape[0] == 3

    def test_kernel_size_tuple_initialization(self):
        """Test initialization with kernel_size as a tuple."""
        kernel_sizes = (3, 5, 7)
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=kernel_sizes,
            kernel_type='rectangular'
        )
        assert loss.kernel_size_list == kernel_sizes
        assert loss.kernel_size == 3

    def test_2d_kernel_size_list(self):
        """Test kernel_size_list with 2D spatial dimensions."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=2,
            kernel_size=[3, 5],
            kernel_type='rectangular'
        )
        assert loss.ndim == 2
        assert loss.kernel_size_list == [3, 5]
        assert loss.kernel_size == 3


class TestScaleAndIterationMethods:
    """Tests for set_scales, set_iterations, and set_current_scale_and_iterations."""

    def test_set_scales_basic(self):
        """Test set_scales method with matching list lengths."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)
        assert loss.scales == scales

    def test_set_scales_without_kernel_size_list(self):
        """Test set_scales works when kernel_size is a single int."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=5,
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)
        assert loss.scales == scales
        # kernel_size should remain unchanged
        assert loss.kernel_size == 5

    def test_set_iterations(self):
        """Test set_iterations method."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],
            kernel_type='rectangular'
        )
        iterations = [100, 50, 25]
        loss.set_iterations(iterations)
        assert loss.iterations == iterations

    def test_set_current_scale_updates_kernel(self):
        """Test that set_current_scale_and_iterations updates kernel size correctly."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)

        # Check each scale updates the kernel correctly
        loss.set_current_scale_and_iterations(1.0, 100)
        assert loss.kernel_size == 3
        assert loss.kernel.shape[0] == 3

        loss.set_current_scale_and_iterations(0.5, 50)
        assert loss.kernel_size == 5
        assert loss.kernel.shape[0] == 5

        loss.set_current_scale_and_iterations(0.25, 25)
        assert loss.kernel_size == 7
        assert loss.kernel.shape[0] == 7

    def test_set_current_scale_no_change_when_same_size(self):
        """Test that kernel is not recreated when size doesn't change."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[5, 5, 5],  # Same size for all scales
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)
        
        original_kernel = loss.kernel
        loss.set_current_scale_and_iterations(0.5, 50)
        # Kernel should be the same object (no update needed)
        assert loss.kernel_size == 5

    def test_set_current_scale_without_kernel_size_list(self):
        """Test set_current_scale_and_iterations with single kernel_size."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=5,
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)
        
        # Should not raise error and kernel should remain unchanged
        loss.set_current_scale_and_iterations(0.5, 50)
        assert loss.kernel_size == 5


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_even_kernel_size_raises_error(self):
        """Test that even kernel size raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=4,
                kernel_type='rectangular'
            )

    def test_even_kernel_size_in_list_raises_error(self):
        """Test that even kernel size in list raises error during init."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            loss = LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3,
                kernel_size=[4, 5, 7],  # 4 is even
                kernel_type='rectangular'
            )

    def test_mismatched_kernel_size_list_and_scales_raises_error(self):
        """Test that mismatched lengths raise AssertionError."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],  # 3 elements
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5]  # 2 elements
        with pytest.raises(AssertionError, match="kernel_size_list must have the same length as scales"):
            loss.set_scales(scales)

    def test_update_kernel_even_size_raises_error(self):
        """Test that _update_kernel with even size raises error."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=5,
            kernel_type='rectangular'
        )
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            loss._update_kernel(6)


class TestKernelTypes:
    """Tests for different kernel types with kernel_size_list."""

    @pytest.mark.parametrize("kernel_type", ["rectangular", "triangular", "gaussian"])
    def test_kernel_types_with_list(self, kernel_type):
        """Test that all kernel types work with kernel_size_list."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],
            kernel_type=kernel_type
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)

        # Update to each scale and verify kernel is created
        for scale, expected_size in zip(scales, [3, 5, 7]):
            loss.set_current_scale_and_iterations(scale, 100)
            assert loss.kernel_size == expected_size
            assert loss.kernel.shape[0] == expected_size
            # Verify kernel sums to 1 (normalized)
            assert torch.isclose(loss.kernel.sum(), torch.tensor(1.0), atol=1e-5)


class TestForwardPass:
    """Tests for forward pass with kernel_size_list."""

    def test_forward_3d_with_kernel_size_list(self):
        """Test forward pass with 3D images and kernel_size_list."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],
            kernel_type='rectangular',
            reduction='mean'
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)

        img1 = torch.rand(1, 1, 32, 32, 32)
        img2 = torch.rand(1, 1, 32, 32, 32)

        # Test at each scale
        for scale in scales:
            loss.set_current_scale_and_iterations(scale, 100)
            output = loss(img1, img2)
            assert output.ndim == 0  # scalar output with mean reduction
            assert not torch.isnan(output)
            assert not torch.isinf(output)

    def test_forward_2d_with_kernel_size_list(self):
        """Test forward pass with 2D images and kernel_size_list."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=2,
            kernel_size=[3, 5],
            kernel_type='rectangular',
            reduction='mean'
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)

        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.rand(1, 1, 64, 64)

        for scale in scales:
            loss.set_current_scale_and_iterations(scale, 100)
            output = loss(img1, img2)
            assert output.ndim == 0
            assert not torch.isnan(output)

    def test_forward_no_reduction_with_kernel_size_list(self):
        """Test forward pass with reduction='none' and kernel_size_list."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5],
            kernel_type='rectangular',
            reduction='none'
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)

        img1 = torch.rand(2, 1, 16, 16, 16)
        img2 = torch.rand(2, 1, 16, 16, 16)

        for scale in scales:
            loss.set_current_scale_and_iterations(scale, 100)
            output = loss(img1, img2)
            assert output.shape == img1.shape

    def test_forward_consistent_values_after_scale_change(self):
        """Test that forward pass gives consistent values after changing scales back and forth."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5],
            kernel_type='rectangular',
            reduction='mean'
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)

        torch.manual_seed(42)
        img1 = torch.rand(1, 1, 32, 32, 32)
        img2 = torch.rand(1, 1, 32, 32, 32)

        # Compute at scale 1.0
        loss.set_current_scale_and_iterations(1.0, 100)
        output1_first = loss(img1, img2)

        # Change to scale 0.5 and compute
        loss.set_current_scale_and_iterations(0.5, 50)
        output2 = loss(img1, img2)

        # Change back to scale 1.0 and verify same result
        loss.set_current_scale_and_iterations(1.0, 100)
        output1_second = loss(img1, img2)

        assert torch.isclose(output1_first, output1_second, atol=1e-6)
        # Outputs should be different for different kernel sizes
        assert not torch.isclose(output1_first, output2, atol=1e-3)


class TestKernelVolume:
    """Tests for kernel volume computation with different kernel sizes."""

    def test_kernel_vol_normalized(self):
        """Test that kernel_vol is 1.0 (normalized kernel) for all kernel sizes."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5],
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)

        # Kernel is normalized, so vol should always be 1.0
        loss.set_current_scale_and_iterations(1.0, 100)
        assert torch.isclose(loss.kernel_vol, torch.tensor(1.0), atol=1e-5)

        loss.set_current_scale_and_iterations(0.5, 50)
        assert torch.isclose(loss.kernel_vol, torch.tensor(1.0), atol=1e-5)

    def test_kernel_nd_shape_updates(self):
        """Test that kernel_nd shape is updated when kernel size changes."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5],
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)

        loss.set_current_scale_and_iterations(1.0, 100)
        assert loss.kernel_nd.shape == (3, 3, 3)

        loss.set_current_scale_and_iterations(0.5, 50)
        assert loss.kernel_nd.shape == (5, 5, 5)


class TestImagePadding:
    """Tests for get_image_padding with kernel_size_list."""

    def test_image_padding_updates_with_kernel_size(self):
        """Test that get_image_padding returns correct value for current kernel size."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5, 7],
            kernel_type='rectangular'
        )
        scales = [1.0, 0.5, 0.25]
        loss.set_scales(scales)

        loss.set_current_scale_and_iterations(1.0, 100)
        assert loss.get_image_padding() == 1  # (3-1)//2

        loss.set_current_scale_and_iterations(0.5, 50)
        assert loss.get_image_padding() == 2  # (5-1)//2

        loss.set_current_scale_and_iterations(0.25, 25)
        assert loss.get_image_padding() == 3  # (7-1)//2


class TestMaskedMode:
    """Tests for masked mode with kernel_size_list."""

    @pytest.mark.parametrize("mask_mode", ["mult", "max"])
    def test_masked_mode_with_kernel_size_list(self, mask_mode):
        """Test that masked mode works with kernel_size_list."""
        loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=[3, 5],
            kernel_type='rectangular',
            masked=True,
            mask_mode=mask_mode
        )
        scales = [1.0, 0.5]
        loss.set_scales(scales)

        # Create image with mask channel
        img1 = torch.rand(1, 2, 16, 16, 16)  # 2 channels: image + mask
        img1[:, 1, :, :, :] = 1.0  # All ones mask
        img2 = torch.rand(1, 2, 16, 16, 16)
        img2[:, 1, :, :, :] = 1.0

        for scale in scales:
            loss.set_current_scale_and_iterations(scale, 100)
            output = loss(img1, img2)
            assert not torch.isnan(output)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
