# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


from tqdm import tqdm
import numpy as np
from typing import List, Optional, Union, Callable
import torch
from torch import nn
from fireants.io.image import BatchedImages, FakeBatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from fireants.utils.globals import MIN_IMG_SIZE
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.util import compose_warp
from fireants.utils.warputils import compositive_warp_inverse

from fireants.registration.deformablemixin import DeformableMixin

class SyNRegistration(AbstractRegistration, DeformableMixin):
    """Symmetric Normalization (SyN) registration class for non-linear image alignment.

    This class implements symmetric diffeomorphic registration between fixed and moving images.
    Unlike greedy registration, SyN optimizes two deformation fields simultaneously to map
    both images to a common midpoint space, ensuring inverse consistency. The final
    transformation is composed of these bidirectional warps.

    Args:
        scales (List[int]): Downsampling factors for multi-resolution optimization.
            Must be in descending order (e.g. [4,2,1]).
        iterations (List[float]): Number of iterations to perform at each scale.
            Must be same length as scales.
        fixed_images (BatchedImages): Fixed/reference images to register to.
        moving_images (BatchedImages): Moving images to be registered.
        loss_type (str, optional): Similarity metric to use. Defaults to "cc".
        deformation_type (str, optional): Type of deformation model - 'geodesic' or 'compositive'. 
            Defaults to 'compositive'.
        optimizer (str, optional): Optimization algorithm - 'SGD' or 'Adam'. Defaults to 'Adam'.
        optimizer_params (dict, optional): Additional parameters for optimizer. Defaults to {}.
        optimizer_lr (float, optional): Learning rate for optimizer. Defaults to 0.1.
        integrator_n (Union[str, int], optional): Number of integration steps for geodesic shooting.
            Only used if deformation_type='geodesic'. Defaults to 10.
        mi_kernel_type (str, optional): Kernel type for MI loss. Defaults to 'b-spline'.
        cc_kernel_type (str, optional): Kernel type for CC loss. Defaults to 'rectangular'.
        smooth_warp_sigma (float, optional): Gaussian smoothing sigma for warp field. Defaults to 0.25.
        smooth_grad_sigma (float, optional): Gaussian smoothing sigma for gradient field. Defaults to 1.0.
        reduction (str, optional): Loss reduction method - 'mean' or 'sum'. Defaults to 'sum'.
        cc_kernel_size (float, optional): Kernel size for CC loss. Defaults to 3.
        loss_params (dict, optional): Additional parameters for loss function. Defaults to {}.
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_tolerance_iters (int, optional): Max iterations for convergence check. Defaults to 10.
        init_affine (Optional[torch.Tensor], optional): Initial affine transformation. Defaults to None.
        warp_reg (Optional[Union[Callable, nn.Module]], optional): Regularization on warp field. Defaults to None.
        displacement_reg (Optional[Union[Callable, nn.Module]], optional): Regularization on displacement field.
            Defaults to None.
        blur (bool, optional): Whether to blur images during downsampling. Defaults to True.
        custom_loss (nn.Module, optional): Custom loss module. Defaults to None.

    Attributes:
        fwd_warp: Forward deformation model (StationaryVelocity or CompositiveWarp)
        rev_warp: Reverse deformation model (StationaryVelocity or CompositiveWarp)
        affine (torch.Tensor): Initial affine transformation matrix
        smooth_warp_sigma (float): Smoothing sigma for warp field
    """
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                deformation_type: str = 'compositive',
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 0.1, 
                integrator_n: Union[str, int] = 10,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                smooth_warp_sigma: float = 0.25,
                smooth_grad_sigma: float = 1.0,
                reduction: str = 'mean',
                cc_kernel_size: float = 3,
                loss_params: dict = {},
                tolerance: float = 1e-6, max_tolerance_iters: int = 10,
                init_affine: Optional[torch.Tensor] = None,
                warp_reg: Optional[Union[Callable, nn.Module]] = None,
                displacement_reg: Optional[Union[Callable, nn.Module]] = None,
                blur: bool = True,
                custom_loss: nn.Module = None, **kwargs) -> None:
        # initialize abstract registration
        # nn.Module.__init__(self)
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, reduction=reduction,
                         loss_params=loss_params,
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, cc_kernel_size=cc_kernel_size,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        self.dims = fixed_images.dims
        self.blur = blur
        self.reduction = reduction
        # specify regularizations
        self.warp_reg = warp_reg
        self.displacement_reg = displacement_reg

        if deformation_type == 'geodesic':
            fwd_warp = StationaryVelocity(fixed_images, moving_images, integrator_n=integrator_n, 
                                        optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params,
                                        smoothing_grad_sigma=smooth_grad_sigma)
            rev_warp = StationaryVelocity(fixed_images, moving_images, integrator_n=integrator_n, 
                                        optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params,
                                        smoothing_grad_sigma=smooth_grad_sigma)
        elif deformation_type == 'compositive':
            fwd_warp = CompositiveWarp(fixed_images, moving_images, optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params, \
                                   smoothing_grad_sigma=smooth_grad_sigma, smoothing_warp_sigma=smooth_warp_sigma)
            rev_warp = CompositiveWarp(fixed_images, moving_images, optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params, \
                                   smoothing_grad_sigma=smooth_grad_sigma, smoothing_warp_sigma=smooth_warp_sigma)
            smooth_warp_sigma = 0  # this work is delegated to compositive warp
        else:
            raise ValueError('Invalid deformation type: {}'.format(deformation_type))
        self.fwd_warp = fwd_warp
        self.rev_warp = rev_warp
        self.smooth_warp_sigma = smooth_warp_sigma   # in voxels
        # initialize affine
        if init_affine is None:
            init_affine = torch.eye(self.dims+1, device=fixed_images.device).unsqueeze(0).repeat(fixed_images.size(), 1, 1)  # [N, D+1, D+1]
        B, D1, D2 = init_affine.shape
        if D1 == self.dims+1 and D2 == self.dims+1:
            self.affine = init_affine.detach()
        elif D1 == self.dims and D2 == self.dims+1:
            # attach row to affine
            row = torch.zeros(self.opt_size, 1, self.dims+1, device=fixed_images.device)
            row[:, 0, -1] = 1.0
            self.affine = torch.cat([init_affine.detach(), row], dim=1)
        else:
            raise ValueError('Invalid initial affine shape: {}'.format(init_affine.shape))


    def get_warped_coordinates(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None, displacement=False):
        """Get transformed coordinates for warping the moving image.

        Computes the coordinate transformation from fixed to moving image space
        by composing the forward and inverse warps through the midpoint space.
        Includes initial affine transformation.

        Args:
            fixed_images (BatchedImages): Fixed reference images
            moving_images (BatchedImages): Moving images to be transformed
            shape (Optional[tuple]): Output shape for coordinate grid.
                Defaults to fixed image shape.
            displacement (bool, optional): Whether to return displacement field instead of
                transformed coordinates. Defaults to False.

        Returns:
            torch.Tensor: If displacement=False, transformed coordinates in normalized [-1,1] space
                Shape: [N, H, W, [D], dims]
                If displacement=True, displacement field in normalized [-1,1] space
                Shape: [N, H, W, [D], dims]
        """
        fixed_arrays = fixed_images()

        if shape is None:
            shape = fixed_images.shape
        else:
            shape = [fixed_arrays.shape[0], 1] + list(shape) 

        fixed_t2p = fixed_images.get_torch2phy()
        moving_p2t = moving_images.get_phy2torch()
        # fixed_size = fixed_arrays.shape[2:]
        # save init transform
        init_grid = torch.eye(self.dims, self.dims+1).to(fixed_images.device).unsqueeze(0).repeat(fixed_images.size(), 1, 1)  # [N, dims, dims+1]
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]
        fixed_image_affinecoords = F.affine_grid(affine_map_init, fixed_arrays.shape, align_corners=True)
        fixed_image_vgrid  = F.affine_grid(init_grid, fixed_arrays.shape, align_corners=True)
        # get warps
        fwd_warp_field = self.fwd_warp.get_warp()  # [N, HWD, 3]
        rev_warp_field = self.rev_warp.get_warp()
        if tuple(fwd_warp_field.shape[1:-1]) != tuple(shape[2:]):
            # interpolate this
            fwd_warp_field = F.interpolate(
                fwd_warp_field.permute(*self.fwd_warp.permute_vtoimg),
                size=shape[2:],
                mode="trilinear",
                align_corners=True,
            ).permute(*self.fwd_warp.permute_imgtov)
        if tuple(rev_warp_field.shape[1:-1]) != tuple(shape[2:]):
            rev_warp_field = F.interpolate(
                self.rev_warp.get_warp().permute(*self.rev_warp.permute_vtoimg),
                size=shape[2:],
                mode="trilinear",
                align_corners=True,
            ).permute(*self.rev_warp.permute_imgtov)

        rev_inv_warp_field = compositive_warp_inverse(fixed_images, rev_warp_field + fixed_image_vgrid, displacement=True)



        # # smooth them out
        if self.smooth_warp_sigma > 0:
            warp_gaussian = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.dims, device=fixed_arrays.device) + self.smooth_warp_sigma)]
            fwd_warp_field = separable_filtering(fwd_warp_field.permute(*self.fwd_warp.permute_vtoimg), warp_gaussian).permute(*self.fwd_warp.permute_imgtov)
            rev_inv_warp_field = separable_filtering(rev_inv_warp_field.permute(*self.rev_warp.permute_vtoimg), warp_gaussian).permute(*self.rev_warp.permute_imgtov)
        # # compose the two warp fields
        composed_warp = compose_warp(fwd_warp_field, rev_inv_warp_field, fixed_image_vgrid)
        moved_coords_final = fixed_image_affinecoords + composed_warp
        if displacement:
            init_grid = F.affine_grid(torch.eye(self.dims, self.dims+1, device=moved_coords_final.device)[None], \
                            fixed_images.shape, align_corners=True)
            moved_coords_final = moved_coords_final - init_grid
        return moved_coords_final

    def get_inverse_warped_coordinates(self, fixed_images: Union[BatchedImages, FakeBatchedImages], moving_images: Union[BatchedImages, FakeBatchedImages], shape=None):
        raise NotImplementedError('Inverse warp not implemented for SyN registration')


    def optimize(self, save_transformed=False):
        """Optimize the symmetric deformation parameters.

        Performs multi-resolution optimization of both forward and reverse deformation fields
        using the configured similarity metric and optimizer. The deformation fields map
        both images to a common midpoint space. The fields are optionally smoothed at each
        iteration.

        Args:
            save_transformed (bool, optional): Whether to save transformed images
                at each scale. Defaults to False.

        Returns:
            Optional[List[torch.Tensor]]: If save_transformed=True, returns list of
                transformed images at each scale. Otherwise returns None.

        Note:
            The optimization alternates between updating the forward and reverse warps.
            The final transformation is composed of these bidirectional warps through
            the midpoint space.
        """
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]

        # save initial affine transform to initialize grid for the fixed image and moving image
        init_grid = torch.eye(self.dims, self.dims+1).to(self.fixed_images.device).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)  # [N, dims, dims+1]
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]

        # to save transformed images
        transformed_images = []
        # gaussian filter for smoothing the velocity field
        if self.smooth_warp_sigma > 0:
            warp_gaussian = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.dims, device=fixed_arrays.device) + self.smooth_warp_sigma)]
        # multi-scale optimization
        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            # resize images 
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device)
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_image_down = downsample(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
                moving_image_blur = separable_filtering(moving_arrays, gaussians)
            else:
                fixed_image_down = F.interpolate(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, align_corners=True)
                moving_image_blur = moving_arrays

            #### Set size for warp field
            self.fwd_warp.set_size(size_down)
            self.rev_warp.set_size(size_down)

            # Get coordinates to transform
            fixed_image_affinecoords = F.affine_grid(affine_map_init, fixed_image_down.shape, align_corners=True)
            fixed_image_vgrid  = F.affine_grid(init_grid, fixed_image_down.shape, align_corners=True)
            #### Optimize
            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)
            if self.reduction == 'mean':
                scale_factor = 1
            else:
                scale_factor = np.prod(fixed_image_down.shape)
            for i in pbar:
                # set zero grads
                self.fwd_warp.set_zero_grad()
                self.rev_warp.set_zero_grad()
                # get warp fields and smooth them
                fwd_warp_field = self.fwd_warp.get_warp()  # [N, HWD, 3]
                rev_warp_field = self.rev_warp.get_warp()
                # smooth if required
                if self.smooth_warp_sigma > 0:
                    fwd_warp_field = separable_filtering(fwd_warp_field.permute(*self.fwd_warp.permute_vtoimg), warp_gaussian).permute(*self.fwd_warp.permute_imgtov)
                    rev_warp_field = separable_filtering(rev_warp_field.permute(*self.rev_warp.permute_vtoimg), warp_gaussian).permute(*self.rev_warp.permute_imgtov)
                # moved and fixed coords
                moved_coords = fixed_image_affinecoords + fwd_warp_field  # affine transform + warp field
                fixed_coords = fixed_image_vgrid + rev_warp_field
                # warp the "moving image" to moved_image_warp and fixed to "fixed image warp"
                moved_image_warp = F.grid_sample(moving_image_blur, moved_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                fixed_image_warp = F.grid_sample(fixed_image_down, fixed_coords, mode='bilinear', align_corners=True)
                # compute loss
                loss = self.loss_fn(moved_image_warp, fixed_image_warp) 
                # add regularization
                if self.warp_reg is not None:
                    loss = loss + self.warp_reg(moved_coords) + self.warp_reg(fixed_coords)
                if self.displacement_reg is not None:
                    loss = loss + self.displacement_reg(fwd_warp_field) + self.displacement_reg(rev_warp_field)
                # backward
                loss.backward()
                if self.progress_bar:
                    pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, loss.item()/scale_factor))
                # optimize the deformations
                self.fwd_warp.step()
                self.rev_warp.step()
                if self.convergence_monitor.converged(loss.item()):
                    break

            # save transformed image
            if save_transformed:
                fwd_warp_field = self.fwd_warp.get_warp()  # [N, HWD, 3]
                # rev_inv_warp_field = self.rev_warp.get_inverse_warp(n_iters=50, debug=True, lr=0.1)
                rev_inv_warp_field = compositive_warp_inverse(self.fixed_images, self.rev_warp.get_warp() + fixed_image_vgrid, displacement=True,)
                # # smooth them out
                if self.smooth_warp_sigma > 0:
                    fwd_warp_field = separable_filtering(fwd_warp_field.permute(*self.fwd_warp.permute_vtoimg), warp_gaussian).permute(*self.fwd_warp.permute_imgtov)
                    rev_inv_warp_field = separable_filtering(rev_inv_warp_field.permute(*self.rev_warp.permute_vtoimg), warp_gaussian).permute(*self.rev_warp.permute_imgtov)

                # # compose the two warp fields
                composed_warp = compose_warp(fwd_warp_field, rev_inv_warp_field, fixed_image_vgrid)
                moved_coords_final = fixed_image_affinecoords + composed_warp
                moved_image = F.grid_sample(moving_image_blur, moved_coords_final, mode='bilinear', align_corners=True)
                transformed_images.append(moved_image.detach())
                ## compose twice
                # moved_image = F.grid_sample(moved_image_warp, rev_inv_warp_field + fixed_image_vgrid, mode='bilinear', align_corners=True)
                # transformed_images.append(moved_image.detach().cpu())
                # transformed_images.append(moved_image_warp.detach().cpu())
                # transformed_images.append(fixed_image_warp.detach().cpu())

                ## debug
                # transformed_images.append(moved_image_warp.detach().cpu())
                # fixed_warp_and_inverse = compose_warp(rev_inv_warp_field, rev_warp_field, fixed_image_vgrid)
                # fixed_orig_image = F.grid_sample(fixed_image_warp, fixed_warp_and_inverse + fixed_image_vgrid, mode='bilinear', align_corners=True)

                # fixed_orig_image = F.grid_sample(fixed_image_warp, rev_inv_warp_field + fixed_image_vgrid, mode='bilinear', align_corners=True)
                # transformed_images.append(fixed_image_warp.detach().cpu())
                # transformed_images.append(fixed_orig_image.detach().cpu())

        if save_transformed:
            return transformed_images
