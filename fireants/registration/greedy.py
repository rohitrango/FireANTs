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


import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union, Callable
from tqdm import tqdm
import SimpleITK as sitk

from fireants.utils.globals import MIN_IMG_SIZE
from fireants.io.image import BatchedImages, FakeBatchedImages
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.warputils import compositive_warp_inverse

## Deformable utils 
from fireants.registration.deformablemixin import DeformableMixin

class GreedyRegistration(AbstractRegistration, DeformableMixin):
    """Greedy deformable registration class for non-linear image alignment.
    
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
        optimizer_lr (float, optional): Learning rate for optimizer. Defaults to 0.5.
        integrator_n (Union[str, int], optional): Number of integration steps for geodesic shooting.
            Only used if deformation_type='geodesic'. Defaults to 7.
        mi_kernel_type (str, optional): Kernel type for MI loss. Defaults to 'b-spline'.
        cc_kernel_type (str, optional): Kernel type for CC loss. Defaults to 'rectangular'.
        cc_kernel_size (int, optional): Kernel size for CC loss. Defaults to 3.
        smooth_warp_sigma (float, optional): Gaussian smoothing sigma for warp field. Defaults to 0.5.
        smooth_grad_sigma (float, optional): Gaussian smoothing sigma for gradient field. Defaults to 1.0.
        loss_params (dict, optional): Additional parameters for loss function. Defaults to {}.
        reduction (str, optional): Loss reduction method - 'mean' or 'sum'. Defaults to 'sum'.
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_tolerance_iters (int, optional): Max iterations for convergence check. Defaults to 10.
        init_affine (Optional[torch.Tensor], optional): Initial affine transformation. Defaults to None.
        warp_reg (Optional[Union[Callable, nn.Module]], optional): Regularization on warp field. Defaults to None.
        displacement_reg (Optional[Union[Callable, nn.Module]], optional): Regularization on displacement field. 
            Defaults to None.
        blur (bool, optional): Whether to blur images during downsampling. Defaults to True.
        custom_loss (nn.Module, optional): Custom loss module. Defaults to None.

    Attributes:
        warp: Deformation model (StationaryVelocity or CompositiveWarp)
            * StationaryVelocity: Stores the stationary velocity field representation of a dense diffeomorphic transform
            * CompositiveWarp: Stores the compositive warp field representation of a dense diffeomorphic transform. This class is also used to store free-form deformations.
        affine (torch.Tensor): Initial affine transformation matrix
            * Shape: [N, D, D+1]
            * If init_affine is not provided, it is initialized to identity transform
        smooth_warp_sigma (float): Smoothing sigma for warp field

    """
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                deformation_type: str = 'compositive',
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 0.5, 
                integrator_n: Union[str, int] = 7,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                cc_kernel_size: int = 3,
                smooth_warp_sigma: float = 0.5,
                smooth_grad_sigma: float = 1.0,
                loss_params: dict = {},
                reduction: str = 'mean',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                init_affine: Optional[torch.Tensor] = None,
                warp_reg: Optional[Union[Callable, nn.Module]] = None,
                displacement_reg: Optional[Union[Callable, nn.Module]] = None,
                blur: bool = True,
                custom_loss: nn.Module = None, **kwargs) -> None:
        # initialize abstract registration
        # nn.Module.__init__(self)
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, reduction=reduction,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        self.dims = fixed_images.dims
        self.blur = blur
        self.reduction = reduction
        # specify regularizations
        self.warp_reg = warp_reg
        self.displacement_reg = displacement_reg
        self.deformation_type = deformation_type
        # specify deformation type
        if deformation_type == 'geodesic':
            warp = StationaryVelocity(fixed_images, moving_images, integrator_n=integrator_n, optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params,
                                    smoothing_grad_sigma=smooth_grad_sigma, init_scale=scales[0])
        elif deformation_type == 'compositive':
            warp = CompositiveWarp(fixed_images, moving_images, optimizer=optimizer, optimizer_lr=optimizer_lr, \
                                   optimizer_params=optimizer_params, \
                                   smoothing_grad_sigma=smooth_grad_sigma, smoothing_warp_sigma=smooth_warp_sigma, \
                                   init_scale=scales[0])
            smooth_warp_sigma = 0  # this work is delegated to compositive warp
        else:
            raise ValueError('Invalid deformation type: {}'.format(deformation_type))
        self.warp = warp
        self.smooth_warp_sigma = smooth_warp_sigma   # in voxels
        # initialize affine
        if init_affine is None:
            init_affine = torch.eye(self.dims+1, device=fixed_images.device).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, D+1, D+1]
        B, D1, D2 = init_affine.shape
        # affine can be [N, D, D+1] or [N, D+1, D+1]
        if D1 == self.dims+1 and D2 == self.dims+1:
            self.affine = init_affine.detach()
        elif D1 == self.dims and D2 == self.dims+1:
            # attach row to affine
            row = torch.zeros(self.opt_size, 1, self.dims+1, device=fixed_images.device)
            row[:, 0, -1] = 1.0
            self.affine = torch.cat([init_affine.detach(), row], dim=1)
        else:
            raise ValueError('Invalid initial affine shape: {}'.format(init_affine.shape))
    
    def get_inverse_warped_coordinates(self, fixed_images: Union[BatchedImages, FakeBatchedImages], \
                                             moving_images: Union[BatchedImages, FakeBatchedImages], \
                                             smooth_warp_sigma: float = 0, smooth_grad_sigma: float = 0,
                                             shape=None, displacement=False):
        ''' Get inverse warped coordinates for the moving image.

        This method is useful to analyse the effect of how the moving coordinates (fixed images) are transformed
        '''
        moving_arrays = moving_images()
        if shape is None:
            shape = moving_images.shape
        else:
            shape = [moving_arrays.shape[0], 1] + list(shape) 

        warp = self.warp.get_warp().detach().clone()
        warp = warp + F.affine_grid(torch.eye(self.dims, self.dims+1, device=warp.device)[None], [1, 1] + list(warp.shape[1:-1]), align_corners=True)
        warp_inv = compositive_warp_inverse(moving_images, warp, displacement=True, smooth_warp_sigma=smooth_warp_sigma, smooth_grad_sigma=smooth_grad_sigma)
        # resample if needed
        if tuple(warp_inv.shape[1:-1]) != tuple(shape[2:]):
            warp_inv = F.interpolate(warp_inv.permute(*self.warp.permute_vtoimg), size=shape[2:], mode='trilinear', align_corners=True).permute(*self.warp.permute_imgtov)
        
        # get affine transform
        fixed_t2p = fixed_images.get_torch2phy()
        moving_p2t = moving_images.get_phy2torch()
        # save initial affine transform to initialize grid 
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))
        affine_map_inv  = torch.linalg.inv(affine_map_init)
        # get A^-1 * v[y]
        print("inverse here")
        if self.dims == 2:
            warp_inv = torch.einsum('bhwx,byx->bhwy', warp_inv, affine_map_inv[:, :-1, :-1])
        elif self.dims == 3:
            warp_inv = torch.einsum('bhwdx,byx->bhwdy', warp_inv, affine_map_inv[:, :-1, :-1])
        else:
            raise ValueError('Invalid number of dimensions: {}'.format(self.dims))
        #### grid = A^-1 y - A^-1 b = apply A^-1 to regular grid
        grid = F.affine_grid(affine_map_inv[:, :-1], shape, align_corners=True)
        # grid = F.affine_grid(torch.eye(self.dims, self.dims+1, device=warp_inv.device)[None], \
        #     shape, align_corners=True)
        warp_inv = grid + warp_inv
        if displacement:
            grid = F.affine_grid(torch.eye(self.dims, self.dims+1, device=grid.device)[None], \
                                shape, align_corners=True)
            warp_inv = warp_inv - grid
        return warp_inv


    def get_warped_coordinates(self, fixed_images: Union[BatchedImages, FakeBatchedImages], \
                                     moving_images: Union[BatchedImages, FakeBatchedImages], \
                                     shape=None, displacement=False):
        """Get transformed coordinates for warping the moving image.

        Computes the coordinate transformation from fixed to moving image space
        using the current deformation parameters and optional initial affine transform.

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
                If displacement=True, displacement field, displacements in normalized [-1,1] space
                Shape: [N, H, W, [D], dims]
        """

        fixed_arrays = fixed_images() 
        if shape is None:
            shape = fixed_images.shape
        else:
            shape = [fixed_arrays.shape[0], 1] + list(shape) 

        fixed_t2p = fixed_images.get_torch2phy()
        moving_p2t = moving_images.get_phy2torch()
        # save initial affine transform to initialize grid 
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]
        # set affine coordinates
        fixed_image_affinecoords = F.affine_grid(affine_map_init, shape, align_corners=True)
        warp_field = self.warp.get_warp().clone()  # [N, HWD, 3]
        if tuple(warp_field.shape[1:-1]) != tuple(shape[2:]):
            # interpolate this
            warp_field = F.interpolate(warp_field.permute(*self.warp.permute_vtoimg), size=shape[2:], mode='trilinear', align_corners=True).permute(*self.warp.permute_imgtov)

        # smooth out the warp field if asked to 
        if self.smooth_warp_sigma > 0:
            warp_gaussian = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.dims, device=fixed_arrays.device) + self.smooth_warp_sigma)]
            warp_field = separable_filtering(warp_field.permute(*self.warp.permute_vtoimg), warp_gaussian).permute(*self.warp.permute_imgtov)
        # move these coordinates, and return them
        moved_coords = fixed_image_affinecoords + warp_field  # affine transform + warp field   
        if displacement:
            init_grid = F.affine_grid(torch.eye(self.dims, self.dims+1, device=moved_coords.device)[None], \
                            fixed_images.shape, align_corners=True)
            moved_coords = moved_coords - init_grid

        return moved_coords

    def optimize(self, save_transformed=False):
        """Optimize the deformation parameters.

        Performs multi-resolution optimization of the deformation field
        using the configured similarity metric and optimizer. The deformation
        field is optionally smoothed at each iteration.

        Args:
            save_transformed (bool, optional): Whether to save transformed images
                at each scale. Defaults to False.

        Returns:
            Optional[List[torch.Tensor]]: If save_transformed=True, returns list of
                transformed images at each scale. Otherwise returns None.
        """
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]
        # save initial affine transform to initialize grid 
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]

        # to save transformed images
        transformed_images = []
        # gaussian filter for smoothing the velocity field
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
            self.warp.set_size(size_down)
            # Get coordinates to transform
            fixed_image_affinecoords = F.affine_grid(affine_map_init, fixed_image_down.shape, align_corners=True)
            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)
            # reduce 
            if self.reduction == 'mean':
                scale_factor = 1
            else:
                scale_factor = np.prod(fixed_image_down.shape)

            for i in pbar:
                self.warp.set_zero_grad()
                warp_field = self.warp.get_warp()  # [N, HWD, 3]
                # smooth out the warp field if asked to 
                if self.smooth_warp_sigma > 0:
                    warp_field = separable_filtering(warp_field.permute(*self.warp.permute_vtoimg), warp_gaussian).permute(*self.warp.permute_imgtov)
                moved_coords = fixed_image_affinecoords + warp_field  # affine transform + warp field
                # moved_coords.retain_grad()
                # move the image
                moved_image = F.grid_sample(moving_image_blur, moved_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                loss = self.loss_fn(moved_image, fixed_image_down)
                # apply regularization on the warp field
                if self.displacement_reg is not None:
                    loss = loss + self.displacement_reg(warp_field)
                if self.warp_reg is not None:
                    loss = loss + self.warp_reg(moved_coords)
                loss.backward()
                if self.progress_bar:
                    pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, loss.item()/scale_factor))
                # optimize the velocity field
                self.warp.step()
                # check for convergence
                if self.convergence_monitor.converged(loss.item()):
                    break

            # save transformed image
            if save_transformed:
                transformed_images.append(moved_image.detach())

        if save_transformed:
            return transformed_images


if __name__ == '__main__':
    from fireants.io.image import Image
    img1 = Image.load_file('/data/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    img2 = Image.load_file('/data/BRATS2021/training/BraTS2021_00597/BraTS2021_00597_t1.nii.gz')
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    # get registration
    from time import time

    ## affine step
    from fireants.registration.affine import AffineRegistration
    transform = AffineRegistration([8, 4, 2, 1], [200, 100, 50, 20], fixed, moving, \
        loss_type='cc', optimizer='Adam', optimizer_lr=3e-4, optimizer_params={'momentum': 0.9})
    transform.optimize(save_transformed=False)

    reg = GreedyRegistration(scales=[4, 2, 1], iterations=[100, 50, 20], fixed_images=fixed, moving_images=moving,
                                optimizer='Adam', optimizer_lr=1e-3, init_affine=transform.get_affine_matrix().detach())
    a = time()
    reg.optimize()
    print(time() - a)