from tqdm import tqdm
import numpy as np
from typing import List, Optional, Union
import torch
from torch import nn
from cudants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from cudants.utils.globals import MIN_IMG_SIZE
from cudants.io.image import BatchedImages
from cudants.registration.abstract import AbstractRegistration
from cudants.registration.deformation.geodesic import GeodesicShooting
from cudants.losses.cc import gaussian_1d, separable_filtering
from cudants.utils.imageutils import downsample
from cudants.utils.util import compose_warp

class SyNRegistration(AbstractRegistration):
    '''
    This class implements symmetric registration with a custom loss
    The moving image and fixed image are registered to a fixed "mid space"

    smooth_warp_sigma: how much to smooth the final warp field
    smooth_grad_sigma: how much to smooth the gradient of the final warp field  (this is similar to the Green's kernel)
    '''    
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                deformation_type: str = 'geodesic',
                optimizer: str = 'SGD', optimizer_params: dict = {},
                optimizer_lr: float = 0.1, optimizer_momentum: float = 0.0,
                integrator_n: Union[str, int] = 10,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                smooth_warp_sigma: float = 0.5,
                smooth_grad_sigma: float = 0.5,
                cc_kernel_size: float = 3,
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol',
                init_affine: Optional[torch.Tensor] = None,
                blur: bool = True,
                custom_loss: nn.Module = None) -> None:
        # initialize abstract registration
        # nn.Module.__init__(self)
        super().__init__(scales, iterations, fixed_images, moving_images, loss_type, mi_kernel_type, cc_kernel_type, custom_loss, cc_kernel_size,
                         tolerance, max_tolerance_iters, tolerance_mode)
        self.dims = fixed_images.dims
        self.blur = blur
        # get warp
        if optimizer == 'SGD':
            optimizer_params['momentum'] = optimizer_momentum
        # else:
            # del optimizer_params.get('momentum')

        if deformation_type == 'geodesic':
            fwd_warp = GeodesicShooting(fixed_images, moving_images, integrator_n=integrator_n, 
                                        optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params,
                                        smoothing_grad_sigma=smooth_grad_sigma)
            rev_warp = GeodesicShooting(fixed_images, moving_images, integrator_n=integrator_n, 
                                        optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params,
                                        smoothing_grad_sigma=smooth_grad_sigma)
        else:
            raise ValueError('Invalid deformation type: {}'.format(deformation_type))
        self.fwd_warp = fwd_warp
        self.rev_warp = rev_warp
        self.smooth_warp_sigma = smooth_warp_sigma   # in voxels
        # initialize affine
        if init_affine is None:
            init_affine = torch.eye(self.dims+1, device=fixed_images.device).unsqueeze(0).repeat(fixed_images.size(), 1, 1)  # [N, D, D+1]
        self.affine = init_affine.detach()

    def optimize(self, save_transformed=False):
        ''' 
        optimize the warp fields to match the two images based on a common loss function 

        this time, we use both the forward and reverse warp fields
        '''
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
        warp_gaussian = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.dims, device=fixed_arrays.device) + self.smooth_warp_sigma)]
        # multi-scale optimization
        for scale, iters in zip(self.scales, self.iterations):
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
            pbar = tqdm(range(iters))
            for i in pbar:
                # set zero grads
                self.fwd_warp.set_zero_grad()
                self.rev_warp.set_zero_grad()
                # get warp fields and smooth them
                fwd_warp_field = self.fwd_warp.get_warp()  # [N, HWD, 3]
                rev_warp_field = self.rev_warp.get_warp()
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
                loss.backward()
                pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, loss.item()))
                # optimize the velocity field
                self.fwd_warp.step()
                self.rev_warp.step()
            # save transformed image
            if save_transformed:
                with torch.no_grad():
                    fwd_warp_field = self.fwd_warp.get_warp()  # [N, HWD, 3]
                    rev_inv_warp_field = self.rev_warp.get_inverse_warp()
                    # smooth them out
                    fwd_warp_field = separable_filtering(fwd_warp_field.permute(*self.fwd_warp.permute_vtoimg), warp_gaussian).permute(*self.fwd_warp.permute_imgtov)
                    rev_inv_warp_field = separable_filtering(rev_inv_warp_field.permute(*self.rev_warp.permute_vtoimg), warp_gaussian).permute(*self.rev_warp.permute_imgtov)
                    # compose the two warp fields
                    composed_warp = compose_warp(fwd_warp_field, rev_inv_warp_field, fixed_image_vgrid)  # compose two displacement fields with initial grid
                    moved_coords_final = fixed_image_affinecoords + composed_warp
                    moved_image = F.grid_sample(moving_image_blur, moved_coords_final, mode='bilinear', align_corners=True)
                    transformed_images.append(moved_image.detach().cpu())
                    # transformed_images.append(moved_image_warp.detach().cpu())

        if save_transformed:
            return transformed_images