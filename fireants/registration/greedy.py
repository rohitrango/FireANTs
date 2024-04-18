import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm

from fireants.utils.globals import MIN_IMG_SIZE
from fireants.io.image import BatchedImages
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.geodesic import GeodesicShooting
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample

class GreedyRegistration(AbstractRegistration):
    '''
    This class implements greedy registration with a custom loss
    The moving image is interpolated to the fixed image grid, with an initial affine transform

    smooth_warp_sigma: how much to smooth the final warp field
    smooth_grad_sigma: how much to smooth the gradient of the final warp field  (this is similar to the Green's kernel)
    '''    
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                deformation_type: str = 'geodesic',
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 0.1, 
                integrator_n: Union[str, int] = 6,
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                cc_kernel_size: int = 3,
                smooth_warp_sigma: float = 0.5,
                smooth_grad_sigma: float = 0.5,
                loss_params: dict = {},
                reduction: str = 'sum',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                init_affine: Optional[torch.Tensor] = None,
                blur: bool = True,
                custom_loss: nn.Module = None) -> None:
        # initialize abstract registration
        # nn.Module.__init__(self)
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, reduction=reduction,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters)
        self.dims = fixed_images.dims
        self.blur = blur
        self.reduction = reduction
        # specify deformation type
        if deformation_type == 'geodesic':
            warp = GeodesicShooting(fixed_images, moving_images, integrator_n=integrator_n, optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params,
                                    smoothing_grad_sigma=smooth_grad_sigma, init_scale=scales[0])
        elif deformation_type == 'compositive':
            warp = CompositiveWarp(fixed_images, moving_images, optimizer=optimizer, optimizer_lr=optimizer_lr, optimizer_params=optimizer_params, \
                                   smoothing_grad_sigma=smooth_grad_sigma, smoothing_warp_sigma=smooth_warp_sigma, init_scale=scales[0])
            smooth_warp_sigma = 0  # this work is delegated to compositive warp
        else:
            raise ValueError('Invalid deformation type: {}'.format(deformation_type))
        self.warp = warp
        self.smooth_warp_sigma = smooth_warp_sigma   # in voxels
        # initialize affine
        if init_affine is None:
            init_affine = torch.eye(self.dims+1, device=fixed_images.device).unsqueeze(0).repeat(fixed_images.size(), 1, 1)  # [N, D, D+1]
        self.affine = init_affine.detach()
    
    def get_warped_coordinates(self, fixed_images: BatchedImages, moving_images: BatchedImages, shape=None):
        ''' given fixed and moving images, get warp '''
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
        return moved_coords

    def evaluate(self, fixed_images: BatchedImages, moving_images: BatchedImages, shape=None):
        ''' given a new set of fixed and moving images, warp the fixed image '''
        moving_arrays = moving_images()
        moved_coords = self.get_warped_coordinates(fixed_images, moving_images, shape=shape)
        moved_image = F.grid_sample(moving_arrays, moved_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
        return moved_image
    
    def optimize(self, save_transformed=False):
        ''' optimize the warp field to match the two images based on loss function '''
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]
        # save initial affine transform to initialize grid 
        # init_grid = torch.eye(self.dims, self.dims+1).to(self.fixed_images.device).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)  # [N, dims, dims+1]
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
            pbar = tqdm(range(iters))
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
                loss.backward()
                pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, loss.item()/scale_factor))
                # optimize the velocity field
                self.warp.step()
                # check for convergence
                if self.convergence_monitor.converged(loss.item()):
                    break

            # save transformed image
            if save_transformed:
                transformed_images.append(moved_image.detach().cpu())

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