from cudants.registration.abstract import AbstractRegistration
from typing import List, Optional
import torch
from torch import nn
from cudants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from cudants.utils.globals import MIN_IMG_SIZE
from tqdm import tqdm
import numpy as np
from cudants.utils.opticalflow import OpticalFlow
from cudants.losses.cc import gaussian_1d, separable_filtering
from cudants.utils.imageutils import downsample
from cudants.utils.imageutils import scaling_and_squaring
from cudants.utils.imageutils import lie_bracket

class LogDemonsRegistration(AbstractRegistration):
    '''
    This class implements multi-scale log-demons registration

    Returns the warp field in pixel coordinates because optical flow works in pixel coordinates
    TODO: Investigate if we can change this to physical coordinates 
    '''    
    def __init__(self, scales: List[int], iterations: List[float], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, tolerance_mode: str = 'atol',
                init_affine: Optional[torch.Tensor] = None,
                custom_loss: nn.Module = None,
                ##### parameters for log-demons #####
                optical_flow_method: str = 'gauss-newton',
                optical_flow_sigma: float = 0.25,
                optical_flow_eps: float = 1e-3,   # epsilon for minimum gradient value in Thirions, etc.
                update_sigma: float = 0.25,
                eps_prime: float = 0.5,
                use_lie_bracket: bool = False,
                ) -> None:
        # initialize abstract registration
        super().__init__(scales, iterations, fixed_images, moving_images, loss_type, mi_kernel_type, cc_kernel_type, custom_loss,
                         tolerance, max_tolerance_iters, tolerance_mode)
        # log-demons requires that fixed and moving images have the same dimensions for computing optical flow
        assert fixed_images.shape == moving_images.shape, "Fixed and moving images must have the same dimensions"
        self.dims = fixed_images.dims
        # this will be a (D+1, D+1) matrix
        if init_affine is None:
            self.affine = torch.eye(self.dims + 1, device=self.device, dtype=torch.float32).repeat(fixed_images.size(), 1, 1)
        else:
            self.affine = init_affine
        # check for lie bracket
        self.lie_bracket_fn = lie_bracket if use_lie_bracket else lambda *args: 0
        # no optimizer needed for this class
        self.optical_flow = OpticalFlow(optical_flow_method, sigma=optical_flow_sigma, no_grad=True, eps=optical_flow_eps, device=self.device)
        self.eps_prime = eps_prime
        self.update_gaussian = gaussian_1d(torch.tensor(update_sigma, device=self.device), truncated=2, approx='sampled')

    @torch.no_grad()
    def optimize(self, save_transformed=False) -> None:
        '''
        Optimize the log-demons registration given the fixed and moving images
        '''
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]

        # this affine map is the composition of torch2phy and then the affine matrix
        # the grid is in physical space and begins from the affine map, subsequent deformations will be applied to this
        affine_map_fixed2moving = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]
        affine_map_inverse = torch.inverse(affine_map_fixed2moving[:, :, :-1]) # [N, dims, dims]
        init_map = torch.eye(self.dims, self.dims+1, device=self.device).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)

        # keep track of saved images if asked to
        if save_transformed:
            transformed_images = []
        
        velocity_field = None
        scale_factor = 2.0/max(fixed_size) * self.eps_prime
        permute_idx = (0, 2, 3, 4, 1) if self.dims == 3 else (0, 2, 3, 1)
        # Run iterations
        for scale, iters in zip(self.scales, self.iterations):
            tol_ctr = 0
            prev_loss = np.inf
            # downsample fixed array and retrieve coords
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]

            ## create smoothed and downsampled images
            sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device)
            gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
            fixed_image_down = downsample(fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
            moving_image_down = downsample(moving_arrays, size=size_down, mode=self.moving_images.interpolate_mode, gaussians=gaussians)

            ### fixed_image_coords gives the coordinates to the affine map directly
            fixed_image_affinecoords = F.affine_grid(affine_map_fixed2moving, fixed_image_down.shape, align_corners=True)  # [N, H, W, [D], dims]
            fixed_image_vgrid  = F.affine_grid(init_map, fixed_image_down.shape, align_corners=True)                 # [N, H, W, [D], dims]

            # initialize velocity field 
            if velocity_field is None:
                velocity_field = torch.zeros_like(fixed_image_vgrid)  # [N, H, W, [D], dims]
            else:
                velocity_field = F.interpolate(velocity_field.permute(0, 4, 1, 2, 3), fixed_image_vgrid.shape[1:-1], mode=self.fixed_images.interpolate_mode, align_corners=True)
                velocity_field = velocity_field.permute(0, 2, 3, 4, 1)

            pbar = tqdm(range(iters))
            for i in pbar:
                displacement_field = scaling_and_squaring(velocity_field, fixed_image_vgrid, n=6)
                total_displacement_coords = fixed_image_affinecoords + displacement_field
                moved_image = F.grid_sample(moving_image_down, total_displacement_coords, mode='bilinear', align_corners=True)  # [N, C, H, W, [D]]
                v = self.optical_flow(fixed_image_down, moved_image)  # [N, dims, H, W, D]
                v = torch.einsum('nij,nj...->ni...', affine_map_inverse, v)  # [N, dims, H, W, D]
                norm = v.abs().flatten(1).max(1).values[:, None, None, None, None] + 1e-20
                # print(norm)
                v = v / (norm) * scale_factor
                v = v.permute(*permute_idx)  # [N, H, W, D, dims]
                prev_loss = F.mse_loss(moved_image, fixed_image_down)
                # compute update
                velocity_field = velocity_field + v + 0.5 * self.lie_bracket_fn(velocity_field, v)
                velocity_field = separable_filtering(velocity_field, self.update_gaussian)
                pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, prev_loss))

            if save_transformed:
                transformed_images.append(moved_image)

        if save_transformed:
            return transformed_images

